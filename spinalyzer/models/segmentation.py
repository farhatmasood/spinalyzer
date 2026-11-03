"""Segmentation network: DINOv2 ViT-B/14 with LoRA and bottleneck adapters, plane-specific decoders."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from peft import LoraConfig, get_peft_model
import timm

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))

class BottleneckAdapter(nn.Module):
    def __init__(self, hidden_dim: int, adapter_dim: int, dropout: float = 0.1):
        super().__init__()
        self.down = nn.Linear(hidden_dim, adapter_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(adapter_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.down(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.up(x)
        return residual + self.scale * x

class SpatialAdapter(nn.Module):
    def __init__(self, channels: int, adapter_dim: int, spatial_size: int = 24):
        super().__init__()
        self.spatial_down = nn.Conv2d(channels, adapter_dim, 3, 1, 1)
        self.spatial_act = nn.GELU()
        self.spatial_up = nn.Conv2d(adapter_dim, channels, 3, 1, 1)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)
        
        adapted = self.spatial_down(x_2d)
        adapted = self.spatial_act(adapted)
        adapted = self.spatial_up(adapted)
        
        adapted = adapted.flatten(2).transpose(1, 2)
        return x + self.scale * adapted

class CrossPlaneAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        B, N, C = query.shape
        
        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        
        out = self.norm1(query + out)
        out = self.norm2(out + self.ffn(out))
        
        return out

class MultiPlaneFusionAdapter(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 768,
        adapter_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.axial_adapters = nn.ModuleList([
            BottleneckAdapter(hidden_dim, adapter_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.sagittal_adapters = nn.ModuleList([
            BottleneckAdapter(hidden_dim, adapter_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.cross_attention_ax_to_sag = nn.ModuleList([
            CrossPlaneAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.cross_attention_sag_to_ax = nn.ModuleList([
            CrossPlaneAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(
        self,
        axial_features: torch.Tensor,
        sagittal_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        ax = axial_features
        sag = sagittal_features
        
        for i in range(len(self.axial_adapters)):
            ax = self.axial_adapters[i](ax)
            sag = self.sagittal_adapters[i](sag)
            
            ax_cross = self.cross_attention_ax_to_sag[i](ax, sag)
            sag_cross = self.cross_attention_sag_to_ax[i](sag, ax)
            
            gate_ax = self.fusion_gate(torch.cat([ax, ax_cross], dim=-1))
            gate_sag = self.fusion_gate(torch.cat([sag, sag_cross], dim=-1))
            
            ax = ax + gate_ax * (ax_cross - ax)
            sag = sag + gate_sag * (sag_cross - sag)
        
        ax_global = self.global_pool(ax.transpose(1, 2)).squeeze(-1)
        sag_global = self.global_pool(sag.transpose(1, 2)).squeeze(-1)
        global_features = torch.cat([ax_global, sag_global], dim=-1)
        
        return {
            "axial": ax,
            "sagittal": sag,
            "global": global_features
        }

class SegmentationDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 768,
        num_classes: int = 7,
        decoder_channels: List[int] = [256, 128, 64, 32],
        img_size: int = 384,
        patch_size: int = 14
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.feat_size = img_size // patch_size
        
        self.proj = nn.Conv2d(hidden_dim, decoder_channels[0], 1)
        
        self.decoder_blocks = nn.ModuleList()
        in_ch = decoder_channels[0]
        for out_ch in decoder_channels[1:]:
            self.decoder_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 2, 2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                ConvBNReLU(out_ch, out_ch, 3),
                ConvBNReLU(out_ch, out_ch, 3)
            ))
            in_ch = out_ch
        
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels[-1], decoder_channels[-1], 2, 2),
            nn.BatchNorm2d(decoder_channels[-1]),
            nn.ReLU(inplace=True)
        )
        
        self.seg_head = nn.Conv2d(decoder_channels[-1], num_classes, 1)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, N, C = features.shape
        
        if N == self.feat_size * self.feat_size + 1:
            features = features[:, 1:, :]
            N = N - 1
        
        H = W = int(N ** 0.5)
        
        x = features.transpose(1, 2).reshape(B, C, H, W)
        x = self.proj(x)
        
        for block in self.decoder_blocks:
            x = block(x)
        
        x = self.final_upsample(x)
        
        x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=False)
        
        return self.seg_head(x)

class SegmentationAdapter(nn.Module):
    def __init__(
        self,
        backbone_name: str = "facebook/dinov2-base",
        hidden_dim: int = 768,
        adapter_dim: int = 64,
        num_adapter_layers: int = 4,
        num_axial_classes: int = 3,
        num_sagittal_classes: int = 4,
        img_size: int = 384,
        patch_size: int = 14,
        freeze_backbone: bool = True,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32
    ):
        super().__init__()
        
        self.backbone = timm.create_model(
            "vit_base_patch14_dinov2.lvd142m",
            pretrained=False,
            img_size=img_size,
            num_classes=0
        )
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        if use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["qkv", "proj"],
                lora_dropout=0.05,
                bias="none"
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
        
        self.axial_adapter = nn.ModuleList([
            BottleneckAdapter(hidden_dim, adapter_dim)
            for _ in range(num_adapter_layers)
        ])
        
        self.sagittal_adapter = nn.ModuleList([
            BottleneckAdapter(hidden_dim, adapter_dim)
            for _ in range(num_adapter_layers)
        ])
        
        self.fusion_adapter = MultiPlaneFusionAdapter(
            hidden_dim=hidden_dim,
            adapter_dim=adapter_dim,
            num_layers=3,
            num_heads=8
        )
        
        self.axial_decoder = SegmentationDecoder(
            hidden_dim=hidden_dim,
            num_classes=num_axial_classes,
            img_size=img_size,
            patch_size=patch_size
        )
        
        self.sagittal_decoder = SegmentationDecoder(
            hidden_dim=hidden_dim,
            num_classes=num_sagittal_classes,
            img_size=img_size,
            patch_size=patch_size
        )
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)
    
    def forward_axial(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(x)
        
        for adapter in self.axial_adapter:
            features = adapter(features)
        
        masks = self.axial_decoder(features)
        return masks, features
    
    def forward_sagittal(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(x)
        
        for adapter in self.sagittal_adapter:
            features = adapter(features)
        
        masks = self.sagittal_decoder(features)
        return masks, features
    
    def forward(
        self,
        axial_img: torch.Tensor,
        sagittal_img: Optional[torch.Tensor] = None,
        plane: str = "both"
    ) -> Dict[str, torch.Tensor]:
        results = {}
        
        if plane in ["axial", "both"]:
            ax_features = self.extract_features(axial_img)
            for adapter in self.axial_adapter:
                ax_features = adapter(ax_features)
            results["axial_features"] = ax_features
            
        if plane in ["sagittal", "both"] and sagittal_img is not None:
            sag_features = self.extract_features(sagittal_img)
            for adapter in self.sagittal_adapter:
                sag_features = adapter(sag_features)
            results["sagittal_features"] = sag_features
        
        if plane == "both" and sagittal_img is not None:
            fused = self.fusion_adapter(
                results["axial_features"],
                results["sagittal_features"]
            )
            results["axial_fused"] = fused["axial"]
            results["sagittal_fused"] = fused["sagittal"]
            results["global_features"] = fused["global"]
            
            results["axial_masks"] = self.axial_decoder(fused["axial"])
            results["sagittal_masks"] = self.sagittal_decoder(fused["sagittal"])
        else:
            if "axial_features" in results:
                results["axial_masks"] = self.axial_decoder(results["axial_features"])
            if "sagittal_features" in results:
                results["sagittal_masks"] = self.sagittal_decoder(results["sagittal_features"])
        
        return results
    
    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
