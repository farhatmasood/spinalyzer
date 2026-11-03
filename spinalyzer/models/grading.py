"""Disc grading network: DINOv2 backbone with LoRA, ordinal Pfirrmann, stenosis and per-disc pathology heads."""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from peft import LoraConfig, get_peft_model
import timm

class MeasurementEncoder(nn.Module):
    def __init__(
        self,
        num_measurements: int = 12,
        measurement_dim: int = 32,
        hidden_dim: int = 128,
        num_disc_levels: int = 3,
        use_uncertainty: bool = True
    ):
        super().__init__()
        self.use_uncertainty = use_uncertainty
        
        self.level_embedding = nn.Embedding(num_disc_levels, hidden_dim // 4)
        
        self.measurement_encoder = nn.Sequential(
            nn.Linear(num_measurements, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.output_proj = nn.Linear(hidden_dim + hidden_dim // 4, measurement_dim)
        
        if use_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_measurements)
            )
    
    def forward(
        self,
        measurements: torch.Tensor,
        disc_level: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        level_emb = self.level_embedding(disc_level)
        
        encoded = self.measurement_encoder(measurements)
        
        combined = torch.cat([encoded, level_emb], dim=-1)
        
        output = self.output_proj(combined)
        
        result = {"measurement_features": output}
        
        if self.use_uncertainty:
            log_var = self.uncertainty_head(encoded)
            result["measurement_uncertainty"] = torch.exp(0.5 * log_var)
        
        return result

class OrdinalRegressionHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(input_dim, num_classes - 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.fc(x)
        probas = torch.sigmoid(logits)
        
        pred_class = (probas > 0.5).sum(dim=-1)
        
        return logits, pred_class

class MultiTaskGradingHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_pfirrmann_classes: int = 5,
        num_stenosis_classes: int = 3,
        num_disc_levels: int = 3,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.shared_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        self.pfirrmann_head = OrdinalRegressionHead(hidden_dim, num_pfirrmann_classes)
        
        self.stenosis_head = nn.Linear(hidden_dim, num_stenosis_classes)
        
        self.herniation_head = nn.Linear(hidden_dim, 1)
        
        self.pathology_heads = nn.ModuleDict({
            "disc_bulge": nn.Linear(hidden_dim, 1),
            "disc_protrusion": nn.Linear(hidden_dim, 1),
            "disc_extrusion": nn.Linear(hidden_dim, 1),
            "nerve_compression": nn.Linear(hidden_dim, 1),
            "thecal_compression": nn.Linear(hidden_dim, 1),
            "foraminal_narrowing": nn.Linear(hidden_dim, 1),
            "facet_hypertrophy": nn.Linear(hidden_dim, 1),
            "ligamentum_flavum": nn.Linear(hidden_dim, 1)
        })
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared = self.shared_fc(x)
        
        pfirrmann_logits, pfirrmann_pred = self.pfirrmann_head(shared)
        stenosis_logits = self.stenosis_head(shared)
        herniation_logit = self.herniation_head(shared)
        
        pathology_logits = {
            name: head(shared) for name, head in self.pathology_heads.items()
        }
        
        return {
            "pfirrmann_logits": pfirrmann_logits,
            "pfirrmann_pred": pfirrmann_pred,
            "stenosis_logits": stenosis_logits,
            "stenosis_pred": stenosis_logits.argmax(dim=-1),
            "herniation_logit": herniation_logit,
            "pathology_logits": pathology_logits
        }

class GradingAdapter(nn.Module):
    def __init__(
        self,
        backbone_name: str = "facebook/dinov2-base",
        hidden_dim: int = 768,
        adapter_dim: int = 128,
        measurement_dim: int = 32,
        num_pfirrmann_classes: int = 5,
        num_stenosis_classes: int = 3,
        num_disc_levels: int = 3,
        img_size: int = 224,
        freeze_backbone: bool = True,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        use_mc_dropout: bool = True,
        mc_samples: int = 10
    ):
        super().__init__()
        self.use_mc_dropout = use_mc_dropout
        self.mc_samples = mc_samples
        self.hidden_dim = hidden_dim
        
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
                target_modules=["qkv", "proj", "fc1", "fc2"],
                lora_dropout=0.1,
                bias="none"
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
        
        self.domain_adapter = nn.Sequential(
            nn.Linear(hidden_dim, adapter_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(adapter_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.domain_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        self.measurement_encoder = MeasurementEncoder(
            num_measurements=12,
            measurement_dim=measurement_dim,
            hidden_dim=128,
            num_disc_levels=num_disc_levels
        )
        
        self.level_embedding = nn.Embedding(num_disc_levels, hidden_dim // 4)
        
        combined_dim = hidden_dim + measurement_dim + hidden_dim // 4
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        self.grading_head = MultiTaskGradingHead(
            input_dim=hidden_dim,
            num_pfirrmann_classes=num_pfirrmann_classes,
            num_stenosis_classes=num_stenosis_classes,
            num_disc_levels=num_disc_levels
        )
        
        self.mc_dropout = nn.Dropout(0.2)
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        
        features = features + self.domain_embedding
        
        features = self.domain_adapter(features) + features
        
        cls_token = features[:, 0]
        
        return cls_token
    
    def forward(
        self,
        image: torch.Tensor,
        measurements: Optional[torch.Tensor] = None,
        disc_level: Optional[torch.Tensor] = None,
        seg_features: Optional[torch.Tensor] = None,
        return_uncertainty: bool = False
    ) -> Dict[str, torch.Tensor]:
        if return_uncertainty and self.use_mc_dropout:
            return self._forward_with_uncertainty(
                image, measurements, disc_level, seg_features
            )
        
        visual_features = self.extract_features(image)
        
        if measurements is not None and disc_level is not None:
            meas_out = self.measurement_encoder(measurements, disc_level)
            meas_features = meas_out["measurement_features"]
            level_emb = self.level_embedding(disc_level)
            
            combined = torch.cat([visual_features, meas_features, level_emb], dim=-1)
            fused = self.feature_fusion(combined)
        else:
            level_emb = torch.zeros(image.size(0), self.hidden_dim // 4, device=image.device)
            meas_features = torch.zeros(image.size(0), 32, device=image.device)
            combined = torch.cat([visual_features, meas_features, level_emb], dim=-1)
            fused = self.feature_fusion(combined)
        
        if seg_features is not None:
            fused = fused + 0.1 * seg_features.mean(dim=1)
        
        grades = self.grading_head(fused)
        grades["visual_features"] = visual_features
        
        return grades
    
    def _forward_with_uncertainty(
        self,
        image: torch.Tensor,
        measurements: Optional[torch.Tensor],
        disc_level: Optional[torch.Tensor],
        seg_features: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        self.train()
        
        pfirrmann_preds = []
        stenosis_preds = []
        
        for _ in range(self.mc_samples):
            with torch.no_grad():
                visual_features = self.extract_features(image)
                visual_features = self.mc_dropout(visual_features)
                
                if measurements is not None and disc_level is not None:
                    meas_out = self.measurement_encoder(measurements, disc_level)
                    meas_features = meas_out["measurement_features"]
                    level_emb = self.level_embedding(disc_level)
                    combined = torch.cat([visual_features, meas_features, level_emb], dim=-1)
                else:
                    level_emb = torch.zeros(image.size(0), self.hidden_dim // 4, device=image.device)
                    meas_features = torch.zeros(image.size(0), 32, device=image.device)
                    combined = torch.cat([visual_features, meas_features, level_emb], dim=-1)
                
                fused = self.feature_fusion(combined)
                
                grades = self.grading_head(fused)
                
                pfirrmann_preds.append(grades["pfirrmann_pred"].float())
                stenosis_preds.append(grades["stenosis_pred"].float())
        
        pfirrmann_stack = torch.stack(pfirrmann_preds, dim=0)
        stenosis_stack = torch.stack(stenosis_preds, dim=0)
        
        return {
            "pfirrmann_pred": pfirrmann_stack.mean(dim=0).round().long(),
            "pfirrmann_uncertainty": pfirrmann_stack.std(dim=0),
            "stenosis_pred": stenosis_stack.mean(dim=0).round().long(),
            "stenosis_uncertainty": stenosis_stack.std(dim=0)
        }
    
    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
