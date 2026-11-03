"""Structured findings network: ResNet-18 disc encoder with attention aggregation across three disc levels."""
import torch
import torch.nn as nn
from typing import Dict, Optional
import timm


class DiscFeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = False,
        freeze_backbone: bool = True
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )
        
        self.feature_dim = self.backbone.num_features
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.get_classifier().parameters():
                param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class MultiDiscAggregator(nn.Module):
    def __init__(
        self,
        disc_feature_dim: int,
        num_discs: int = 3,
        hidden_dim: int = 256,
        num_pfirrmann_classes: int = 5
    ):
        super().__init__()
        self.num_discs = num_discs
        
        self.pfirrmann_embedding = nn.Embedding(num_pfirrmann_classes, 32)
        
        self.disc_projection = nn.Sequential(
            nn.Linear(disc_feature_dim + 32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.disc_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        self.aggregation = nn.Sequential(
            nn.Linear(hidden_dim * num_discs, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(
        self,
        disc_features: torch.Tensor,
        pfirrmann_grades: torch.Tensor
    ) -> torch.Tensor:
        B, D, F = disc_features.shape
        
        grade_embeds = self.pfirrmann_embedding(pfirrmann_grades)
        
        combined = torch.cat([disc_features, grade_embeds], dim=-1)
        
        projected = self.disc_projection(combined)
        
        attended, _ = self.disc_attention(projected, projected, projected)
        
        flattened = attended.reshape(B, -1)
        
        aggregated = self.aggregation(flattened)
        
        return aggregated


class StructuredFindingsAdapter(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet18",
        hidden_dim: int = 256,
        num_binary_labels: int = 36,
        num_disc_types: int = 6,
        num_severity_classes: int = 3,
        num_pfirrmann_classes: int = 5,
        freeze_backbone: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.disc_encoder = DiscFeatureExtractor(
            backbone_name=backbone_name,
            pretrained=False,
            freeze_backbone=freeze_backbone
        )
        
        self.aggregator = MultiDiscAggregator(
            disc_feature_dim=self.disc_encoder.feature_dim,
            num_discs=3,
            hidden_dim=hidden_dim,
            num_pfirrmann_classes=num_pfirrmann_classes
        )
        
        self.binary_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_binary_labels)
        )
        
        self.l4l5_type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_disc_types)
        )
        
        self.l5s1_type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_disc_types)
        )
        
        self.severity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_severity_classes)
        )
        
        self.num_binary_labels = num_binary_labels
        self.num_disc_types = num_disc_types
        self.num_severity_classes = num_severity_classes
    
    def forward(
        self,
        disc_crops: torch.Tensor,
        pfirrmann_grades: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        B, D, C, H, W = disc_crops.shape
        
        disc_crops_flat = disc_crops.view(B * D, C, H, W)
        disc_features_flat = self.disc_encoder(disc_crops_flat)
        disc_features = disc_features_flat.view(B, D, -1)
        
        aggregated = self.aggregator(disc_features, pfirrmann_grades)
        
        binary_logits = self.binary_head(aggregated)
        l4l5_logits = self.l4l5_type_head(aggregated)
        l5s1_logits = self.l5s1_type_head(aggregated)
        severity_logits = self.severity_head(aggregated)
        
        return {
            "binary_logits": binary_logits,
            "l4l5_type_logits": l4l5_logits,
            "l5s1_type_logits": l5s1_logits,
            "severity_logits": severity_logits,
            "aggregated_features": aggregated
        }
    
    def predict(
        self,
        disc_crops: torch.Tensor,
        pfirrmann_grades: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        outputs = self.forward(disc_crops, pfirrmann_grades)
        
        binary_preds = (torch.sigmoid(outputs["binary_logits"]) > 0.5).float()
        l4l5_preds = outputs["l4l5_type_logits"].argmax(dim=-1)
        l5s1_preds = outputs["l5s1_type_logits"].argmax(dim=-1)
        severity_preds = outputs["severity_logits"].argmax(dim=-1)
        
        return {
            "binary_preds": binary_preds,
            "l4l5_type_pred": l4l5_preds,
            "l5s1_type_pred": l5s1_preds,
            "severity_pred": severity_preds
        }
    
    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class StructuredFindingsLoss(nn.Module):
    def __init__(
        self,
        binary_weight: float = 1.0,
        disc_type_weight: float = 0.5,
        severity_weight: float = 0.5,
        pos_weight: Optional[torch.Tensor] = None,
        severity_class_weight: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.binary_weight = binary_weight
        self.disc_type_weight = disc_type_weight
        self.severity_weight = severity_weight

        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.ce_loss = nn.CrossEntropyLoss()
        self.severity_ce_loss = nn.CrossEntropyLoss(weight=severity_class_weight)
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        binary_loss = self.bce_loss(
            outputs["binary_logits"],
            targets["binary_labels"]
        )
        
        l4l5_loss = self.ce_loss(
            outputs["l4l5_type_logits"],
            targets["l4l5_disc_type"]
        )
        
        l5s1_loss = self.ce_loss(
            outputs["l5s1_type_logits"],
            targets["l5s1_disc_type"]
        )
        
        severity_loss = self.severity_ce_loss(
            outputs["severity_logits"],
            targets["severity"]
        )
        
        disc_type_loss = (l4l5_loss + l5s1_loss) / 2
        
        total_loss = (
            self.binary_weight * binary_loss +
            self.disc_type_weight * disc_type_loss +
            self.severity_weight * severity_loss
        )
        
        return {
            "loss": total_loss,
            "binary_loss": binary_loss,
            "l4l5_type_loss": l4l5_loss,
            "l5s1_type_loss": l5s1_loss,
            "severity_loss": severity_loss
        }
