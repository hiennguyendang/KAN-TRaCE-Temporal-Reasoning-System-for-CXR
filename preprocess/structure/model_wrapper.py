from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class BioViLWrapper(nn.Module):
    """
    Wrapper for BiomedVLP-BioViL-T Image Encoder.
    Extracts patch-wise spatial features (196 x 512) and global visual embeddings (128).
    
    If local pre-trained weights are not found, falls back to standard ResNet-50 weights.
    """
    def __init__(self, pretrained_weights_path: str | None = None, pretrained_imagenet: bool = False) -> None:
        super().__init__()
        
        # Load backbone ResNet-50
        if pretrained_weights_path and os.path.exists(pretrained_weights_path):
            print(f"[INFO] Loading visual backbone from custom weights: {pretrained_weights_path}")
            self.backbone = resnet50(weights=None)
        elif pretrained_imagenet:
            try:
                print("[INFO] Initializing with ImageNet pretrained ResNet-50...")
                self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
            except Exception as e:
                print(f"[WARNING] Failed to load ImageNet pretrained weights: {e}. Falling back to randomly initialized ResNet-50.")
                self.backbone = resnet50(weights=None)
        else:
            print("[INFO] Initializing with randomly initialized ResNet-50 (no weight download).")
            self.backbone = resnet50(weights=None)

        # 1x1 Convolution projector to map 2048 feature channels to 512 channels (patch features)
        self.patch_projector = nn.Conv2d(2048, 512, kernel_size=1)
        
        # Global linear projection to map pooled 2048 features to 128 (joint multimodal space)
        self.global_projector = nn.Linear(2048, 128)
        
        # CLS Token linear projection to map pooled 2048 features to 512 (global visual token)
        self.cls_projector = nn.Linear(2048, 512)
        
        if pretrained_weights_path and os.path.exists(pretrained_weights_path):
            self._load_custom_state_dict(pretrained_weights_path)

    def _load_custom_state_dict(self, path: str) -> None:
        try:
            state_dict = torch.load(path, map_location="cpu")
            # Map keys if needed (health_multimodal names might be prefixed)
            mapped_state_dict = {}
            for k, v in state_dict.items():
                new_k = k
                if k.startswith("encoder.backbone."):
                    new_k = k.replace("encoder.backbone.", "backbone.")
                elif k.startswith("encoder."):
                    new_k = k.replace("encoder.", "backbone.")
                elif k.startswith("projector."):
                    if "model.0" in k:
                        new_k = k.replace("projector.model.0", "patch_projector")
                mapped_state_dict[new_k] = v
                
            missing, unexpected = self.load_state_dict(mapped_state_dict, strict=False)
            print(f"[INFO] Custom weights loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        except Exception as e:
            print(f"[WARNING] Failed to load custom weights state dict: {e}. Falling back to initialized weights.")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Input:
            x: (batch_size, 3, H, W) input images.
        Returns:
            patch_embeddings: (batch_size, 196, 512) spatial visual features.
            global_embedding: (batch_size, 128) l2-normalized joint-space embedding.
            cls_token: (batch_size, 512) global visual CLS feature vector.
        """
        # ResNet-50 visual feature extraction up to layer4
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)  # Shape: (batch_size, 2048, H_out, W_out)

        # 1. Spatial Patch Embeddings
        # Map channels: 2048 -> 512
        projected_patches = self.patch_projector(x)  # Shape: (batch_size, 512, H_out, W_out)
        
        # Adaptive pooling to guarantee 14x14 grid (196 patches)
        pooled_patches = F.adaptive_avg_pool2d(projected_patches, (14, 14)) # Shape: (batch_size, 512, 14, 14)
        
        # Flatten spatial grid: (batch_size, 512, 14, 14) -> (batch_size, 512, 196) -> (batch_size, 196, 512)
        batch_size = x.shape[0]
        patch_embeddings = pooled_patches.view(batch_size, 512, 196).transpose(1, 2)
        
        # 2. Global Visual Embedding & CLS Token
        # Average pool across spatial dimensions: (batch_size, 2048, H_out, W_out) -> (batch_size, 2048, 1, 1)
        global_pooled = F.adaptive_avg_pool2d(x, (1, 1)).view(batch_size, -1)  # Shape: (batch_size, 2048)
        
        # Project and normalize: (batch_size, 2048) -> (batch_size, 128)
        global_embedding = self.global_projector(global_pooled)
        global_embedding = F.normalize(global_embedding, p=2, dim=1)

        # 512-dim global CLS token feature
        cls_token = self.cls_projector(global_pooled)  # Shape: (batch_size, 512)
        
        return patch_embeddings, global_embedding, cls_token


if __name__ == "__main__":
    # Test model shape correctness
    print("Testing BioViLWrapper shape correctness...")
    model = BioViLWrapper()
    dummy_input = torch.randn(2, 3, 512, 512)
    patches, global_emb, cls_tok = model(dummy_input)
    print("Input shape: ", dummy_input.shape)
    print("Patch embeddings shape: ", patches.shape, " (Expected: [2, 196, 512])")
    print("Global embedding shape: ", global_emb.shape, " (Expected: [2, 128])")
    print("CLS token shape: ", cls_tok.shape, " (Expected: [2, 512])")
    
    assert patches.shape == (2, 196, 512), "Incorrect patch embeddings shape!"
    assert global_emb.shape == (2, 128), "Incorrect global embedding shape!"
    assert cls_tok.shape == (2, 512), "Incorrect CLS token shape!"
    print("[SUCCESS] Test passed!")

