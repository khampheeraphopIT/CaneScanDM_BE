import torch
import torch.nn as nn
from torchvision import models

# Constants
NUM_CLASSES = 6
NUMERICAL_FEATURES_COUNT = 11

class SugarcaneDiseaseModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SugarcaneDiseaseModel, self).__init__()
        # Using ResNet50 as the backbone
        self.base_model = models.resnet50(weights='IMAGENET1K_V1')
        
        # Unfreeze for fine-tuning to recover performance
        for param in self.base_model.parameters():
            param.requires_grad = True
            
        num_ftrs = self.base_model.fc.in_features
        
        # Simple, robust classification head
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, numerical=None):
        # numerical argument kept for compatibility during transition, but ignored
        return self.base_model(images)

