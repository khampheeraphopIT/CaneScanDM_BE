import torch
import torch.nn as nn
from torchvision import models

# Constants
NUM_CLASSES = 6
NUMERICAL_FEATURES_COUNT = 11

class SugarcaneDiseaseModel(nn.Module):
    def __init__(self, num_numerical_features=NUMERICAL_FEATURES_COUNT, num_classes=NUM_CLASSES):
        super(SugarcaneDiseaseModel, self).__init__()
        # Upgrade to ResNet50 for better feature extraction
        self.base_model = models.resnet50(weights='IMAGENET1K_V1')
        
        # Unfreeze more layers for better fine-tuning on specific leaf textures
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Unfreeze layer3 and layer4 for higher-level feature learning
        for param in self.base_model.layer3.parameters():
            param.requires_grad = True
        for param in self.base_model.layer4.parameters():
            param.requires_grad = True
            
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        
        # Improved numerical feature processing
        self.numerical_fc = nn.Sequential(
            nn.Linear(num_numerical_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # Final classification head
        self.final_fc = nn.Sequential(
            nn.Linear(num_ftrs + 32, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, images, numerical):
        img_features = self.base_model(images)
        num_features = self.numerical_fc(numerical)
        combined = torch.cat((img_features, num_features), dim=1)
        output = self.final_fc(combined)
        return output
