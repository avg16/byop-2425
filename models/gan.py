import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, img_dim=3, hidden_dim=64, latent_dim=100, output_dim=32):
        super(Generator, self).__init__()

        # Encoding the 2D image to a feature representation
        self.img_conv1 = nn.Conv2d(img_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.img_conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        self.img_conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1)
        
        # Fully connected layer for combining image features and noise vector
        self.fc1 = nn.Linear(hidden_dim * 4 * 8 * 8 + latent_dim, hidden_dim * 4 * 4 * 4)

        # 3D Transpose Convolutions to generate the 3D voxel grid
        self.deconv1 = nn.ConvTranspose3d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose3d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose3d(hidden_dim, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, img, z):
        # Encode 2D image to feature representation
        img_features = F.relu(self.img_conv1(img))
        img_features = F.relu(self.img_conv2(img_features))
        img_features = F.relu(self.img_conv3(img_features))
        img_features = img_features.view(img_features.size(0), -1)

        # Concatenate noise vector with encoded image features
        combined = torch.cat([img_features, z], dim=1)
        x = F.relu(self.fc1(combined))
        x = x.view(x.size(0), -1, 4, 4, 4)

        # Generate 3D voxel grid from concatenated features
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))  # Sigmoid for voxel output
        return x
    
class Discriminator(nn.Module):
    def __init__(self, voxel_dim=1, img_dim=3, hidden_dim=64):
        super(Discriminator, self).__init__()

        # Encoding 3D voxel grid
        self.voxel_conv1 = nn.Conv3d(voxel_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.voxel_conv2 = nn.Conv3d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        self.voxel_conv3 = nn.Conv3d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1)

        # Encoding 2D image as a condition
        self.img_conv1 = nn.Conv2d(img_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.img_conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        self.img_conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1)

        # Fully connected layer to combine image and voxel grid features
        self.fc = nn.Linear(hidden_dim * 8 * 4 * 4, 1)

    def forward(self, voxels, img):
        # Encode 3D voxel grid
        voxel_features = F.leaky_relu(self.voxel_conv1(voxels), 0.2)
        voxel_features = F.leaky_relu(self.voxel_conv2(voxel_features), 0.2)
        voxel_features = F.leaky_relu(self.voxel_conv3(voxel_features), 0.2)
        voxel_features = voxel_features.view(voxel_features.size(0), -1)

        # Encode 2D image
        img_features = F.leaky_relu(self.img_conv1(img), 0.2)
        img_features = F.leaky_relu(self.img_conv2(img_features), 0.2)
        img_features = F.leaky_relu(self.img_conv3(img_features), 0.2)
        img_features = img_features.view(img_features.size(0), -1)

        # Concatenate encoded voxel and image features
        combined = torch.cat([voxel_features, img_features], dim=1)
        output = torch.sigmoid(self.fc(combined))  # Sigmoid for binary classification
        return output