import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
# class Generator(nn.Module):
#     def __init__(self, img_dim=3, hidden_dim=64, latent_dim=100, output_dim=128):
#         super(Generator, self).__init__()

#         self.img_conv1 = nn.Conv2d(img_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
#         self.img_conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
#         self.img_conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1)
#         self.img_conv4 = nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2, padding=1)
        
#         self.fc1 = nn.Linear(hidden_dim * 8 * 8 * 8 + latent_dim, hidden_dim * 16 * 4 * 4 * 4)

#         self.deconv1 = nn.ConvTranspose3d(hidden_dim * 16, hidden_dim * 8, kernel_size=4, stride=2, padding=1)
#         self.deconv2 = nn.ConvTranspose3d(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2, padding=1)
#         self.deconv3 = nn.ConvTranspose3d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
#         self.deconv4 = nn.ConvTranspose3d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1)
#         self.deconv5 = nn.ConvTranspose3d(hidden_dim, 1, kernel_size=4, stride=2, padding=1)

#     def forward(self, img, z):
#         img_features = F.leaky_relu(self.img_conv1(img), 0.2)
#         img_features = F.leaky_relu(self.img_conv2(img_features), 0.2)
#         img_features = F.leaky_relu(self.img_conv3(img_features), 0.2)
#         img_features = F.leaky_relu(self.img_conv4(img_features), 0.2)
#         img_features = img_features.view(img_features.size(0), -1)

#         combined = torch.cat([img_features, z], dim=1)
#         x = F.leaky_relu(self.fc1(combined), 0.2)
#         x = x.view(x.size(0), -1, 4, 4, 4)

#         x = F.leaky_relu(self.deconv1(x), 0.2)
#         x = F.leaky_relu(self.deconv2(x), 0.2)
#         x = F.leaky_relu(self.deconv3(x), 0.2)
#         x = F.leaky_relu(self.deconv4(x), 0.2)
#         x = torch.sigmoid(self.deconv5(x))
#         return x
class Generator(nn.Module):
    def __init__(self, img_dim=3, hidden_dim=64, latent_dim=100, output_dim=128):
        super(Generator, self).__init__()

        # Image branch
        self.img_conv1 = nn.Conv2d(img_dim, hidden_dim, kernel_size=4, stride=2, padding=1)  # [4, 64, 64, 64]
        self.img_conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1)  # [4, 128, 32, 32]
        self.img_conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1)  # [4, 256, 16, 16]
        self.img_conv4 = nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2, padding=1)  # [4, 512, 8, 8]

        # Fully connected layer, adjusted to take img_features + latent vector
        self.fc1 = nn.Linear(hidden_dim * 8 * 8 * 8 + latent_dim, hidden_dim * 16 * 4 * 4 * 4)

        # Transposed convolutions to upscale to [batch_size, 1, 128, 128, 128]
        self.deconv1 = nn.ConvTranspose3d(hidden_dim * 16, hidden_dim * 8, kernel_size=4, stride=2, padding=1)  # [4, 512, 8, 8, 8]
        self.deconv2 = nn.ConvTranspose3d(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2, padding=1)   # [4, 256, 16, 16, 16]
        self.deconv3 = nn.ConvTranspose3d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1)   # [4, 128, 32, 32, 32]
        self.deconv4 = nn.ConvTranspose3d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1)       # [4, 64, 64, 64, 64]
        self.deconv5 = nn.ConvTranspose3d(hidden_dim, 1, kernel_size=4, stride=2, padding=1)                    # [4, 1, 128, 128, 128]

    def forward(self, img, z):
        # Process the image input through conv layers
        img_features = F.leaky_relu(self.img_conv1(img), 0.2)
        img_features = F.leaky_relu(self.img_conv2(img_features), 0.2)
        img_features = F.leaky_relu(self.img_conv3(img_features), 0.2)
        img_features = F.leaky_relu(self.img_conv4(img_features), 0.2)
        img_features = img_features.view(img_features.size(0), -1)  # Flatten

        # Concatenate img_features with latent vector z
        combined = torch.cat([img_features, z], dim=1)
        x = F.leaky_relu(self.fc1(combined), 0.2)
        x = x.view(x.size(0), -1, 4, 4, 4)  # Reshape for 3D transpose convs

        # Upsample through transpose conv layers
        x = F.leaky_relu(self.deconv1(x), 0.2)
        x = F.leaky_relu(self.deconv2(x), 0.2)
        x = F.leaky_relu(self.deconv3(x), 0.2)
        x = F.leaky_relu(self.deconv4(x), 0.2)
        x = torch.sigmoid(self.deconv5(x))  # Output voxel grid
        return x


# class Discriminator(nn.Module):
#     def __init__(self, voxel_dim=1, img_dim=3, hidden_dim=64):
#         super(Discriminator, self).__init__()

#         self.voxel_conv1 = nn.Conv3d(voxel_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
#         self.voxel_conv2 = nn.Conv3d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
#         self.voxel_conv3 = nn.Conv3d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1)
#         self.voxel_conv4 = nn.Conv3d(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2, padding=1)

#         self.img_conv1 = nn.Conv2d(img_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
#         self.img_conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
#         self.img_conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1)
#         self.img_conv4 = nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2, padding=1)

#         self.fc = nn.Linear(hidden_dim * 8 * 8 * 8 + hidden_dim * 8 * 8 * 8, 1)

#     def forward(self, voxels, img):
#         voxel_features = F.leaky_relu(self.voxel_conv1(voxels), 0.2)
#         voxel_features = F.leaky_relu(self.voxel_conv2(voxel_features), 0.2)
#         voxel_features = F.leaky_relu(self.voxel_conv3(voxel_features), 0.2)
#         voxel_features = F.leaky_relu(self.voxel_conv4(voxel_features), 0.2)
#         voxel_features = voxel_features.view(voxel_features.size(0), -1)

#         img_features = F.leaky_relu(self.img_conv1(img), 0.2)
#         img_features = F.leaky_relu(self.img_conv2(img_features), 0.2)
#         img_features = F.leaky_relu(self.img_conv3(img_features), 0.2)
#         img_features = F.leaky_relu(self.img_conv4(img_features), 0.2)
#         img_features = img_features.view(img_features.size(0), -1)

#         combined = torch.cat([voxel_features, img_features], dim=1)
#         output = torch.sigmoid(self.fc(combined))
#         return output

class Discriminator(nn.Module):
    def __init__(self, voxel_dim=1, img_dim=3, hidden_dim=64):
        super(Discriminator, self).__init__()

        # Voxel branch (3D Convolutions)
        self.voxel_conv1 = nn.Conv3d(voxel_dim, hidden_dim, kernel_size=4, stride=2, padding=1)  # [4, 64, 64, 64, 64]
        self.voxel_conv2 = nn.Conv3d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1)  # [4, 128, 32, 32, 32]
        self.voxel_conv3 = nn.Conv3d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1)  # [4, 256, 16, 16, 16]
        self.voxel_conv4 = nn.Conv3d(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2, padding=1)  # [4, 512, 8, 8, 8]

        # Image branch (2D Convolutions)
        self.img_conv1 = nn.Conv2d(img_dim, hidden_dim, kernel_size=4, stride=2, padding=1)  # [4, 64, 64, 64]
        self.img_conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1)  # [4, 128, 32, 32]
        self.img_conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1)  # [4, 256, 16, 16]
        self.img_conv4 = nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2, padding=1)  # [4, 512, 8, 8]

        # Fully connected layer input dimension updated for batch size of 4
        self.fc = nn.Linear(hidden_dim * 8 * 8 * 8 * 8 + hidden_dim * 8 * 8 * 8, 1)  # [4, 1]

    def forward(self, voxels, img):
        # Voxel branch
        voxel_features = F.leaky_relu(self.voxel_conv1(voxels), 0.2)
        voxel_features = F.leaky_relu(self.voxel_conv2(voxel_features), 0.2)
        voxel_features = F.leaky_relu(self.voxel_conv3(voxel_features), 0.2)
        voxel_features = F.leaky_relu(self.voxel_conv4(voxel_features), 0.2)
        voxel_features = voxel_features.view(voxel_features.size(0), -1)  # Flatten

        # Image branch
        img_features = F.leaky_relu(self.img_conv1(img), 0.2)
        img_features = F.leaky_relu(self.img_conv2(img_features), 0.2)
        img_features = F.leaky_relu(self.img_conv3(img_features), 0.2)
        img_features = F.leaky_relu(self.img_conv4(img_features), 0.2)
        img_features = img_features.view(img_features.size(0), -1)  # Flatten

        # Concatenate features and pass through fully connected layer
        combined = torch.cat([voxel_features, img_features], dim=1)
        output = torch.sigmoid(self.fc(combined))
        return output

