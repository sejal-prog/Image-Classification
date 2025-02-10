import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic residual block used in ResNet
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        # First convolution layer: Applies 3x3 convolution with stride and padding.
        # This helps extract features from the input image or feature map.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) 
        # Second convolution layer: Further refines the features extracted by the first convolution.
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection setup:
        # If stride is not 1 or input and output dimensions don't match, we use a 1x1 convolution to match the dimensions.
        # This allows the residual connection to add the input to the output directly.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Pass through the first convolution and apply ReLU activation to introduce non-linearity.
        out = F.relu(self.bn1(self.conv1(x)))
        # Second convolution
        out = self.bn2(self.conv2(out))
        # Adding the input to the output (skip connection). This helps mitigate vanishing gradients
        # by allowing the network to preserve low-level features and make learning easier.
        out += self.shortcut(x) 
        out = F.relu(out)  # Apply ReLU to the final output
        return out

# Full ResNet model using basic blocks
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()
        self.in_channels = 64  # Initial number of channels for the first convolution layer

        # First convolution layer: Uses a large kernel to extract initial features from the input image.
        # The input image is expected to have 1 channel (grayscale).
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        self.bn1 = nn.BatchNorm2d(64)  
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Max pooling to reduce size

        # Creating the residual layers (using the block and number of blocks per layer)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Fully connected layer for classification
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        # First block has a stride to reduce the spatial size (downsampling)
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        # Add remaining blocks without changing stride (no downsampling)
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # First convolution + ReLU activation
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)  # Max-pooling to downsample the feature map

        # Pass through all the residual layers (layer1 to layer4)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Apply global average pooling to reduce the feature map to a single value per feature map
        x = self.avg_pool(x)
         # Flatten the output of the global pooling to prepare for the fully connected layer
        x = torch.flatten(x, 1)
        # Fully connected layer to make final class prediction
        x = self.fc(x)
        return x

# ResNet architecture configuration - ResNet-18
def ResNet18(num_classes=2):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

# Initialize model
if __name__ == "__main__":
    model = ResNet18()
    print(model)  # Print model architecture
