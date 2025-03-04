import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T
from xmipp_metadata.metadata import XmippMetaData
from xmipp_metadata.image_handler import ImageHandler
from CUSTOM_DATASET import MetadataDataset
from Volume_projection import VolumeProjection
from preprocessing_data import Preprocesing
from WrongVolumeProjection import WrongVolumeProjection
import math 

# Hyperparameters
LEARNING_RATE = 1e-3
EPOCHS = 5
#BATCH_SIZE= 50 #length 

def mrcs_dataloader(metadata_path, image_file, volume_path, batch_size, train_split):
    # Load dataset
    #dataset = np.load(file_path)  # Assuming shape (n_samples, height, width)
    #with this dataset we are loading the metadata for 
    dataset = MetadataDataset(metadata_path, image_file)
    #this returns  'angles' (euler angles) and 'metadata'
    
    # Split dataset 
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True) #batch 1/2 length
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)
        
    return train_loader, test_loader

class BottleneckBlock(nn.Module):
    expansion = 4  # Increases channel depth

    def __init__(self, in_channels, out_channels, stride=1, downsample=True): 
        super(BottleneckBlock, self).__init__()
        
        # Reduced channels for the 3x3 conv
        #print(out_channels)
        reduced_channels = out_channels // self.expansion
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(reduced_channels)
        
        self.conv2 = nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3, 
                                stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(reduced_channels)
        
        self.conv3 = nn.Conv2d(reduced_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Downsampling layer for matching dimensions
        #self.downsample = downsample
        #self.downsample = downsample or nn.Sequential(
        #    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        #    nn.BatchNorm2d(out_channels)
        #)
        #self.downsample = T.Resize(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # First 1x1 convolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 3x3 convolution
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # Final 1x1 convolution
        out = self.conv3(out)
        out = self.bn3(out)


        # Handle downsampling if needed
        if self.downsample:
            # identity = self.downsample(x)
            identity = T.Resize(out.shape[-1])(x)
        


        # Residual connection
        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # Initial convolution
        #aqui antes ponia channel 1 no 50 
        self.conv1 = nn.Conv2d(1, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding="same", bias=False)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding="same", bias=False)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        # Residual layers
        self.layer1 = self._make_layer(block, self.in_channels, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, 512, layers[3], stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)  
       # self.fc = nn.Sequential(
        #    nn.Linear(512, 1),
     #  nn.Sigmoid()
    #    )

        # Weight initialization
        self._initialize_weights()

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):

        downsample = False
        if stride != 1: #or self.in_channels != out_channels * block.expansion:
            downsample = True
        #    downsample = nn.Sequential(
        #        nn.Conv2d(self.in_channels, out_channels * block.expansion, 
        #                  kernel_size=1, stride=stride, bias=False),
        #        nn.BatchNorm2d(out_channels * block.expansion)
        #    )

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        # self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if x.dim() == 3:
        # If x is (N, H, W), add a channel dimension
            x = x.unsqueeze(1)
        elif x.dim() == 2:
            # If x is (N, D), reshape to (N, 1, H, W)
            x = x.view(x.size(0), 1, int(math.sqrt(x.size(1))), int(math.sqrt(x.size(1))))
        elif x.dim() == 4:
            # If x is already (N, C, H, W), ensure C=1
            if x.size(1) != 1:
                raise ValueError("Expected input with 1 channel, but got {} channels.".format(x.size(1)))
        else:
            raise ValueError("Unsupported input dimensions: {}".format(x.shape))


        # Initial convolution layers
        x = self.conv1(x)  #makes [64,25,8192]
        x = self.bn1(x)
        x = self.relu(x)

        # Residual layers
        x = self.layer1(x)
        x = self.conv2(x)
        x = self.layer2(x)
        x = self.conv3(x)
        x = self.layer3(x)
        x = self.conv4(x)
        x = self.layer4(x)

        # Global pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
def train_and_validate(model, train_loader, test_loader, metadata_path, volume_path,
                       epochs=EPOCHS, 
                       learning_rate=LEARNING_RATE, 
                       device=None):

    # Device configuration
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy for binary classification
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

    # Training history tracking
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_accuracy': []
    }

    preprocessor=Preprocesing(metadata_path, volume_path)

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        
        for batch_idx, batch_data in enumerate(train_loader):
            # get metadata and indices from the batch
            metadata = batch_data['metadata']
            # Move data to device
            #inputs = inputs.to(device)
            #targets = targets.float().to(device)
            #preprocesamiento 

            subs_tensor, labels_tensor = preprocessor.process_data(metadata)

              # Move data to device
            inputs = subs_tensor.to(device)
            targets = labels_tensor.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs).squeeze()
            # Normalize targets to [0,1] range instead of adding 1
            outputs = torch.sigmoid(outputs)
           # targets_normalized = targets.float() / targets.max()  
            loss = criterion(outputs, targets)
            #outputs = torch.sigmoid(outputs)
            
            # Compute loss
           # loss = criterion(outputs, targets + 1)
            #loss = torch.mean(torch.abs(outputs - targets))
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Compute average training loss
        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation phase
        model.eval()
        total_test_loss = 0.0
        correct_predictions = 0
        total_samples = 0
    
        with torch.no_grad():
            for batch_data in test_loader:
            
                metadata = batch_data['metadata']
                subs_tensor, labels_tensor = preprocessor.process_data(metadata)
                
                # Move data to device
                inputs = subs_tensor.to(device)
                targets = labels_tensor.to(device)

                # Forward pass
                outputs = model(inputs).squeeze()
                outputs = torch.sigmoid(outputs)
                
                # Compute loss
                loss = criterion(outputs, targets)
                total_test_loss += loss.item()

                # Compute accuracy
                predictions = (outputs > 0.5).float()
                correct_predictions += (predictions == targets).float().sum().item()
                total_samples += targets.size(0)

        # Compute metrics
        avg_test_loss = total_test_loss / len(test_loader)
        test_accuracy = correct_predictions / total_samples * 100

        # Store metrics
        history['test_loss'].append(avg_test_loss)
        history['test_accuracy'].append(test_accuracy)

        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Test Loss: {avg_test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.2f}%\n")

    return history

def main():
    # Load data
    train_loader, test_loader = mrcs_dataloader('metadata.xmd', 'scaled_particles.stk','volume.mrc',50,0.8)
    
    # Initialize model
    model = ResNet(block=BottleneckBlock, layers=[3, 4, 6, 3], )
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
   #train and validate
    history = train_and_validate(
        model, 
        train_loader, 
        test_loader, 
        'metadata.xmd',
        'volume.mrc',
        epochs=EPOCHS, 
        learning_rate=LEARNING_RATE
    )

if __name__ == "__main__":
    main()