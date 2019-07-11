## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # Covolutional Layers
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 2)
        
        # Maxpooling Layer
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        # Fully Connected Layers
        # The number of input gained by "print("Flatten size: ", x.shape)" in below
        self.fc1 = nn.Linear(36864, 1000) 
        self.fc2 = nn.Linear(1000, 1000)
        # the output 136 in order to having 2 for each of the 68 keypoint (x, y) pairs
        self.fc3 = nn.Linear(1000, 136) 
        
        # Dropout layers
        self.drop1 = nn.Dropout(p = 0.1)
        self.drop2 = nn.Dropout(p = 0.2)
        self.drop3 = nn.Dropout(p = 0.3)
        self.drop4 = nn.Dropout(p = 0.4)
        self.drop5 = nn.Dropout(p = 0.5)
        self.drop6 = nn.Dropout(p = 0.6)


        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # First - Convolution + Activation + Pooling + Dropout
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.drop1(x)
        #print("First layer size: ", x.shape)

        # Second
        x = self.drop2(self.pool(F.relu(self.conv2(x))))

        # Third
        x = self.drop3(self.pool(F.relu(self.conv3(x))))

        # Forth
        x = self.drop4(self.pool(F.relu(self.conv4(x))))

        # Flattening the layer
        x = x.view(x.size(0), -1)

        # First Fully-connected Layer - Dense + Activation + Dropout
        x = self.drop5(F.relu(self.fc1(x)))

        # Second
        x = self.drop6(F.relu(self.fc2(x)))

        # Final Dense
        x = self.fc3(x)

        return x
        
