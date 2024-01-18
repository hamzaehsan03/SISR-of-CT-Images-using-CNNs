import torch
import os
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim



class SISRDataSet(Dataset):
    def __init__(self, hr_dir, lr_dir, transform=None):
        # ... define dirs + transformation code
        pass

    def __len__(self):
        # ... return dataset code
        pass

    def __getitem__(self, index):
        # ... load + return lr hr pair 
        pass


class SISRCNN(nn.Module):
    def __init__(self):
        super(SISRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        # ... continue
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        # ... continue
        return x
    
current_directory = os.getcwd()
train_dataset = SISRDataSet(hr_dir=(current_directory, "\\ProcessedImages\\HR\\Train"), lr_dir=(current_directory, "\\ProcessedImages\\LR\\Train"))
train_load = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = SISRCNN()
criterion = nn.MSELoss()
optimiser = optim.Adam(model.parameters(), lr=0.001)

# Add training loop


# Add checks for CUDA compatability, prefer GPU over CPU