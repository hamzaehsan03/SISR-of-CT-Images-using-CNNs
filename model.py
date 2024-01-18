import torch
import os
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt

class SISRDataSet(Dataset):
    def __init__(self, hr_dir, lr_dir, transform=None):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_images = os.listdir(hr_dir)
        self.lr_images = os.listdir(lr_dir)
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.lr_images)
        

    def __getitem__(self, index):
        hr_image_dir = os.path.join(self.hr_dir, self.hr_images[index])
        lr_image_dir = os.path.join(self.lr_dir, self.lr_images[index])

        hr_image = Image.open(hr_image_dir).convert('L')
        lr_image = Image.open(lr_image_dir).convert('L')

        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)
        
        return hr_image, lr_image
        


class SISRCNN(nn.Module):
    def __init__(self):
        super(SISRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
    
current_directory = os.getcwd()
hr_dir = os.path.join(current_directory, "ProcessedImages\\HR\\Train")
lr_dir = os.path.join(current_directory, "ProcessedImages\\LR\\Train")

train_dataset = SISRDataSet(hr_dir=hr_dir, lr_dir=lr_dir)
train_load = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = SISRCNN()
criterion = nn.MSELoss()
optimiser = optim.Adam(model.parameters(), lr=0.001)

# Add checks for CUDA compatability, prefer GPU over CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def show_image_pair(hr_image, lr_image):
    # Convert the tensors to numpy arrays and remove the channel dimension
    if lr_image.dim() == 3:
        lr_image_np = lr_image.squeeze(0).numpy()  # Remove the first dimension
    else:
        lr_image_np = lr_image.numpy()

    if hr_image.dim() == 3:
        hr_image_np = hr_image.squeeze(0).numpy()
    else:
        hr_image_np = hr_image.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(lr_image_np, cmap='gray')
    axes[0].set_title("Low-Resolution Image")
    axes[1].imshow(hr_image_np, cmap='gray')
    axes[1].set_title("High-Resolution Image")
    plt.show()


for i in range(5):
    lr_image, hr_image = train_dataset[i]
    show_image_pair(lr_image, hr_image)

lr_image, hr_image = train_dataset[0]
print("LR Image Type:", type(lr_image), "Shape:", lr_image.shape)
print("HR Image Type:", type(hr_image), "Shape:", hr_image.shape)

print("Number of LR images:", len(train_dataset.lr_images))
print("Number of HR images:", len(train_dataset.hr_images))
print("Dataset length:", len(train_dataset))


num_epochs = 10
# Add training loop
for epoch in range(num_epochs):
    for lr_images, hr_images in train_load:
        # Move images to GPU if available
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)

        # Forward pass
        outputs = model(lr_images)

        # Compute loss
        loss = criterion(outputs, hr_images)

        # Backward pass and optimization
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
