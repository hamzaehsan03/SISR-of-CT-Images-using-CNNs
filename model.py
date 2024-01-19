import torch
import os
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
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

        # print(f"Loading HR image: {hr_image_dir}, Size: {hr_image.size}")
        # print(f"Loading LR image: {lr_image_dir}, Size: {lr_image.size}")


        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)
            # print(f"Transformed HR Image Shape: {hr_image.shape}")
            # print(f"Transformed LR Image Shape: {lr_image.shape}")
        
        
        return hr_image, lr_image
        

class SISRCNN(nn.Module):
    def __init__(self):
        super(SISRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        # Upsample LR images to match HR
        self.upsample1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.upsample1(x))
        x = self.relu(self.upsample2(x))
        x = self.conv3(x)
        return x

class PerceptualLoss(nn.Module):
    def __init__(self, feature_extractor):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = feature_extractor
    
    def forward(self, resolved_images, hr_images):
        resolved_features = self.feature_extractor(resolved_images)
        hr_features = self.feature_extractor(hr_images)
        loss = nn.functional.mse_loss(resolved_features, hr_features)
        return loss


def main():

    def save_epoch_checkpoint(state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)

    def tensor_to_pil(tensor):
        tensor = tensor.squeeze()
        return transforms.ToPILImage()(tensor)
    
    def show_images(lr_image, hr_image, sr_image):
        lr_image, hr_image, sr_image = [tensor_to_pil(image) for image in [lr_image, hr_image, sr_image]]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(lr_image, cmap='gray')
        axes[0].set_title('Low-Resolution')
        axes[1].imshow(hr_image, cmap='gray')
        axes[1].set_title('High-Resolution')
        axes[2].imshow(sr_image, cmap='gray')
        axes[2].set_title('Super-Resolved')
        for ax in axes:
            ax.axis('off')
        plt.show()

    # Add checks for CUDA compatability, prefer GPU over CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = GradScaler()

    current_directory = os.getcwd()
    hr_dir = os.path.join(current_directory, "ProcessedImages\\HR\\Train")
    lr_dir = os.path.join(current_directory, "ProcessedImages\\LR\\Train")

    train_dataset = SISRDataSet(hr_dir=hr_dir, lr_dir=lr_dir)
    train_load = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=12)

    model = SISRCNN()
    criterion = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=0.001)
    model = model.to(device)

    # Convert VGG19 to use 1 layer instead of RGB
    '''
    model_vgg19 = models.vgg19(pretrained=True).features 
    first_conv_layer = model_vgg19[0]
    model_vgg19[0] = nn.Conv2d(1, first_conv_layer.out_channels, 
                            kernel_size=first_conv_layer.kernel_size, 
                            stride=first_conv_layer.stride, 
                            padding=first_conv_layer.padding)
    model_vgg19 = model_vgg19.to(device)
    model_vgg19.eval()

    perceptual_loss_fn = PerceptualLoss(model_vgg19)
   
    '''
    mse_loss_fn = nn.MSELoss()
    alpha = 1.0
    beta = 0.001

    num_epochs = 10 
    print_every_n_batches = 400  # Print information every n batches
    hr_image, lr_image = train_dataset[0]
    # print("LR Image Shape:", lr_image.shape)  # [1, 128, 128]
    # print("HR Image Shape:", hr_image.shape)  # [1, 512, 512]


    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for batch_idx, (hr_images, lr_images) in enumerate(train_load):
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            with autocast():
                # forward pass
                sr_images = model(lr_images)

                # compute losses
                # print("Shape of SR images:", sr_images.shape)
                # print("Shape of HR images:", hr_images.shape)

                mse_loss = mse_loss_fn(sr_images, hr_images)
                #perceptual_loss = perceptual_loss_fn(sr_images, hr_images)
                total_loss = mse_loss
                    # alpha * mse_loss + beta * perceptual_loss

            # backward pass and optimization
            optimiser.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimiser)
            scaler.update()

            running_loss += total_loss.item()

            if (batch_idx + 1) % print_every_n_batches == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Batch [{batch_idx+1}/{len(train_load)}], "
                    f"Loss: {running_loss / print_every_n_batches:.4f}")
                running_loss = 0.0
                lr_image, hr_image = lr_images[0:1], hr_images[0:1]

                # Move to device and perform inference
                lr_image, hr_image = lr_image.to(device), hr_image.to(device)
                with torch.no_grad():
                    sr_image = model(lr_image)
                    

            save_epoch_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimiser.state_dict()
            }, filename=f"checkpoint_epoch_{epoch+1}.pth.tar")
        
        # Visualize the first image in the batch
        show_images(lr_image.cpu(), hr_image.cpu(), sr_image.cpu())

    print("Training completed")

if __name__ == '__main__':
    main()



    # Test code: check whether the dataloader is working as planned
    # This should be in main, moving it to the bottom for readability

    # def show_image_pair(hr_image, lr_image):
    #     # Convert the tensors to numpy arrays and remove the channel dimension
    #     if lr_image.dim() == 3:
    #         lr_image_np = lr_image.squeeze(0).numpy()  # Remove the first dimension
    #     else:
    #         lr_image_np = lr_image.numpy()

    #     if hr_image.dim() == 3:
    #         hr_image_np = hr_image.squeeze(0).numpy()
    #     else:
    #         hr_image_np = hr_image.numpy()

    #     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    #     axes[0].imshow(lr_image_np, cmap='gray')
    #     axes[0].set_title("Low-Resolution Image")
    #     axes[1].imshow(hr_image_np, cmap='gray')
    #     axes[1].set_title("High-Resolution Image")
    #     plt.show()


    # for i in range(5):
    #     lr_image, hr_image = train_dataset[i]
    #     show_image_pair(lr_image, hr_image)

    # lr_image, hr_image = train_dataset[0]
    # print("LR Image Type:", type(lr_image), "Shape:", lr_image.shape)
    # print("HR Image Type:", type(hr_image), "Shape:", hr_image.shape)

    # print("Number of LR images:", len(train_dataset.lr_images))
    # print("Number of HR images:", len(train_dataset.hr_images))
    # print("Dataset length:", len(train_dataset))