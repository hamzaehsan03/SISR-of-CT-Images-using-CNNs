import torch
import os
import random
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.nn import functional
import torch.nn as nn
import torch.optim as optim
import pytorch_ssim
from PIL import Image
import matplotlib.pyplot as plt


# TD: Fix artifacting in SR images 
#       -> lr scheduler, higher epochs, refine loss function (vgg based?), add more layers to cnn, regularisation (dropout/weight decay)
#       -> experiment with larger models, try fitting images better (based on patient as opposed to flat dirs)

def gaussian_blur(image, probability = 0.5, max_dev = 0.5):
    if random.random() < probability:
        std_dev = random.uniform(0, max_dev)
        return transforms.functional.gaussian_blur(image, kernel_size=3, sigma=std_dev)
    return image

class SISRDataSet(Dataset):
    def __init__(self, hr_dir, lr_dir, transform=None, training=False):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_images = os.listdir(hr_dir)
        self.lr_images = os.listdir(lr_dir)
        self.transform = transforms.ToTensor()
        self.training = training

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

            if self.training:
                lr_image = gaussian_blur(lr_image)
        
        
        return hr_image, lr_image
        

class SISRCNN(nn.Module):
    def __init__(self):
        super(SISRCNN, self).__init__()
        self.relu = nn.ReLU()
        self.scale_factor = 2
        
        # First block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        #self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        #self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        #self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        #self.bn4 = nn.BatchNorm2d(32)
        
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        #self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 32 * (self.scale_factor ** 2), kernel_size=3, padding=1)
       # self.bn6 = nn.BatchNorm2d(32 * (self.scale_factor ** 2))
        self.pixel_shuffle = nn.PixelShuffle(self.scale_factor)
        # Second block
        self.conv7 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        #self.bn7 = nn.BatchNorm2d(32)
        self.conv8 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        #self.bn8 = nn.BatchNorm2d(32)
        
        self.conv9 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        #self.bn9 = nn.BatchNorm2d(32)
        self.conv10 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        f1_output = self.f1_compute(x)
        return self.f2_compute(f1_output)
    
    def f1_compute(self, x):

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        return self.pixel_shuffle(self.conv6(x))
        # x = self.relu(self.bn1(self.conv1(x)))
        # x = self.relu(self.bn2(self.conv2(x)))
        # x = self.relu(self.bn3(self.conv3(x)))
        # x = self.relu(self.bn4(self.conv4(x)))
        # x = self.relu(self.bn5(self.conv5(x)))
        # return self.pixel_shuffle(self.bn6(self.conv6(x)))
    
    def f2_compute(self, f1_output):
        x = self.relu(self.conv7(f1_output))
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))

        # x = self.relu(self.bn7(self.conv7(f1_output)))
        # x = self.relu(self.bn8(self.conv8(x)))
        # x = self.relu(self.bn9(self.conv9(x)))
        return self.conv10(x)

class PerceptualLoss(nn.Module):
    def __init__(self, feature_extractor):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = feature_extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        input_features = self.feature_extractor(input)
        target_features = self.feature_extractor(target)
        return nn.functional.mse_loss(input_features, target_features)

def main():

    def save_epoch_checkpoint(state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)

    def tensor_to_pil(tensor):
        tensor = tensor.squeeze()
        return transforms.ToPILImage()(tensor)
    
    def show_images(lr_image, hr_image, sr_image, epoch, output_dir='./output_images'):
        lr_image, hr_image, sr_image = [tensor_to_pil(image) for image in [lr_image, hr_image, sr_image]]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        lr_image.save(os.path.join(output_dir, f'epoch_{str(epoch)}_lr.png'))
        hr_image.save(os.path.join(output_dir, f'epoch_{str(epoch)}_hr.png'))
        sr_image.save(os.path.join(output_dir, f'epoch_{str(epoch)}_sr.png'))
    
    def psnr_loss(output, target, max_pixel=1.0):
        mse = torch.mean((output - target) ** 2)
        if mse == 0:
            return torch.tensor(float('inf'))
        return 20 * torch.log10(max_pixel / torch.sqrt(mse))
    
        # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # axes[0].imshow(lr_image, cmap='gray')
        # axes[0].set_title('Low-Resolution')
        # axes[1].imshow(hr_image, cmap='gray')
        # axes[1].set_title('High-Resolution')
        # axes[2].imshow(sr_image, cmap='gray')
        # axes[2].set_title('Super-Resolved')
        # for ax in axes:
        #     ax.axis('off')
        # plt.show()
    
    # def load_checkpoint(model, optimiser, filename='checkpoint.pth.tar', device='cuda'):
    #     start_epoch = 0
    #     validation_loss_min = float('inf')  # It's better to start with the highest possible loss

    #     if os.path.isfile(filename):
    #         print(f"Loading checkpoint '{filename}'")
    #         checkpoint = torch.load(filename, map_location=device)  # Ensure the checkpoint is loaded to the correct device
    #         start_epoch = checkpoint['epoch']
    #         validation_loss_min = checkpoint.get('validation_loss', validation_loss_min)
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimiser.load_state_dict(checkpoint['optimizer'])
    #         print(f"Loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")
    #         # Move the model to the device after loading the checkpoint
    #         model.to(device)
    #     else:
    #         print(f"No checkpoint found at '{filename}'")
        
    #     return model, optimiser, start_epoch, validation_loss_min

    # Add checks for CUDA compatability, prefer GPU over CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = GradScaler()

    current_directory = os.getcwd()
    hr_dir = os.path.join(current_directory, "ProcessedImages\\HR\\Train")
    lr_dir = os.path.join(current_directory, "ProcessedImages\\LR\\Train")
    val_hr_dir = os.path.join(current_directory, "ProcessedImages\\HR\\Validation")
    val_lr_dir = os.path.join(current_directory, "ProcessedImages\\LR\\Validation")

    train_dataset = SISRDataSet(hr_dir=hr_dir, lr_dir=lr_dir, training=True)
    train_load = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, persistent_workers=True)

    validation_dataset = SISRDataSet(hr_dir=val_hr_dir, lr_dir=val_lr_dir, transform=transforms.ToTensor(), training=False)
    validation_load = DataLoader(validation_dataset, batch_size=16, shuffle=True, num_workers=12)

    model = SISRCNN()
    optimiser = optim.Adam(model.parameters(), lr=0.0001)
#     model, optimser, start_epoch, validation_loss_min = load_checkpoint(
#     model, 
#     optimiser, 
#     filename='checkpoint_epoch_10.pth.tar',
#     device=device
# )

    scheduler = ReduceLROnPlateau(optimiser, 'min', patience=5, factor=0.1, verbose=True)
    #scheduler = StepLR(optimiser, step_size=10, gamma=0.1)
    model = model.to(device)

    mse_loss_fn = nn.MSELoss()
    #ssim_loss_fn = pytorch_ssim.SSIM()
    #ssim_module = pytorch_ssim.SSIM(window_size = 11)
    lambda_mse = 1.0
    lambda_psnr = 0.5
    #lambda_ssim = 0.5

    num_epochs = 20 
    print_every_n_batches = 100  # Print information every n batches
    hr_image, lr_image = train_dataset[0]

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()

        for batch_idx, (hr_images, lr_images) in enumerate(train_load):
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)

            with autocast():

                sr_images = model(lr_images)
                mse_loss = mse_loss_fn(sr_images, hr_images)
                psnr_loss_value = psnr_loss(sr_images, hr_images)
                #ssim_loss_value = 1 - ssim_module(sr_images, hr_images)

                total_loss = lambda_mse * mse_loss #+ lambda_psnr * psnr_loss_value# + lambda_ssim + ssim_loss_value
                    # alpha * mse_loss + beta * perceptual_loss

            # backward pass and optimization
            optimiser.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimiser)
            scaler.update()

            running_loss += total_loss.item()

            if (batch_idx + 1) % print_every_n_batches == 0 or batch_idx == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Batch [{batch_idx+1}/{len(train_load)}], "
                    f"Loss: {running_loss / print_every_n_batches:.4f}")
                running_loss = 0.0
                lr_image, hr_image = lr_images[0:1], hr_images[0:1]

                lr_image, hr_image = lr_image.to(device), hr_image.to(device)
                with torch.no_grad():
                    sr_image = model(lr_image)
                
            
        model.eval()
        validation_loss = 0.0
        with torch.no_grad(): 
            for hr_images, lr_images in validation_load:
                lr_images = lr_images.to(device)
                hr_images = hr_images.to(device)
                sr_images = model(lr_images)
                loss = mse_loss_fn(sr_images, hr_images)
                validation_loss += loss.item()
            validation_loss /= len(validation_load)
            print(f'Validation Loss: {validation_loss:.4f}')
            

        scheduler.step(validation_loss)
                    
        
        save_epoch_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimiser.state_dict(),
            'validation_loss': validation_loss
        }, filename=f"checkpoint_epoch_{epoch+1}.pth.tar")
        show_images(lr_image.cpu(), hr_image.cpu(), sr_image.cpu(), epoch)
        

        model.train()
        #scheduler.step()
        
    
    print("Training completed")

if __name__ == '__main__':
    main()