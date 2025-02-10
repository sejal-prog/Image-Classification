# Preprocessing - We do this to ensure that all images are consistent. 

import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Preprocessing 
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # For this task, color is not important. Conversion to grayscale helps to remove color noise. 
    transforms.Resize((64, 64)),                  # Resized all images to 64x64. Important because CNNs needs identical image size.  
    transforms.ToTensor(),                        # Convert to tensor because pytorch require tensor inputs. 
    transforms.Normalize(mean=[0.5], std=[0.5])   # Images are normalised here as it makes sure that extreme values dont dominate learning process.
])

# Dataset class
class CatDogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        # Read images from "cat" and "dog" directories and ignore "horse"
        for label, category in enumerate(["cat", "dog"]):
            category_path = os.path.join(root_dir, category)
            if not os.path.exists(category_path):
                raise FileNotFoundError(f"Directory not found: {category_path}")

            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("L") 

        if self.transform:
            image = self.transform(image)

        return image, label

# DataLoaders
def get_dataloaders(batch_size=32):
    train_dataset = CatDogDataset(root_dir="Dataset/train", transform=transform)
    test_dataset = CatDogDataset(root_dir="Dataset/test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
