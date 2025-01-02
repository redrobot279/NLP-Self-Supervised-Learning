import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from lightly.transforms.simclr_transform import SimCLRTransform

# Custom dataset wrapper for models like MoCo and SimCLR
class SSLDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Return two augmented views of the same image
        img, label = self.dataset[index]
        return img, label

# Function to apply consistent augmentations
def get_transform(model_type="simclr", input_size=32):
    """
    Args:
        model_type: str, one of 'simclr', 'moco', 'barlow_twins'.
        input_size: int, the input image size.
    Returns:
        transform: torchvision.transforms.Compose or SimCLRTransform
    """
    if model_type.lower() == "simclr":
        # SimCLR uses its own transformation wrapper
        return SimCLRTransform(input_size=input_size, gaussian_blur=0.1)
    elif model_type.lower() in ["moco", "barlow_twins"]:
        # Standard augmentations for MoCo and Barlow Twins
        return T.Compose([
            T.RandomResizedCrop(size=input_size),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def get_dataloader(dataset_name="cifar10", batch_size=256, model_type="simclr", num_workers=8):
    """
    Args:
        dataset_name: str, the name of the dataset ('cifar10' or 'imagenet').
        batch_size: int, the number of samples per batch.
        model_type: str, one of 'simclr', 'moco', 'barlow_twins'.
        num_workers: int, number of workers for DataLoader.
    Returns:
        dataloader: DataLoader object for the specified dataset.
    """
    # Select dataset
    if dataset_name.lower() == "cifar10":
        transform = get_transform(model_type=model_type, input_size=32)
        dataset = torchvision.datasets.CIFAR10(
            root="datasets/cifar10",
            train=True,
            download=True,
            transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Wrap dataset if needed
    if model_type.lower() in ["moco", "simclr"]:
        dataset = SSLDataset(dataset)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    return dataloader
