import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from lightly.transforms.simclr_transform import SimCLRTransform


from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.moco_transform import MoCoV2Transform
from lightly.utils.scheduler import cosine_schedule




from lightly.loss import BarlowTwinsLoss
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.transforms.byol_transform import (
    BYOLTransform,
    BYOLView1Transform,
    BYOLView2Transform,
)


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
    elif model_type.lower() == "moco":
        # Standard augmentations for MoCo and Barlow Twins
        return MoCoV2Transform(input_size=32)
    elif model_type.lower() == "barlow_twins":
        return BYOLTransform(
            view_1_transform=BYOLView1Transform(input_size=32, gaussian_blur=0.0),
            view_2_transform=BYOLView2Transform(input_size=32, gaussian_blur=0.0),
            )
        
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
