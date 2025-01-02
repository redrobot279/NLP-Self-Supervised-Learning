import copy
import torch
import torchvision
from torch import nn

from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
from DataLoader import get_dataloader  # Import custom data loader


class MoCo(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = MoCoProjectionHead(512, 512, 128)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        query = self.backbone(x).flatten(start_dim=1)
        query = self.projection_head(query)
        return query

    def forward_momentum(self, x):
        key = self.backbone_momentum(x).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key


def train(dataset_path, batch_size, num_workers, epochs):
    """
    Train the MoCo model.
    
    Args:
        dataset_path (str): Path to the dataset directory.
        batch_size (int): Batch size for the dataloader.
        num_workers (int): Number of workers for the dataloader.
        epochs (int): Number of training epochs.
    """
    # Initialize the backbone and model
    resnet = torchvision.models.resnet18()
    backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove the classification layer
    model = MoCo(backbone)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Use the custom data loader for MoCo
    dataloader = get_dataloader(
        dataset_name="cifar10",
        batch_size=batch_size,
        model_type="moco",  # Use "moco" for appropriate transformations
        num_workers=num_workers,
    )

    # Define loss, optimizer, and training parameters
    criterion = NTXentLoss(memory_bank_size=4096)  # Memory bank for negatives
    optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

    # Training Loop
    print("Starting MoCo Training")
    for epoch in range(epochs):
        total_loss = 0
        momentum_val = cosine_schedule(epoch, epochs, 0.996, 1)  # Momentum schedule

        for batch in dataloader:
            x_query, x_key = batch[0]  # Extract query and key views
            update_momentum(model.backbone, model.backbone_momentum, m=momentum_val)
            update_momentum(model.projection_head, model.projection_head_momentum, m=momentum_val)

            x_query = x_query.to(device)
            x_key = x_key.to(device)

            query = model(x_query)
            key = model.forward_momentum(x_key)

            loss = criterion(query, key)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.5f}")


if __name__ == "__main__":
    # Default settings for standalone execution
    train(
        dataset_path="datasets",
        batch_size=256,
        num_workers=8,
        epochs=10,
    )
