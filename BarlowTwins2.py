import torch
import torchvision
from torch import nn

from lightly.loss import BarlowTwinsLoss
from lightly.models.modules import BarlowTwinsProjectionHead
from DataLoader import get_dataloader  # Import the custom data loader


class BarlowTwins(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(512, 2048, 2048)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z


def train(dataset_path, batch_size, num_workers, epochs):
    """
    Train the Barlow Twins model.
    
    Args:
        dataset_path (str): Path to the dataset directory.
        batch_size (int): Batch size for the dataloader.
        num_workers (int): Number of workers for the dataloader.
        epochs (int): Number of training epochs.
    """
    # Initialize the model
    resnet = torchvision.models.resnet18()
    backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove the final classification layer
    model = BarlowTwins(backbone)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Use the custom data loader for Barlow Twins
    dataloader = get_dataloader(
        dataset_name="cifar10",
        batch_size=batch_size,
        model_type="barlow_twins",  # Use "barlow_twins" to apply appropriate transformations
        num_workers=num_workers,
    )

    # Define the criterion and optimizer
    criterion = BarlowTwinsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

    # Training Loop
    print("Starting Barlow Twins Training")
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x0, x1 = batch[0]  # Extract the two views for Barlow Twins
            x0 = x0.to(device)
            x1 = x1.to(device)
            z0 = model(x0)
            z1 = model(x1)
            loss = criterion(z0, z1)
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
