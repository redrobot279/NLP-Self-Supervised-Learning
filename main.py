import time
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
from SimClr2 import train as train_simclr
from MoCo2 import train as train_moco
from BarlowTwins2 import train as train_barlow_twins
from DataLoader import get_dataloader


def evaluate_classification(backbone, dataloader, num_classes=10, k=5):
    """
    Perform classification evaluation on a trained backbone.
    
    Args:
        backbone: Trained feature extractor.
        dataloader: Dataloader for labeled data.
        num_classes: Number of output classes (e.g., 10 for CIFAR-10).
        k: Top-k accuracy value.
    
    Returns:
        metrics: Dictionary with evaluation metrics.
    """
    # Freeze the backbone
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    # Add a linear classifier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = torch.nn.Linear(512, num_classes).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    # Train classifier
    classifier.train()
    for epoch in range(5):  # Short training for evaluation
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            features = backbone(inputs).flatten(start_dim=1)
            outputs = classifier(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate classifier
    classifier.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            features = backbone(inputs).flatten(start_dim=1)
            outputs = classifier(features)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate metrics
    linear_eval_accuracy = 100.0 * correct / total
    precision = precision_score(all_labels, all_predictions, average="weighted")
    recall = recall_score(all_labels, all_predictions, average="weighted")
    f1 = f1_score(all_labels, all_predictions, average="weighted")

    return {
        "Linear Eval Accuracy": linear_eval_accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
    }


def fine_tune_model(backbone, dataloader, num_classes=10):
    """
    Fine-tune the model and return accuracy.
    
    Args:
        backbone: Trained feature extractor.
        dataloader: Dataloader for labeled data.
        num_classes: Number of output classes (e.g., 10 for CIFAR-10).
    
    Returns:
        fine_tune_accuracy: Fine-tuning accuracy.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.nn.Sequential(
        backbone,
        torch.nn.Flatten(),
        torch.nn.Linear(512, num_classes)
    ).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    # Fine-tuning loop
    model.train()
    for epoch in range(5):  # Fine-tune for 5 epochs
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    fine_tune_accuracy = 100.0 * correct / total
    return fine_tune_accuracy


def main():
    # Configuration
    dataset_path = "datasets"
    batch_size = 256
    num_workers = 8
    epochs = 10

    # Initialize results table
    results = {
        "Model": [],
        "Linear Eval Accuracy (%)": [],
        "Fine-tune Accuracy (%)": [],
        "Precision": [],
        "Recall": [],
        "F1 Score": [],
        "Training Time (hours)": [],
        "Memory Usage (GB)": [],
    }

    for model_name, train_func, model_type in [
        ("SimCLR", train_simclr, "simclr"),
        ("MoCo", train_moco, "moco"),
        ("Barlow Twins", train_barlow_twins, "barlow_twins"),
    ]:
        print(f"\nRunning {model_name}...")
        start_time = time.time()
        train_func(dataset_path, batch_size, num_workers, epochs)
        training_time = (time.time() - start_time) / 3600  # Training time in hours

        memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Memory usage in GB

        # Linear evaluation
        dataloader = get_dataloader("cifar10", batch_size=128, model_type=model_type, num_workers=num_workers)
        backbone = train_func.backbone
        classification_metrics = evaluate_classification(backbone, dataloader)

        # Fine-tuning
        fine_tune_accuracy = fine_tune_model(backbone, dataloader)

        # Append results
        results["Model"].append(model_name)
        results["Linear Eval Accuracy (%)"].append(classification_metrics["Linear Eval Accuracy"])
        results["Fine-tune Accuracy (%)"].append(fine_tune_accuracy)
        results["Precision"].append(classification_metrics["Precision"])
        results["Recall"].append(classification_metrics["Recall"])
        results["F1 Score"].append(classification_metrics["F1 Score"])
        results["Training Time (hours)"].append(training_time)
        results["Memory Usage (GB)"].append(memory_usage)

    # Print results table
    print("\nResults Table:")
    print(f"| {'Model':<15} | {'Linear Eval Accuracy (%)':<25} | {'Fine-tune Accuracy (%)':<25} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'Training Time (hours)':<20} | {'Memory Usage (GB)':<15} |")
    print("-" * 150)
    for i in range(len(results["Model"])):
        print(
            f"| {results['Model'][i]:<15} | {results['Linear Eval Accuracy (%)'][i]:<25.2f} | {results['Fine-tune Accuracy (%)'][i]:<25.2f} | {results['Precision'][i]:<10.2f} | {results['Recall'][i]:<10.2f} | {results['F1 Score'][i]:<10.2f} | {results['Training Time (hours)'][i]:<20.2f} | {results['Memory Usage (GB)'][i]:<15.2f} |"
        )


if __name__ == "__main__":
    main()
