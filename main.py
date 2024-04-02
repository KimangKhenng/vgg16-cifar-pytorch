import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import VGG16
from data.datasets import CIFAR10
from utiles import evaluate_classification
from data.datasets.cifar10 import CustomTransforms
import numpy as np

from config import Config


def main():
    # Set random seed for reproducibility
    torch.manual_seed(Config.seed)

    # Define transformations
    custom_transforms = CustomTransforms(resize_size=256, crop_size=224)
    train_transforms = custom_transforms.get_train_transforms()
    test_transforms = custom_transforms.get_test_transforms()

    # Create CIFAR-10 train and test datasets
    train_dataset = CIFAR10(root=Config.data_dir, train=True, transform=train_transforms, download=True)
    test_dataset = CIFAR10(root=Config.data_dir, train=False, transform=test_transforms, download=True)

    # Initialize data loaders
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

    # Initialize the VGG16 model
    model = VGG16(num_classes=Config.num_classes)
    model.to(Config.device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)

    # Training loop
    for epoch in range(Config.num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(Config.device), labels.to(Config.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            if (batch_idx + 1) % Config.log_interval == 0:
                avg_loss = running_loss / Config.log_interval
                accuracy = (correct_predictions / total_samples) * 100
                print(f'Epoch {epoch + 1}/{Config.num_epochs}, '
                      f'Batch {batch_idx + 1}/{len(train_loader)}, '
                      f'Loss: {avg_loss:.4f}, '
                      f'Accuracy: {accuracy:.2f}%')
                running_loss = 0.0
                correct_predictions = 0
                total_samples = 0

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{Config.num_epochs}, Loss: {epoch_loss:.4f}')

    print('Training finished.')

    # Evaluation on test set
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(Config.device), labels.to(Config.device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate evaluation metrics
    metrics = evaluate_classification(np.array(all_labels), np.array(all_preds))
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    # Save the trained model
    torch.save(model.state_dict(), Config.model_path)
    print(f"Trained model saved at: {Config.model_path}")


if __name__ == "__main__":
    main()
