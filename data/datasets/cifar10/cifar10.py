import torch
from torchvision.datasets import CIFAR10 as TorchCIFAR10
from torchvision import transforms


class CIFAR10(TorchCIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR10, self).__init__(root=root, train=train, transform=transform,
                                      target_transform=target_transform, download=download)

        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # def __getitem__(self, index):
    #     image, target = self.data[index], int(self.targets[index])
    #
    #     # Convert image to PIL and apply transformations if specified
    #     if self.transform is not None:
    #         image = self.transform(image)
    #
    #     return image, target

    def __len__(self):
        return len(self.data)


# Example usage:
if __name__ == "__main__":
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # Create CIFAR-10 train and test datasets
    train_dataset = CIFAR10(root='./', train=True, transform=transform, download=True)
    test_dataset = CIFAR10(root='./', train=False, transform=transform, download=True)

    # Example: print the length of the train dataset
    print("Train dataset length:", len(train_dataset))
