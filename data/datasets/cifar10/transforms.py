from torchvision import transforms


class CustomTransforms:
    def __init__(self, resize_size=256, crop_size=224):
        self.resize_size = resize_size
        self.crop_size = crop_size

    def get_train_transforms(self):
        return transforms.Compose([
            transforms.Resize(self.resize_size),
            transforms.RandomResizedCrop(self.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_test_transforms(self):
        return transforms.Compose([
            transforms.Resize(self.resize_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


# Example usage:
if __name__ == "__main__":
    # Initialize custom transforms
    custom_transforms = CustomTransforms(resize_size=256, crop_size=224)

    # Get train and test transforms
    train_transforms = custom_transforms.get_train_transforms()
    test_transforms = custom_transforms.get_test_transforms()

    # Print the train and test transforms
    print("Train Transforms:\n", train_transforms)
    print("\nTest Transforms:\n", test_transforms)
