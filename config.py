import torch
class Config:
    # Data paths
    data_dir = 'data/datasets/cifar10/'
    train_data_dir = data_dir + 'train/datasets/cifar10/'
    test_data_dir = data_dir + 'test/datasets/cifar10/'

    log_interval = 1
    # Model hyperparameters
    input_size = 224
    num_classes = 10
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # Checkpoint paths
    checkpoint_dir = 'models/checkpoints/'
    best_model_path = checkpoint_dir + 'best_model.pth'
    last_model_path = checkpoint_dir + 'last_model.pth'

    # Logging and visualization
    log_dir = 'experiments/logs/'
    tensorboard_log_dir = log_dir + 'tensorboard/'

    # Other configurations
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 42
