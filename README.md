# VGG16-CIFAR PyTorch Project

This repository contains a PyTorch implementation of the VGG16 model for the CIFAR-10 dataset. The VGG16 architecture is a widely used convolutional neural network for image classification tasks.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/KimangKhenng/vgg16-cifar-pytorch.git
   cd vgg16-cifar-pytorch
   ```

2. Install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Make sure you have downloaded the CIFAR-10 dataset or set up the data directory according to your needs.

2. Customize the configuration parameters in `config.py` if required.

3. Run the `main.py` script to train and evaluate the VGG16 model:
   ```bash
   python main.py
   ```

## Project Structure

- `models/`: Contains the VGG16 model architecture (`vgg16.py`) and the `__init__.py` file to initialize the models module.
- `data/datasets/`: Includes dataset-related files such as `cifar10.py` for loading CIFAR-10 data and `__init__.py` to initialize the datasets module.
- `utils/`: Contains utility scripts such as `config.py` for project configurations, `evaluation.py` for evaluation metrics, `visualization.py` for data visualization, and `__init__.py` to initialize the utils module.
- `main.py`: Entry point script to train and evaluate the VGG16 model on the CIFAR-10 dataset.
- `requirements.txt`: Specifies the required Python packages and their versions.

## Acknowledgements

- The CIFAR-10 dataset is obtained from the official CIFAR website (https://www.cs.toronto.edu/~kriz/cifar.html).
- The VGG16 architecture is based on the original paper "Very Deep Convolutional Networks for Large-Scale Image Recognition" by Karen Simonyan and Andrew Zisserman.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
