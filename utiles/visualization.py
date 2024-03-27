import matplotlib.pyplot as plt
import numpy as np


def plot_image(image, title=None):
    """
    Plot a single image.

    Args:
    - image (numpy array): Image data in numpy array format.
    - title (str): Title for the plot (optional).
    """
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plot the confusion matrix.

    Args:
    - cm (numpy array): Confusion matrix.
    - classes (list): List of class labels.
    - normalize (bool): Whether to normalize the confusion matrix (default False).
    - title (str): Title for the plot (default 'Confusion Matrix').
    - cmap (colormap): Colormap for the plot (default plt.cm.Blues).
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Example confusion matrix
    cm = np.array([[10, 2],
                   [3, 15]])

    # Define class labels
    classes = ['Class A', 'Class B']

    # Plot confusion matrix
    plot_confusion_matrix(cm, classes, normalize=True, title='Normalized Confusion Matrix')
