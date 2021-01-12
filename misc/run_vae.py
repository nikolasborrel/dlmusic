from loaders.dataloader_mnist import load_mnist
from models.model_vae import VariationalInference, VariationalAutoencoder
from training.train_vae import train_vae

num_epochs = 100

# The digit classes to use
classes = [3, 7]
#classes = [0, 1, 4, 9]

# load data
path_train = "~/Documents/PhD/courses/deep_learning/assignments/7_Unsupervised"
path_test  = "~/Documents/PhD/courses/deep_learning/assignments/7_Unsupervised"

train_loader, test_loader = load_mnist(path_train, path_test, classes)

# Load a batch of images into memory
images, labels = next(iter(train_loader))

# TRAINING AND EVALUATION
train_vae(train_loader, test_loader, num_epochs, images[0].shape)