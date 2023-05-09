# TODO add option to take in MNIST / CIFAR datasets or pass your own in 
# TODO have option to pass a holdout model pb in, or otherwise to train the holdout model
from utils import download_qmnist, prep_data
images, labels = download_qmnist()
(x_train, y_train), (x_test, y_test), (x_holdout, y_holdout) = prep_data(images, labels)