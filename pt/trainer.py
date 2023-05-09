# TODO add option to take in MNIST / CIFAR datasets or pass your own in 
# TODO have option to pass a holdout model pb in, or otherwise to train the holdout model
import argparse
import tensorflow as tf
from tensorflow import keras
from utils import download_qmnist, prep_data
from models import ConvModelMNIST

# TODO where should I store all of the hyperparameters?

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prioritized train")
    parser.add_argument('--batch_size'          , type=int, default=64, help='training batch size')
    parser.add_argument('--epochs'              , type=int, default=10, help='number of train epochs for prioritized, random subsample, and null hypothesis models')
    parser.add_argument('--holdout_epochs'      , type=int, default=10, help='number of train epochs for holdout model')
    parser.add_argument('--steps_per_epoch'     , type=int, default=None, help='control metric logging freq since we log metrics once per epoch')
    parser.add_argument('--loss'                , type=str, default='categorical_crossentropy', help='loss to train all models with')
    parser.add_argument('--learning_rate'       , type=float, default=None)
    parser.add_argument('--random_subsample'    , action='store_true', help='if true, train random subsample model')
    parser.add_argument('--null_hypot'          , action='store_true', help='if true, train null hypothesis model')
    parser.add_argument('--prioritized_training', action='store_true', help='if true, train prioritized training')
    parser.add_argument('--output_data'         , default='pt-data/', help='input data filenames or directory name containing the files')
    args = parser.parse_args()


    # Create tf data objects for MNIST 
    images, labels = download_qmnist()
    (x_train, y_train), (x_val, y_val), (x_holdout, y_holdout) = prep_data(images, labels)
    train_dataset   = tf.data.Dataset.from_tensor_slices((x_train, y_train))    .batch(args.batch_size)
    val_dataset     = tf.data.Dataset.from_tensor_slices((x_val, y_val))        .batch(args.batch_size)
    holdout_dataset = tf.data.Dataset.from_tensor_slices((x_holdout, y_holdout)).batch(args.batch_size)
        

    # Create the model
    model = ConvModelMNIST()
    model.compile(
            optimizer= 'adam',
            loss     = args.loss,
            metrics  = ['accuracy']
        )

    history = model.fit(train_dataset, 
                        epochs=1, 
                        validation_data=val_dataset, 
                        verbose=1)
