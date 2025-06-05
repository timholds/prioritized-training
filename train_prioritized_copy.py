import argparse
import opendatasets as od
import numpy as np
import tensorflow as tf
import statistics
import json
import datetime
import subprocess
import threading
import time
import os
import math
import cv2
import leap.learn.data_tools.bin_io as bin_io
import tensorflow.keras.backend as K
import random
from numba import cuda

from gc import callbacks
from patch_viewer import view_patch_predictions, view_patches, view_patches_from_numpy
from leap.learn.custom_objects.prioritized_callbacks import PrioritizedTrainingCallback

from queue import Queue
from multiprocessing.pool import ThreadPool

from tensorflow import keras
from leap.learn.data_tools.bin_io import read_binary_params, read_binary_block, read_binary_samples, read_indices





from target_model import ConvModel, ConvModelPatches, ConvModelSuper, ConvModelMasked, ConvModelMaskedPatches

# TODO make sure we are using the right train batch size for holdout model
# TODO make sure arg is getting passed to the Callback so it can set self.train_batch_size in model
# TODO see if i can run non eagerly and check if it speeds anything up

tf.config.run_functions_eagerly(True)
import os
#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# tf.data.experimental.enable_debug_mode()
# for gpu in tf.config.list_physical_devices('GPU'):
#   # tf.config.experimental.set_memory_growth(gpu, True)
#   tf.config.set_soft_device_placement(True)

# ------------ Utility Functions ------------ 
# Read in labels
def bin_to_numpy(bin_file, y_channels, n_samples=88941):
    index = 0
    with open(bin_file, 'rb') as f_y: # y will always be a bin file
        y_params = read_binary_params(f_y)
        print(y_params)
        y_dtype = np.uint8 if y_params[3] == 1 else np.float32
        print(y_dtype)
        # seq_y = read_binary_block(f_y, y_params, index * y_channels, seq_len * y_channels)
        # seq_y = read_binary_block(f_y, y_params, 0, 88941)
        seq_y = read_binary_block(f_y, y_params, 0, n_samples)

        print(seq_y.shape)
        seq_y = seq_y.reshape((-1, y_channels, y_params[2]))
        index += index * y_channels
    return seq_y

def get_images_and_poses_as_numpy(mp4_file, labels_file,x_channels=1):
    cap = cv2.VideoCapture(mp4_file)

    if (cap.isOpened() == False): 
        print("Error opening video stream or file; is it in the same folder as this script?")

    frame_count  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale_factor = 1 # scale images up for easier viewing

    images_arr = np.empty((frame_count, int(frame_height//2) * scale_factor, frame_width * scale_factor, x_channels), dtype=np.uint8)
    poses = bin_io.read_binary(labels_file)
    print("Pose Data Shape:", poses.shape)
    # Exclude the last 4 points from the learning process, since they are on the elbow and outside of patch
    labels_arr = poses[:, :72]
    #labels_arr = poses[:, :84] # * 12.5 # * scale_factor + frame_width/2 # 84 is the number of joints
    #labels_arr = (labels_arr * 12.5) + 16 # 84 is the number of joints
    frame_num = 0
    while frame_num < frame_count:
        received_image, frame = cap.read()
        if received_image:
            # Massage our raw image into a nice, intelligible array

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                       # Convert To GrayScale
            frame = frame.T
            frame = frame[:, :frame.shape[1]//2]                                  # Cut off the right camera view               
            frame = cv2.resize(frame, (frame.shape[1] * scale_factor, 
                                       frame.shape[0] * scale_factor))            # Scale it up 8x so we can see it
           
            images_arr[frame_num] = np.expand_dims(frame, -1)

            if frame_num % 10000 == 0:
                plot_scale_factor = 4

                print("Frame Num:", frame_num)
                pose = labels_arr[frame_num].reshape((-1, 3))
                plot_frame = cv2.resize(frame, (frame.shape[1] * plot_scale_factor, 
                                                frame.shape[0] * plot_scale_factor))            # Scale it up 8x so we can see it
                    
                for joint in pose:
                    # Scale the Joint X/Y by 12.5 * scale_factor
                    cv2.circle(plot_frame, (int(joint[0] * 12.5 * plot_scale_factor) + plot_frame.shape[1]//2, 
                                            int(joint[1] * 12.5 * plot_scale_factor) + plot_frame.shape[0]//2), plot_scale_factor//4, 255)

                # Display the resulting frames
                cv2.imshow('Patch with Pose', plot_frame)
                cv2.waitKey(1)

            frame_num += 1
    
    # cap.release()
    cv2.destroyAllWindows()

    return images_arr, labels_arr

def mp4_to_numpy(mp4_file, x_channels):
    # Create a VideoCapture object for the grayscale video
    cap = cv2.VideoCapture(mp4_file)
    frame_count  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_frame = cap.read()[1]

    print('frame_count', frame_count)
    print('frame_width', frame_width)
    print('frame_height', frame_height)
    print('first_frame', first_frame.shape)

    data = np.zeros((frame_count, int(frame_height/x_channels), frame_width, x_channels), dtype=np.uint8)

    success, image = cap.read()
   
    scale_factor = 8
    frame_num = 0
    while success:
        success, frame = cap.read()
        # if not success:
        #     raise Exception('Failed to read from MP4')
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                       # Convert To GrayScale
            frame = frame.T                                                       # Transpose
            frame = frame[:, :frame.shape[1]//2]                                  # Cut off the right camera view
            frame = cv2.resize(frame, (frame.shape[1] * scale_factor, 
                                    frame.shape[0] * scale_factor))  
            #pose = poses[frame_num].reshape((-1, 3))[:28]

            l_frame = frame[:32,   :, [0]]
            r_frame = frame[32:64, :, [0]]
            #frame_reshape = frame.reshape((int(frame_height/x_channels), frame_width, 2))
            data[frame_num]   = np.concatenate([l_frame, r_frame], axis=-1)

            frame_num += 1

    print('processes {} frames'.format(frame_num))
    return data

def download_qmnist():
    ''' 120k QMNIST downloads to /qmnist-the-extended-mnist-dataset-120k-images/'''
    os.environ['KAGGLE_USERNAME'] = 'timholdsworth'
    #os.environ['KAGGLE_KEY']      = '0932f4bda4797bfbcb6caed20eca8cac'
    os.environ['KAGGLE_KEY']      = 'be622baf9f35230810dfc231a86b2feb'
    od.download('https://www.kaggle.com/datasets/fedesoriano/qmnist-the-extended-mnist-dataset-120k-images?resource=download')
    
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_leap_patches(mp4_file, labels_file, normalize=False):
    #view_patches(mp4_file, labels_file)
    #images, labels = get_images_and_poses_as_numpy(mp4_file, labels_file)#  ,x_channels=2)
    
    images = np.load('patch_images_array.npy')
    labels = np.load('patch_labels_array.npy')[:, :72]

    images = images.astype("float32") / 255
    #images = np.expand_dims(images, -1)

    # TODO if you do normalize, exclude testing and holdout data (or doesnt matter since doing patch wise normalization)
    if normalize:
        print('\nNORMALIZING DATA\n')
        # TODO dont reshape the original images, but make a reshaped copy
        shape = images.shape
        image_patches = np.reshape(images, (shape[0], -1))
        image_mean    = np.mean(image_patches, axis=1, keepdims=True)
        image_std     = np.sqrt(np.var(image_patches, axis=1, keepdims=True) + 10)
        images        = (image_patches - image_mean) / image_std
        images        = np.reshape(images, (shape[0], shape[1], shape[2], shape[3]))

        # scale_factor = 8
        # for image in images[:10000]:
        #     frame = cv2.resize(image, (image.shape[1] * scale_factor, 
        #                           image.shape[0] * scale_factor)) 
        #     cv2.imshow('normalized frame', frame)
        #     cv2.waitKey(1)

        # cv2.destroyAllWindows()
    #view_patches_from_numpy(images, labels)

    # Get the 40% of the images as holdout images, 40% as traun images, and 20% as validation images
    x_holdout = images[:int(images.shape[0] * .2)]
    y_holdout = labels[:int(labels.shape[0] * .2)]
    x_train   = images[int(images.shape[0] * .2) : int(images.shape[0] * .9)]
    y_train   = labels[int(labels.shape[0] * .2) : int(labels.shape[0] * .9)]
    x_test    = images[int(images.shape[0] * .9):]
    y_test    = labels[int(labels.shape[0] * .9):]

    # x_holdout = np.expand_dims(x_holdout,   -1).astype("float32") / 255
    # x_train   = np.expand_dims(x_train  ,   -1).astype("float32") / 255
    # x_test    = np.expand_dims(x_test   ,   -1).astype("float32") / 255
    # x_holdout = x_holdout.astype("float32") / 255
    # x_train   = x_train  .astype("float32") / 255
    # x_test    = x_test   .astype("float32") / 255

    print('patches x_train.shape {}'.format(x_train.shape))
    print('patches y_train.shape {}'.format(y_train.shape))

    return (x_train, y_train), (x_test, y_test), (x_holdout, y_holdout)
  
def prep_mnist_data(images, labels, num_classes=10, embed=False):#, test_per, holdout_per):
    print(images.shape)
    print(labels.shape)

    if embed:
        # from sklearn.manifold import TSNE
        from tsnecuda import TSNE
        tsne = TSNE(n_components=2)
        images_embedded = tsne.fit_transform(images.reshape(images.shape[0], -1))

        y_train   = images_embedded[:50000]
        y_test    = images_embedded[50000: 70000]
        y_holdout = images_embedded[70000:]
    else:
        y_train   = labels[:50000]
        y_test    = labels[50000: 70000]
        y_holdout = labels[70000:]

        #print('encoding as one hot')
        y_train   = keras.utils.to_categorical(y_train, num_classes)
        y_test    = keras.utils.to_categorical(y_test,  num_classes)
        y_holdout = keras.utils.to_categorical(y_holdout,  num_classes) 

    x_train   = images[:50000      ].astype("float32") / 255
    x_test    = images[50000: 70000].astype("float32") / 255
    x_holdout = images[70000:      ].astype("float32") / 255

    # Make sure images have shape (28, 28, 1)
    x_train    = np.expand_dims(x_train,   -1)
    x_test     = np.expand_dims(x_test,    -1)
    x_holdout  = np.expand_dims(x_holdout, -1)

    return (x_train, y_train), (x_test, y_test), (x_holdout, y_holdout)

def reload_data(images, labels, patches=False):
    if patches:
        (x_train, y_train), (x_test, y_test), (x_holdout, y_holdout) = load_leap_patches(images, labels)
    else:
        (x_train, y_train), (x_test, y_test), (x_holdout, y_holdout) = prep_mnist_data(images, labels, num_classes=10, embed=args.tsne)


    return (x_train, y_train), (x_test, y_test), (x_holdout, y_holdout)

def compile_model(model, loss='categorical_crossentropy', learning_rate=None, run_eagerly=False):
    metrics = ['accuracy', 'mse', 'mae', 'categorical_crossentropy']
    if learning_rate:
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        opt = keras.optimizers.Adam()

    # opt = keras.optimizers.SGD()#learning_rate=0.0001)
    #opt   = keras.optimizers.RMSprop()
    model.compile(loss=loss, optimizer=opt, metrics=metrics, run_eagerly=run_eagerly)
    return model


# For Prioritized Training, create the il loss dict by evaluating the holdout model on the training data
def make_il_loss_dict(model, x_train, y_train, loss='categorical_crossentropy'):
    ''' Calculate the loss of the holdout model on the training data.
        Make sure this is done before shuffling the data'''

    preds = model.predict(x_train, batch_size=2048, use_multiprocessing=True)
    print('calculating the il loss dict with {}'.format(loss))
    # Use tf loss functions here since we are comparing loss values 
    if loss == 'categorical_crossentropy':
        loss_vals = keras.losses.categorical_crossentropy(preds, y_train).numpy()
    elif loss == 'mse':
        loss_vals =                      keras.losses.mse(preds, y_train).numpy()
    elif loss == 'mae':
        loss_vals =                      keras.losses.mae(preds, y_train).numpy()
    print('loss_vals shape is {}'.format(loss_vals.shape))

    il_loss_dict = {}
    for i, loss_val in enumerate(loss_vals):
        il_loss_dict[i] = loss_val

    return il_loss_dict

def log_util_usage():
    print('\n *** logging utility usage ***\n')
    mem_data = []
    cpu_data = []
    gpu_mem_data = []
    while True:
        t0 = time.time()
        cpu_data.append(psutil.cpu_percent())
        json_content = {"datasets": {"log_cpu_util": cpu_data},
                        "type": "line-chart",
                        "version": 1,
                        "xAxisLabel": "Minute",
                        "yAxisLabel": "Percent"}

        with open('cpu_usage.json', "w+") as f:
            json.dump(json_content, f)

        mem_data.append(psutil.virtual_memory().used / (2 ** 20))
        json_content = {"datasets": {"log_memory": mem_data},
                        "type": "line-chart",
                        "version": 1,
                        "xAxisLabel": "Minute",
                        "yAxisLabel": "Memory Used (MiB)"}

        with open('cpu_memory.json', "w+") as f:
            json.dump(json_content, f)

        try:
            nvidia_smi_proc = subprocess.Popen("nvidia-smi --query-gpu=memory.used --format=csv", stdout=subprocess.PIPE, shell=True)
            stdout, _= nvidia_smi_proc.communicate(timeout=2)
            gpu_mem_used = [line for line in stdout.decode().split("\n") if len(line) > 2][1:]
            gpu_mem_used = [int(line.split(" ")[0]) for line in gpu_mem_used]
            total_gpu_mem_used = sum(gpu_mem_used)
            gpu_mem_data.append(total_gpu_mem_used)
        except (UnicodeDecodeError, IndexError, ValueError):
            gpu_mem_data.append(gpu_mem_data[-1])

        json_content = {"datasets": {"log_gpu_memory": gpu_mem_data},
                        "type": "line-chart",
                        "version": 1,
                        "xAxisLabel": "Minute",
                        "yAxisLabel": "GPU Memory Used (MiB)"}

        with open('gpu_memory.json', "w+") as f:
            json.dump(json_content, f)

        time.sleep(60 - (time.time() - t0))

# ------ Train functions that return a dict of metrics ------
def train_holdout_model(holdout_model, x_holdout, y_holdout, x_test, y_test, holdout_loss, holdout_epochs, lr, batch_size):
    holdout_model_dict = {'holdout': {}}
    holdout_model_dict['holdout']['hyperparams'] = {
        'model'        : 'holdout_model',
        'batch_size'   : batch_size,
        'epochs'       : holdout_epochs, 
        'learning_rate': lr, 
        'loss'         : holdout_loss}

    t0_holdout = time.time()
    print('\n *** Fitting Holdout Model ***\n')
    hist = holdout_model.fit(x_holdout, y_holdout, epochs=holdout_epochs, batch_size=batch_size, 
                            validation_data=(x_test, y_test), verbose=1, shuffle=True)
    t1_holdout = time.time()
    print('Took {}s to train the holdout model'.format(t1_holdout - t0_holdout))
    
    holdout_model_dict['holdout']['metrics'] = hist.history
    return holdout_model_dict, holdout_model

def train_null_hypothesis_model(model, x_train, y_train, x_test, y_test, epochs, lr, batch_size):
    # use the batch size arg 
    null_hypot_model_dict = {'nh': {'hyperparams': {}}}
    null_hypot_model_dict['nh']['hyperparams'] = {
        'model'        : 'null_hypothesis',
        'batch_size'   : batch_size, 
        'epochs'       : epochs, 
        'learning_rate': lr, 
        'loss'         : model.loss}
        
    hist = model.fit(x_train, y_train, batch_size = batch_size, epochs=epochs,
                     validation_data=(x_test, y_test), verbose=1, shuffle=True,
                     callbacks=[tensorboard_callback])
    null_hypot_model_dict['nh']['metrics'] = hist.history
    return null_hypot_model_dict

def train_random_subsample_model(x_train, y_train, x_test, y_test, loss, epochs, lr, 
        batch_size, big_batch_sizes, patches=None, tsne=None):
    ''' Train one random subsample model per big batch size'''

    # TODO should we define model: random_sample here or in 
    # random_subsample_model_dict = {'rs': {'model': 'random_subsample', 'hyperparams': {}}}
    random_subsample_model_dict = {'rs': {}}
    for big_batch_size in big_batch_sizes:
        random_subsample_model_dict['rs'][big_batch_size] = {'hyperparams': {}, 'metrics': {}}
        train_accuracies, train_losses, train_mse_list, train_mae_list, train_cce_list = [], [], [], [], []
        val_accuracies  , val_losses  , val_mse       , val_mae,        val_cce        = [], [], [], [], []
        model = ConvModelSuper().create_model(patches=patches, embed=tsne)
        model = compile_model(model, loss=loss, learning_rate=lr)
        random_subsample_model_dict['rs'][big_batch_size]['hyperparams'] = {
            'model'        :'random_subsample',
            'batch_size'   : batch_size,
            'big_batch_size': big_batch_size,
            'epochs'       : epochs, 
            'learning_rate': lr, 
            'loss'         : model.loss}

       
        for epoch in range(epochs):
            print('\nepoch {}'.format(epoch))
            batch_train_accuracies, batch_train_losses, batch_train_mse, batch_train_mae, batch_train_cce = [], [], [], [], []
            image_label_dict_shuffled = shuffle_train_set_with_idx(x_train, y_train)

            i = 0
            while i + int(big_batch_size) <= (len(image_label_dict_shuffled)-1):                
                x_batch = np.array([image_label_dict_shuffled[x]['x'] for x in range(i, i+batch_size)])
                y_batch = np.array([image_label_dict_shuffled[x]['y'] for x in range(i, i+batch_size)])
    
                # Shuffle is false because we already shuffled the data before the start of the epoch
                batch_hist = model.fit(x_batch, y_batch, epochs=1, steps_per_epoch=1, verbose=1, shuffle=False)
            
                batch_train_accuracies.append(batch_hist.history['accuracy'][0])
                batch_train_losses    .append(batch_hist.history['loss'][0])
                batch_train_mse       .append(batch_hist.history['mse'][0])
                batch_train_mae       .append(batch_hist.history['mae'][0])
                batch_train_cce       .append(batch_hist.history['categorical_crossentropy'][0])

                i += big_batch_size
            
            train_accuracies.append(statistics.mean(batch_train_accuracies))
            train_losses.    append(statistics.mean(batch_train_losses))
            train_mse_list.  append(statistics.mean(batch_train_mse))
            train_mae_list.  append(statistics.mean(batch_train_mae))
            train_cce_list.  append(statistics.mean(batch_train_cce))

            print('avg accuracy over epoch: {:.2f}'.format(statistics.mean(batch_train_accuracies)))
            print('avg loss over epoch: {:.2f}'    .format(statistics.mean(batch_train_losses)))
            print('avg mse over epoch: {:.3f}'         .format(statistics.mean(batch_train_mse)))
            print('avg mae over epoch: {:.3f}'         .format(statistics.mean(batch_train_mae)))
            print('avg cce over epoch: {:.3f}'         .format(statistics.mean(batch_train_cce)))

            val_hist = model.evaluate(x_test, y_test, return_dict=True)#, training=False)
            val_accuracies.append(val_hist['accuracy'])                 
            val_losses    .append(val_hist['loss'])                    
            val_mse       .append(val_hist['mse'])                     
            val_mae       .append(val_hist['mae'])                     
            val_cce       .append(val_hist['categorical_crossentropy'])
    
        t1 = time.time()
        train_time = t1 - t0
        print('took {} in total to run  for {} epochs'.format(train_time, epochs))

        train_val_dict = {}
        train_val_dict   = {'val_accuracy': val_accuracies , 
                            'val_loss'    : val_losses, 
                            'val_mse'     : val_mse,
                            'val_mae'     : val_mae,
                            'val_cce'     : val_cce}
        train_val_dict   = {'accuracy': train_accuracies ,
                            'loss'    : train_losses ,
                            'mse'     : train_mse_list,
                            'mae'     : train_mae_list,
                            'cce'     : train_cce_list,
                            'time'    : train_time}
        random_subsample_model_dict['rs'][big_batch_size]['metrics'] = train_val_dict
    
    return random_subsample_model_dict

def train_pt_model(il_model_loss_dict, x_train, y_train, x_test, y_test,
        loss, epochs, lr, batch_size, big_batch_sizes,
        patches=None, tsne=None):
    ''' Train one prioritized model per big batch size'''

    pt_model_dict = {'pt': {}}
    for big_batch_size in big_batch_sizes:
        # pt_model_dict['pt'][big_batch_size] = {'hyperparams': {}, 'metrics': {}}
        pt_model_dict['pt'][big_batch_size] = {
            'hyperparams': {
                'model'         :'prioritized_training',
                'batch_size'    : batch_size, 
                'big_batch_size': big_batch_size, 
                'epochs'        : epochs, 
                'learning_rate' : lr, 
                'loss'          : model.loss},
            'metrics': {}}

        if args.patches:
            model = ConvModelMaskedPatches()
        else:
            model = ConvModelMasked()

        (x_train, y_train), (x_test, y_test), (x_holdout, y_holdout) = reload_data(images, labels, patches=args.patches)
        callbacks = [PrioritizedTrainingCallback(train_data=(x_train, y_train), 
                                   il_loss_dict=il_model_loss_dict,
                                   train_batch_size=batch_size,
                                   cand_batch_size=big_batch_size),
                    tensorboard_callback]

        # model = ConvModelSuper().create_model(patches=patches, embed=tsne)
        model = compile_model(model, loss=loss, learning_rate=lr, run_eagerly=True)
        
        steps_per_epoch = 40
        mult        = math.ceil((big_batch_size*steps_per_epoch)/len(x_train)) # ensure we don't run out of data
        x_train_rep = np.repeat(x_train, mult, axis=0)
        y_train_rep = np.tile(y_train, mult)

        history = model.fit(x_train_rep, y_train_rep, validation_data=(x_test, y_test),
            batch_size=big_batch_size,
            epochs=epochs,
            shuffle=False, #using a custom shuffling callback to shuffle il losses in sync with images/labels
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks)
        
        print(history.keys)
        return history.history
        pt_model_dict['pt'][big_batch_size]['metrics'] = train_val_dict
        
    return pt_model_dict

 

if __name__ == '__main__':
    ''' 
    big_batch_sizes: List of big_batch_sizes to try, with larger values resulting in a smaller number of samples per epoch
    teacher_regression: If True, use regression loss (mse) on holdout model. If False, use a classification loss (categorical_crossentropy) 
    
    The function runs roughly as follows
    - 1) train a model on holdout dataset with mse or cce loss
    - 2) evaluate the holdout model against the *training* set and make a dict of frame_idx: loss value
    - 3) for each size in big_batch_sizes:
            - train 4 target models:
              - train a cce network with null hypothesis & another with prioritized training (classif target)
              - train a mse network with null hypothesis & another with prioritized training (regression target)

    '''
    # TODO fix parsing of big_batch_size we can pass things like 2500, 5000 and still have it work on generics
    parser = argparse.ArgumentParser(description="Prioritized train")
    parser.add_argument('--patches'    , action='store_true', help='if true, use the patches passed in')
    parser.add_argument('--mp4_file'   , default=None, help='image patches if using mp4')
    parser.add_argument('--labels_file', default=None, help='patch labels if running on leap patches instead of MNIST')    
    
    parser.add_argument('--qmnist'     , help='input data filenames or directory name containing the files')
    parser.add_argument('--tsne'         ,  action='store_true', help='whether to encode as 2d regression problem to tsne location if true, or one hot if false')

    parser.add_argument('--epochs'        , type=int, default=10, help='number of train epochs for prioritized, random subsample, and null hypothesis models')
    parser.add_argument('--holdout_epochs', type=int, default=10, help='number of train epochs for holdout model')
    parser.add_argument('--loss'          , type=str, default='categorical_crossentropy', help='loss to train all models with')

    parser.add_argument('--big_batch_size', type=int, nargs='+',  default=[640], help='input data filenames or directory name containing the files')
    parser.add_argument('--batch_size'    , type=int, default=64, help='training batch size')
    parser.add_argument('--learning_rate',  type=float, default=None)

    parser.add_argument('--random_subsample'    , action='store_true', help='if true, train random subsample model')
    parser.add_argument('--null_hypot'          , action='store_true', help='if true, train null hypothesis model')
    parser.add_argument('--prioritized_training', action='store_true', help='if true, train prioritized training')
    
    parser.add_argument('--output_data', default='data/', help='input data filenames or directory name containing the files')
    args = parser.parse_args()

    if not os.path.exists(args.output_data):
        print(args.output_data)
        os.makedirs(args.output_data)

    t0 = time.time()
    log_util_thread = threading.Thread(target=log_util_usage) # , args=(args,))
    log_util_thread.daemon = True
    log_util_thread.start()
    log_dir = "logs/fit/elbow-excluded/" + datetime.datetime.now().strftime("%Y-%m-%d__%H-%M") + \
        '-lr-{}-epochs-{}'.format(args.learning_rate, args.epochs)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # TODO should i move this somewhere else since we have to reload data for every separate PT run anyway
    # if we are just training a random subsample or vanilla model, dont care if the data is ordered
    # Load and prepare the data
    if args.patches:
        assert args.mp4_file is not None and args.labels_file is not None, 'if using patches, must pass in mp4 file and labels file'
        normalize_images = True
        (x_train, y_train), (x_test, y_test), (x_holdout, y_holdout) = load_leap_patches(
                args.mp4_file, args.labels_file, normalize=normalize_images)
        print('Patches with shape images {}, labels {}'.format(x_train.shape, y_train.shape))
        dataset_fn = os.path.join(os.path.curdir, args.output_data, 'patches-data-{}-{}-{}-epochs-{}-ho_epochs-{}-lr-{}.json'.format(
            args.batch_size, args.big_batch_size[0], args.epochs, args.holdout_epochs, args.loss, args.learning_rate))
    else:
        print('\nRUNNING ON QMNIST\n')
        # download_qmnist()
        qmnist = unpickle(args.qmnist)
        images = qmnist['data']
        labels = qmnist['labels']
        num_classes = len(set(labels.flatten()))
        (x_train, y_train), (x_test, y_test), (x_holdout, y_holdout) = prep_mnist_data(images, labels,
                num_classes=10, embed=args.tsne)
        print('QMNIST with shape images {}, labels {}'.format(x_train.shape, y_train.shape))
        # TODO any other params to add to the filename?
        dataset_fn = os.path.join(os.path.curdir, args.output_data, 'patches-data-{}-{}-{}-epochs-{}-ho_epochs-{}-lr-{}.json'.format(
            args.batch_size, args.big_batch_size[0], args.epochs, args.holdout_epochs, args.loss, args.learning_rate))
    
    # TODO add the model dicts to a json file
    metric_dicts = []
    if args.prioritized_training:
        # Step 1: Train the holdout model
        holdout_model = ConvModelSuper().create_model(patches=args.patches, embed=args.tsne)
        holdout_model = compile_model(holdout_model, args.loss, learning_rate=args.learning_rate)
        x0 = x_holdout[10]
        holdout_model_dict, holdout_model = train_holdout_model(holdout_model, x_holdout, y_holdout, 
                x_test, y_test, args.loss, args.epochs, args.learning_rate, args.batch_size)
        print(holdout_model.summary())
        x1 = x_holdout[10]
        print(x1 - x0)
        print(np.count_nonzero(x1 - x0)) # if 0, that means x_holdout has not been shuffled

        # Step 2: Calculate "irreducible losses" for the holdout model on the training set
        #    Note: do this step before shuffling data
        il_model_loss_dict        = make_il_loss_dict(holdout_model, x_train, y_train, loss=args.loss)

        # Step 3: Train the target models using the holdout model's losses on the train set 
        pt_model_dict = train_pt_model(il_model_loss_dict, x_train, y_train, x_test, y_test,
                args.loss, args.epochs, args.learning_rate, 
                args.batch_size, args.big_batch_size,
                args.patches, args.tsne)

        metric_dicts.append(pt_model_dict)
        metric_dicts.append(holdout_model_dict)

    if args.null_hypot:
        null_hypot_model = ConvModelSuper().create_model(patches=args.patches, embed=args.tsne)
        null_hypot_model = compile_model(null_hypot_model, loss=args.loss, learning_rate=args.learning_rate)
        null_hypot_model_dict = train_null_hypothesis_model(null_hypot_model,
                x_train, y_train, x_test, y_test, 
                args.epochs, args.learning_rate, args.batch_size)
        metric_dicts.append(null_hypot_model_dict)
            
    if args.random_subsample:
        # TODO parse the big batch size arg
        rand_subsample_model_dict = train_random_subsample_model(x_train, y_train, x_test, y_test, 
                args.loss, args.epochs, args.learning_rate, 
                args.batch_size, args.big_batch_size,
                args.patches, args.tsne)
        metric_dicts.append(rand_subsample_model_dict)

    # TODO save the metric dicts to a json file
    # dataset_fn = os.path.join(os.path.curdir, args.output_data, 'data-{}-{}-{}-epochs-{}-holdout_epochs-{}.json'.format(small_batch_size, args.big_batch_size[0], epochs, holdout_epochs, args.loss))
    with open(dataset_fn, 'w', encoding='utf-8') as f:
        json.dump(metric_dicts, f, ensure_ascii=False, indent=4)
    
    # visualize how the holdout networks predictions on the test data
    #view_patch_predictions(holdout_model, x_test)
    