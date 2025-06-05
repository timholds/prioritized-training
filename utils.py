import psutil
import time
import subprocess
import json
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.losses as losses
import gzip
import zipfile
import pickle
import random


def make_il_loss_dict(model, x_train, y_train, loss_type='categorical_crossentropy'):
    """
    Create irreducible loss dictionary by evaluating holdout model on training data.
    Fixed to use consistent loss functions.
    """
    # Map common abbreviations to full loss names
    loss_map = {
        'cce': 'categorical_crossentropy',
        'CCE': 'categorical_crossentropy',
        'categorical': 'categorical_crossentropy',
        'MSE': 'mse',
        'mean_squared_error': 'mse'
    }
    
    # Convert loss to full name if it's an abbreviation
    if loss_type in loss_map:
        loss_type = loss_map[loss_type]
        
    print(f"Computing IL losses using {loss_type}")
    
    preds = model.predict(x_train, batch_size=2048)
    
    # Use the same loss type consistently
    if loss_type == 'categorical_crossentropy':
        loss_vals = losses.categorical_crossentropy(y_train, preds).numpy()
    elif loss_type == 'mse':
        loss_vals = losses.mse(y_train, preds).numpy()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    il_loss_dict = {}
    for i, loss_val in enumerate(loss_vals):
        il_loss_dict[i] = loss_val

    print(f"Created IL loss dict with {len(il_loss_dict)} entries")
    return il_loss_dict

def shuffle_train_set_with_idx(x, y):
    """
    Create shuffled dictionary mapping indices to x,y pairs.
    Used for prioritized training batch selection.
    """
    image_label_dict = {}
    assert len(x) == len(y), "length of x and y should be the same"
    for k in range(len(x)):
        image_label_dict[k] = {'x': x[k], 'y': y[k]}

    keys = list(image_label_dict.keys())
    random.shuffle(keys)
    image_label_dict_shuffled = {key: image_label_dict[key] for key in keys}

    return image_label_dict_shuffled

def eval_current_model_on_batch(model, x_batch_candidates, y_batch_candidates, idxs, loss_type='categorical_crossentropy'):
    """
    Evaluate current model on candidate batch to compute current losses for rho calculation.
    """
    # Map common abbreviations to full loss names
    loss_map = {
        'cce': 'categorical_crossentropy',
        'CCE': 'categorical_crossentropy',
        'categorical': 'categorical_crossentropy',
        'MSE': 'mse',
        'mean_squared_error': 'mse'
    }
    
    # Convert loss to full name if it's an abbreviation
    if loss_type in loss_map:
        loss_type = loss_map[loss_type]
        
    preds = model.predict(np.array(x_batch_candidates), batch_size=64)

    # Use consistent loss type
    if loss_type == 'categorical_crossentropy':
        loss_vals = losses.categorical_crossentropy(y_batch_candidates, preds).numpy()
    elif loss_type == 'mse':
        loss_vals = losses.mse(y_batch_candidates, preds).numpy()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    loss_dict = {}
    for idx, loss_val in list(zip(idxs, loss_vals)):
        loss_dict[idx] = loss_val

    return loss_dict

