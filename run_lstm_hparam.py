#!/usr/bin/env python
# coding: utf-8

# # Offset LSTM
# 
# The objective of this notebook is to make a model that will attempt to predict overflow within the next [start] to [end] hours of a given sample. For the purposes of this test, start will be 2 hours and end will be 24 hours.
# 
# The idea is that the model will take in some given number of samples (maybe 48 hours or so) then attempt to predict if an overflow event is ocurring within [start] to [end] hours of the final sample in the given dataset. 
# 

# # Imports
# 
# In order to have this code run effectively, we will need to import and setup lots of existing libraries.
# 
# If this section of code has difficulty running, you may need to install more libraries to have it work effectively. 
# 

# In[1]:


# Loading tensorflow library
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# Import os path libraries (basic python)
import os
# Math library for some operations (basic python)
import math
# Library for shell utility functions (basic python)
import shutil
# Library for permuting lists (basic python)
import itertools

# Numpy computation libraries
import numpy as np

# pandas library for loading, creating, and manipulating data frames
import pandas as pd

# Scipy library for pre processing and computing funcations
from scipy.ndimage.filters import maximum_filter1d
from sklearn import preprocessing

# Imports from numba for parallel and optimized operations
from numba import prange, njit

# Tensorflow dependent functions and features
from tensorboard.plugins.hparams import api as hp

import argparse


# # Setup Data Files
# 
# This next section will setup the paths for specific sections of input data and logging for this training process

# In[2]:


# Code to load the dataset files
def load_dataset_from_file(file_path, key='delta_1', start = np.datetime64("2017-08-18 17:30:00"), end = np.datetime64("2018-09-05 23:58:00")):
    # Load the data file using pandas and filter between timestamps
    loaded_data = pd.read_hdf(file_path, key)

    # Filter data by timestamps
    timestamps = loaded_data["timestamp"].astype("datetime64").to_numpy()
    filtered_indices = np.logical_and(timestamps >= start, timestamps <= end)

    # load the dataset between those timestamps
    loaded_data = loaded_data.loc[filtered_indices, :].reset_index(drop=True)
    filtered_timestamps = timestamps[filtered_indices]

    return loaded_data, filtered_timestamps


# # Computing Class Labels
# 
# This next section of code will compute this class labels.
# 
# We care about overflow within the next 2-24 hours (start time to end time)
# 
# In order to generate the list of data with overflow between 2-24 hours in the future. 
# 
# To do this, compute the maximum for each 22 hour segment of data (`end - start`)
# 
# 
# Achieve this by computing a rolling window maximum across the data with a window radius of (end - start) / 2 samples long (11 hours in our case)
# Starting at 11 hours in we will have the maximum for every 22 hour period of data
# 
# Remove the first 11 and last 11 hours of data from the dataset (They have an incomplete window of data. 
# Now we have our labels aligned with our original dataset where the first (end - start) samples of data have no label. They can just be given a label of -1 or something so they will throw an error if sampled.
# 
# In addition to this, we don't care about the first 2 [start] hours of the dataset.
# To achieve this, remove the first 2 [start] hours of label data and give those samples a label of -1 as well. 
# 
# Now we need to align the samples properly. Every time there is 24 hours of data, the label at the end of the dataset will represent

# In[3]:


def compute_class_labels(
        # Dataset to draw information from
        dataset,
        # Level at which to consider flow "elevated", overflow is 546.58
        elevated_threshold = 546.25,
        # Default starting of 1 hours * 60 minutes per hour * 60 seconds per minutes * 1 sample per second
        start_offset = 1 * 60 * 60,
        # Default ending of 4 hours * 60 minutes per hour * 60 seconds per minutes * 1 sample per second
        end_offset = 4 * 60 * 60):
    # Get the level information from the dataset
    levels_dataset = dataset["outfall level"].to_numpy().astype(np.float32)
    # Length of data we care about for samples
    window_size = end_offset - start_offset + 1
    
    # Start off by computing the rolling maximum within the dataset
    rolling_max = maximum_filter1d(levels_dataset, mode="constant", cval=0.0, size=window_size)
    # Find where that rolling maximum is greater than the threshold
    elevated = (rolling_max >= elevated_threshold).astype(np.int32)
    
    # Discard the first window_size / 2 and last window_size / 2 portions of data
    half_window_size = window_size // 2
    
    # Compute the start and endpoints of our new subset
    subset_start = half_window_size
    subset_end   = levels_dataset.shape[0] - half_window_size
    # We also want to push the samples forward in time by the start_offset ammount.
    #  To achieve this, we will need to shift back the subset_end value by start_offset samples
    subset_start  += start_offset
    
    # Subset the labels for what we care about
    elevated = elevated[subset_start:subset_end]
    
    # Now we have our subset of labels that we are about, we need to align them with the actual dataset
    # This is easiest to do when the datasets are the same size. 
    # We will actually trim off the final portion of the dataset (start_offset + end_offset) since
    #   we don't have labels for them and they aren't included in any sequence. 
    # This means that the label at index 0 will correspond to the sequence which ends at index 0
    #   In other words, within the next (start_offset) to (end_offset) hours, was the elevated level achieved
    
    # Trim the end of the dataset
    dataset_start = 0
    dataset_end = elevated.shape[0] - 1
    dataset_trim = pd.DataFrame(dataset.loc[dataset_start:dataset_end,:])
    # Append our class labels to the dataset under name "Class"
    dataset_trim["Class"] = elevated
    # Return the new dataset
    return dataset_trim
    


# # Train Test Validation Split
# 
# Now that we have the labels and corresponding sequences of our dataset, we need to generate train, test, and validation splits to our dataset.
# 
# In order to achieve this, we will be using a time series generator for our dataset. We will split the dataset at three points to divide it into
# train, test, and validation data.
# 
# This step will also include any pre-processing such as normalizing the dataset to be between 0 and 1 and any other data cleaning methodologies 
# we need to improve the dataset. 
# 
# Lets split up the model into to sections, one for training and the other for testing the model. To make thing simple and avoid as much overlap or correlation between the two sections, I will just cutoff the dataset at a given date. For the purposes of this test, that cutoff will be October 2017. Then we have a few months for test and the rest of the dataset is for training purposes.
# 
# There will be three splits in the dataset, train, test, validation.
# Validation is a subset of the training dataset. 
# 
# Data will be structured in the order of `Test > Train > Validation`
# 
# | Subset | Start | End |
# |--------|-------|-----|
# | Test   | 2017-08 | 2017-09-05 |
# | Train  | 2017-09-05 | 2018-08-20 |
# | Validation | 2018-08-20 | 2018-09 |
# 

# In[4]:


def optimize_normalize_dataset(
        dataset,
        label_column = "Class",
        input_columns = ["flow", "level", "velocity", "rainfall", "outfall level"],
        min_max_scalar = None):
    input_values = dataset[input_columns].to_numpy().astype(np.float32)
    label_values = dataset[label_column].to_numpy().astype(np.int32)

    # Normalize the input values
    if min_max_scalar == None:
        min_max_scalar = preprocessing.MinMaxScaler()
        input_matrix = min_max_scalar.fit_transform(input_values)
    else:
        input_matrix = min_max_scalar.transform(input_values)
    
    # Convert labels to one hot vectors
    label_one_hot_vectors = np.zeros((label_values.shape[0], 2))
    label_one_hot_vectors[np.arange(label_values.shape[0]), label_values] = 1
    
    return input_matrix, label_one_hot_vectors, min_max_scalar

def slice_dataset(
        labeled_dataset, timestamp_column = "timestamp",
        cutoff_test = np.datetime64("2017-10"), cutoff_validation = np.datetime64("2018-08-20")):
    # Get the timestamps from the dataset
    timestamp_set = labeled_dataset[timestamp_column].astype("datetime64").to_numpy()
    # Train is after test cutoff and before validation cutoff
    train_indices = np.logical_and(timestamp_set > cutoff_test, timestamp_set <= cutoff_validation)
    # Test is before test cutoff
    test_indices = timestamp_set <= cutoff_test
    # Validation is after validation cutoff
    validation_indices = timestamp_set > cutoff_validation

    train_dataset = labeled_dataset.loc[train_indices,:]
    test_dataset = labeled_dataset.loc[test_indices,:]
    validation_dataset = labeled_dataset.loc[validation_indices,:]
    
    return train_dataset, test_dataset, validation_dataset

def get_prepared_datasets(
        dataset_file,
        sequence_length=12 * 60 * 60,
        sampling_rate= 300,
        stride = 1,
        batch_size=128,
        # Default starting of 1 hours * 60 minutes per hour * 60 seconds per minutes * 1 sample per second
        start_offset = 1 * 60 * 60,
        # Default ending of 4 hours * 60 minutes per hour * 60 seconds per minutes * 1 sample per second
        end_offset = 4 * 60 * 60):
    dataset_loaded, timestamps_loaded = load_dataset_from_file(dataset_file)
    print("Loaded dataset")

    # Get the labels for each sample/sequence in the dataset (and trim off the dataset we don't need)
    labeled_dataset = compute_class_labels(dataset_loaded, start_offset=start_offset, end_offset=end_offset)
    unique_labels, label_counts = np.unique(labeled_dataset["Class"], return_counts=True)
    print("Labeled dataset")

    # Get the train, test, validation splits
    train, test, validation = slice_dataset(labeled_dataset)

    # Make generators form these dataset splits
    train_input, train_labels, train_norm = optimize_normalize_dataset(train)
    test_input, test_labels, _ = optimize_normalize_dataset(test, min_max_scalar = train_norm)
    validation_input, validation_labels, _ = optimize_normalize_dataset(validation, min_max_scalar = train_norm)

    # Make a timeseries generator from the dataset
    train_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(
        train_input, train_labels,
        length = sequence_length, # Train it on 48 hours of data * 60 minutes per hour * 60 samples per minute
        sampling_rate = sampling_rate,   # Sampling rate is one sample per 5 minutes (* 60 seconds per minute)
        stride = stride,          # Stride between samples is 5 minutes (* 60 seconds per minute)
        shuffle = True,
        batch_size = batch_size
    )
    test_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(
        test_input, test_labels,
        length = sequence_length, # Train it on 48 hours of data * 60 minutes per hour / one sample per 5 minutes
        sampling_rate = sampling_rate,   # Sampling rate is one sample per 5 minutes (* 60 seconds per minute)
        stride = stride,          # Stride between samples is 5 minutes (* 60 seconds per minute)
        shuffle = False,
        batch_size = batch_size
    )
    validation_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(
        validation_input, validation_labels,
        length = sequence_length, # Train it on 48 hours of data * 60 minutes per hour * 60 samples per minute
        sampling_rate = sampling_rate,   # Sampling rate is one sample per 5 minutes (* 60 seconds per minute)
        stride = stride,          # Stride between samples is 5 minutes (* 60 seconds per minute)
        shuffle = False,
        batch_size = batch_size
    )
    
    return train_gen, test_gen, validation_gen


# # Setup Hyper Parameters and Model
# 
# The model requires various metrics and hyper parameters to operate correctly.
# 
# In order to have this model work as expected, this will setup the metrics for tracking the model and the various hyper parameters that can be used to vary the model.

# In[5]:


# Setup hyper parameter logging metrics
METRIC_ACCURACY  = 'accuracy'
METRIC_RECALL_0  = 'recall-0'
METRIC_PRECISION_0 = 'precision-0'
METRIC_RECALL_1  = 'recall-1'
METRIC_PRECISION_1 = 'precision-1'
METRIC_F1 = 'f1'

metric_names = [METRIC_ACCURACY, METRIC_RECALL_0, METRIC_PRECISION_0,
                METRIC_RECALL_1, METRIC_PRECISION_1, METRIC_F1]

hp_metrics = [hp.Metric(METRIC_ACCURACY, display_name='Accuracy'),
              hp.Metric(METRIC_RECALL_0, display_name='Recall-0'),
              hp.Metric(METRIC_PRECISION_0, display_name='Precision-0'),
              hp.Metric(METRIC_RECALL_1, display_name='Recall-1'),
              hp.Metric(METRIC_PRECISION_1, display_name='Precision-1'),
              hp.Metric(METRIC_F1, display_name='F1')]


# In[6]:


HP_NUM_UNITS     = hp.HParam('num_units', hp.IntInterval(4, 64))
HP_DROPOUT       = hp.HParam('dropout',   hp.RealInterval(0.0, 1.0))
HP_NUM_LAYERS    = hp.HParam('layers',    hp.Discrete([1, 2]))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1e-3]))
HP_CLASS_WEIGHTS = hp.HParam('class_weight', hp.Discrete([1, 2, 4]))
HP_START_OFFSET  = hp.HParam('start_offset', hp.RealInterval(0.0, 48.0))
HP_END_OFFSET    = hp.HParam('end_offset', hp.RealInterval(0.0, 48.0))
HP_SEQ_LENGTH    = hp.HParam('sequence_length', hp.RealInterval(0.0, 72.0))


# In[7]:


def create_lstm_model(hparams, sequence_length=None, num_classes=2):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input((sequence_length, 5)))
    for layer in range(hparams[HP_NUM_LAYERS]):
        intermediate_layer = layer < hparams[HP_NUM_LAYERS] - 1
        model.add(tf.keras.layers.LSTM(hparams[HP_NUM_UNITS], return_sequences=intermediate_layer, dropout=hparams[HP_DROPOUT]))
    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
    model.compile(loss="binary_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(learning_rate=hparams[HP_LEARNING_RATE]),
                  metrics=[
                      tf.keras.metrics.BinaryAccuracy(),
                      tf.keras.metrics.Recall(class_id=1),
                      tf.keras.metrics.AUC(),
                      tf.keras.metrics.Precision(class_id=1)])

    return model


# # Search Hyper Parameter Space
# 
# 

# In[8]:

def get_results(dataset, model):
    pred, truth = [], []
    for i in range(len(dataset)):
        test_input, test_labels = dataset[i]
        pred_labels = model(test_input).numpy()
        pred.append(np.argmax(pred_labels, axis=1))
        truth.append(np.argmax(test_labels, axis=1))
    # Combine predictions for each batch
    all_pred = np.hstack(pred)
    all_true = np.hstack(truth)

    total_samples = all_pred.shape[0]
    correct = all_pred == all_true
    incorrect = np.logical_not(correct)

    index_0 = all_true == 0
    index_1 = all_true == 1
    pred_0 = all_pred == 0
    pred_1 = all_pred == 1

    recall_0 = np.sum(np.logical_and(correct, index_0)) / np.sum(index_0)
    recall_1 = np.sum(np.logical_and(correct, index_1)) / np.sum(index_1)
    precision_0 = np.sum(np.logical_and(correct, index_0)) / np.sum(pred_0) if np.sum(pred_0) > 0 else 0
    precision_1 = np.sum(np.logical_and(correct, index_1)) / np.sum(pred_1) if np.sum(pred_1) > 0 else 0
    f1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
    accuracy = np.sum(correct) / total_samples
    results = {METRIC_ACCURACY: accuracy,
               METRIC_RECALL_0: recall_0,
               METRIC_RECALL_1: recall_1,
               METRIC_PRECISION_0: precision_0,
               METRIC_PRECISION_1: precision_1,
               METRIC_F1: f1}
    return results

def evaluate_model(
        run_dir, hparams, 
        train_data, val_data, test_data,
        complete_file,
        sequence_length,
        checkpoint_path,
        model_path,
        num_classes = 2,
        verbose=1,
        epochs =100,
        early_stop_threshold=10,
        min_delta_early_stopping=1e-3):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', mode='min',
        verbose=verbose, patience=early_stop_threshold,
        min_delta=min_delta_early_stopping, restore_best_weights=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = run_dir)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, mode='min',
        verbose=verbose, save_freq='epoch',
        save_best_only=True)

    # clear out existing logs
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)

    results = {}
    tf.keras.backend.clear_session()
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        evaluation_model = create_lstm_model(hparams, sequence_length, num_classes)
        if verbose > 0:
            evaluation_model.summary()

        print("Fitting model")
        history_callback = evaluation_model.fit(
            train_data,
            epochs=epochs,
            validation_data=val_data,
            callbacks=[early_stopping,
                tensorboard_callback,
                hp.KerasCallback(run_dir, hparams)],
            verbose=verbose,
            class_weight={
                0: 1,
                1: hparams[HP_CLASS_WEIGHTS]
            })
        

        # Save final model
        evaluation_model.save(model_path)
        results = get_results(val_data, evaluation_model)
        test_results = get_results(test_data, evaluation_model)
        
        if verbose > 0:
            print(results)
        tf.summary.scalar(METRIC_ACCURACY, results[METRIC_ACCURACY], step=1)
        tf.summary.scalar(METRIC_RECALL_0, results[METRIC_RECALL_0], step=1)
        tf.summary.scalar(METRIC_RECALL_1, results[METRIC_RECALL_1], step=1)
        tf.summary.scalar(METRIC_PRECISION_0, results[METRIC_PRECISION_0], step=1)
        tf.summary.scalar(METRIC_PRECISION_1, results[METRIC_PRECISION_1], step=1)
        tf.summary.scalar(METRIC_F1, results[METRIC_F1], step=1)
        
        if not os.path.exists(os.path.dirname(complete_file)):
            os.makedirs(os.path.dirname(complete_file))
        with open(complete_file, 'w') as complete_data:
            complete_data.write("validation: " + str(results) + "\n")
            complete_data.write("test: " + str(test_results) + "\n")
    return results

# In[9]:


# Project path
project_path = "."

# Run name
run_name = "lstm_hparams"

# Path of various folders for hyper parmaters and model data
results_dir = os.path.join(project_path, "Dataset-Analysis", run_name)
checkpoint_dir = os.path.join(results_dir, "checkpoints")
model_dir = os.path.join(results_dir, "models")
log_dir = os.path.join(results_dir, "logs")

synchronized_data_file = os.path.join(project_path, "Datasets-Synchronized", "60-sec-norm.hdf")
samples_per_minute = 1

parser = argparse.ArgumentParser(description="Train LSTM Model from parameters")
parser.add_argument('--end_offset', type=float, help="end offset for predicting overflow in hours")
parser.add_argument('--start_offset', type=float, help="start offset for predicting overflow in hours")
parser.add_argument('--seq_len', type=float, help="length of the sequence in hours")
parser.add_argument('--num_units', type=int, help="number of units found in the LSTM")
parser.add_argument('--dropout', type=float, help="dropout value to use for the LSTM")
parser.add_argument('--num_layers', type=int, help="number of layers to use in the LSTM")
parser.add_argument('--class_weight', type=int, help="weight of classes to use when training")
parser.add_argument('--learning_rate', type=float, help="learning rate for training the lstm model")
parser.add_argument('--batch_size', type=int, default=128, help="batch size to use while training", required=False)

args=parser.parse_args()

start_offset_hours = args.start_offset #1
end_offset_hours = args.end_offset #25
sequence_length_hours = args.seq_len #12
num_units = args.num_units #8
dropout = args.dropout #0.1
num_layers = args.num_layers #1
class_weight = args.class_weight #1
learning_rate = args.learning_rate #1e-3
batch_size = args.batch_size #128

start_offset_minutes = int(start_offset_hours * 60)
end_offset_minutes = int(end_offset_hours * 60)
sequence_length_minutes = int(sequence_length_hours * 60)
sampling_rate = 5 * samples_per_minute
start_offset = start_offset_minutes * samples_per_minute
end_offset = end_offset_minutes * samples_per_minute
sequence_length_samples = sequence_length_minutes // sampling_rate

# Setup directory for logging hyperparameters and logs
with tf.summary.create_file_writer(log_dir).as_default():
    hp.hparams_config(
        hparams=[HP_NUM_UNITS,
            HP_DROPOUT,
            HP_NUM_LAYERS,
            HP_LEARNING_RATE,
            HP_CLASS_WEIGHTS,
            HP_START_OFFSET,
            HP_END_OFFSET,
            HP_SEQ_LENGTH],
        metrics=hp_metrics)

train_gen, test_gen, validation_gen = get_prepared_datasets(
    synchronized_data_file, sequence_length=sequence_length_minutes,
    batch_size=batch_size, sampling_rate=sampling_rate, stride=1,
    start_offset=start_offset, end_offset=end_offset)

# In[ ]:

assert HP_NUM_UNITS.domain.min_value <= num_units < HP_NUM_UNITS.domain.max_value
assert HP_DROPOUT.domain.min_value <= dropout <= HP_DROPOUT.domain.max_value
assert num_layers in HP_NUM_LAYERS.domain.values
assert class_weight in HP_CLASS_WEIGHTS.domain.values
assert learning_rate in HP_LEARNING_RATE.domain.values

# Train a model given these settings
hparams = {
    HP_NUM_UNITS: num_units,
    HP_DROPOUT: dropout,
    HP_NUM_LAYERS: num_layers,
    HP_CLASS_WEIGHTS: class_weight,
    HP_LEARNING_RATE: learning_rate,
    HP_START_OFFSET : start_offset_hours,
    HP_END_OFFSET   : end_offset_hours,
    HP_SEQ_LENGTH   : sequence_length_hours,
}

# run this five times to get a good, random sample
final_results = {metric: [] for metric in metric_names}
run_group = "units-%i_dropout-%0.2f_layers-%i_class-weight-%i_lr-%0.5f_start-%2.2f_end-%2.2f_len-%2.2f" % (
    num_units, dropout, num_layers, class_weight, learning_rate, start_offset_hours, end_offset_hours, sequence_length_hours)
for run_number in range(5):
    run_name = "%s_run-%i" % (
        run_group, run_number
    )
    complete_file = os.path.join(log_dir, "complete", run_name + ".txt")
    print('--- Starting trial: %s' % run_name)
    print({h.name: hparams[h] for h in hparams})
    # Skip the run if folder already exists
    if os.path.exists(complete_file):
        # skip this run
        print("Skipping run complete file exists")
        continue
    results = evaluate_model(
        os.path.join(log_dir, run_name), hparams,
        train_gen, validation_gen, test_gen, complete_file,
        sequence_length_samples,
        checkpoint_path = os.path.join(checkpoint_dir, run_name, "checkpoint", "weights-improvement-{epoch:03d}.hdf5"),
        model_path = os.path.join(model_dir, run_name),
        num_classes = 2,
        verbose=2)

