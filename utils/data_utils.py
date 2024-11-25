import gc
import numpy as np
import os
import random
import tensorflow as tf

"""
This file contains utility functions used in the python-planet-tile respository. 
These functions are all used to help manipulate and analyze data used in training
a machine learning model.
"""


# random_samples
#
# arg arrays: arrays to take random samples from
# arg num_samples: number of samples to return. If int, returns that many samples. If float,
#   returns that percentage of samples.
#
# returns map of arrays, each with same index of random samples
def random_samples(*arrays, num_samples):
    total_num = len(arrays[0])
    if type(num_samples) == float:
        num_samples = int(total_num * num_samples)
    elif type(num_samples) != int:
        raise TypeError(
            f"num_samples argument should be an int or float. Received: {type(num_samples)}"
        )
    if total_num < num_samples:
        return arrays
    idx = np.arange(total_num)
    np.random.shuffle(idx)
    return [array[idx[:num_samples]] for array in arrays]


# get_sample_weights
#
# It is helpful to use weights when the data is imbalanced. This function, given a 2d list of labels,
#   will calculate the sample weight for each item.
#
# arg y_train: labels to get weights for
# arg use_temporal: for a temporal stack of images, whether to label each time step with a separate weight.
def get_sample_weights(locations, labels):
    num_samples = len(locations)
    counts = np.zeros(100)
    sample_weights = np.zeros((num_samples, locations[0][2] - locations[0][1]))
    for i in range(num_samples):
        location = locations[i]
        for j in range(location[1], location[2]):
            num_positive_pixels = int(
                np.sum(
                    labels[
                        location[0],
                        j,
                        location[3] : location[4],
                        location[5] : location[6],
                    ]
                )
            )
            # -(a // -b) is "upside-down floor division" ie ceiling division
            weight_index = -((num_positive_pixels * 100) // -(128 * 128))
            counts[weight_index] += 1
            sample_weights[i][j - location[1]] = weight_index
    # see sklearn.utils.class_weight.compute_class_weight
    num_weights = np.count_nonzero(counts)
    weights = [
        num_samples / (num_weights * count) if count > 0 else 0 for count in counts
    ]
    sample_weights = [
        [weights[int(j)] for j in range(len(sample_weights[i]))] for i in sample_weights
    ]
    return np.asanyarray(sample_weights)


# get_dataset_weights
#
# It is helpful to use weights when the data is imbalanced. This function, given a list of datasets,
#   will calculate the sample weight for each dataset.
#
# arg dataset_sizes: length of datasets to get weights for
#
# returns array of weights corresponding to size of each dataset
def get_dataset_weights(dataset_sizes):
    unique_labels = len(dataset_sizes)
    total_samples = sum(dataset_sizes)

    # return weights
    def calculate_weight(dataset_size):
        # see sklearn.utils.class_weight.compute_class_weight
        return total_samples / (unique_labels * dataset_size)

    return np.concatenate(
        tuple(np.full(size, calculate_weight(size)) for size in dataset_sizes)
    )


# positive_negative_split
#
# Samples positive data and a proportion of negative data
#
# arg raw_data: data to sample from. Shape (input, labels, *other)
# arg proportion: float representing proportion of negative points to positive points
#
# returns list with positive and negative samples
def positive_negative_split(
    model_input,
    model_labels,
    *other_select,
    proportion_n=1,
    proportion_p=1,
    positive_threshold=0,
    axis=1,
):
    if len(model_input) != len(model_labels):
        raise IndexError("Must have same number of input and labels")
    total_data = (model_input, model_labels, *other_select)
    print(f"{len(model_input)} raw points, proportion = {proportion_n}")
    # Get only positive items
    nonzero_indices = np.where(
        model_labels[:, 1:].sum(axis=axis) > positive_threshold, 1, 0
    ).nonzero()[0]
    positive_select = np.isin(range(model_labels.shape[0]), nonzero_indices)
    positive_points = [array[positive_select] for array in total_data]
    print(f"{len(positive_points[0])} positive points, proportion = {proportion_p}")
    positive_points = list(
        random_samples(
            *positive_points,
            num_samples=(int(len(positive_points[0]) * proportion_p)),
        )
    )
    print(f"{len(positive_points[0])} positive points, proportion = {proportion_p}")

    # Get negative samples
    negative_points = [array[~positive_select] for array in total_data]
    print(
        f"{len(negative_points[0])} negative points total, proportion = {proportion_n}"
    )
    negative_points = list(
        random_samples(
            *negative_points,
            num_samples=(int(len(positive_points[0]) * proportion_n)),
        )
    )
    print(f"{len(negative_points[0])} negative points, proportion = {proportion_n}")

    # Combine the two
    num_datasets = len(total_data)
    return [
        np.append(positive_points[i], negative_points[i], axis=0)
        for i in range(num_datasets)
    ]


def training_validation_split(locations, proportion=0.5):
    locations = np.asanyarray(locations)
    training_mask = np.ma.make_mask(np.empty((locations.shape[0],)))
    validation_mask = np.ma.make_mask(np.empty((locations.shape[0],)))
    for i in range(np.max(locations[:, 0]) + 1):
        dataset_i = locations[:, 0] == i
        locations_x = locations[dataset_i][:, 4]
        max_location = np.max(locations_x)
        half = max_location / 2
        half_len = max_location * (proportion / 2)
        training_mask[dataset_i] = np.logical_and(
            locations_x < half + half_len,
            locations_x > half - half_len,
        )
        validation_mask[dataset_i] = locations_x > half + half_len

    # return training, validation, test
    return (
        locations[training_mask],
        locations[np.logical_and(~training_mask, validation_mask)],
        locations[np.logical_and(~training_mask, ~validation_mask)],
    )


# load_data_slices
#
# Given a path to a folder containing data, loads data to be used in training a machine
#   learning model. This method can take multiple paths and will concatinate the data
#   together into one dataset to be used in the model.
# This method assumes that you are loading tile locations to be fed into the model. If you are loading
#   full tile data instead, please use method load_data.
#
# arg paths: filepath from to numpy files with patches and labels.
# arg suffix: suffix to add onto standard file names
# arg get_weights: whether to calculate sample weights based on the size of each dataset
def load_data_slices(paths, suffix="", get_weights=False):
    locations_file = f"patches_loc{suffix}"
    data_files = [f"images", "labels", locations_file]
    location_index = data_files.index(locations_file)
    num_data_files = len(data_files)
    full_data = {data: [] for data in data_files}
    # Load the stack of images
    dataset_lengths = []
    for i in range(len(paths)):
        path = paths[i]
        for j in range(num_data_files):
            file = data_files[j]
            intermediate_data = np.asanyarray(np.load(f"{path}/{data_files[j]}.npy"))
            if j == location_index:
                intermediate_data = np.asanyarray(
                    [[i, *data] for data in intermediate_data]
                )
                dataset_lengths.append(len(intermediate_data))
                full_data[file].extend(intermediate_data)
            else:
                if j == 0:  # images
                    # Normalize RGBA bands (but not any additional one hot encoding)
                    intermediate_data = intermediate_data.astype(np.float16)
                    intermediate_data[:4] = intermediate_data[:4] / 255.0
                full_data[file].append(intermediate_data)
    full_data = [full_data[key] for key in data_files]
    full_data[location_index] = np.asanyarray(full_data[location_index])
    if get_weights:
        return (*full_data, get_dataset_weights(dataset_lengths))
    return full_data


# get_slice
#
# Returns a slice of the given array according to the given indices.
#
# arg array: array to get slice of
# arg indices: array of indices of the form [array_index, xstart, xstop, ystart, ystop, zstart, zstop]
def get_slice(array, indices):
    return np.asanyarray(array[indices[0]])[
        indices[1] : indices[2],
        (indices[3]) : (indices[4]),
        (indices[5]) : (indices[6]),
    ]


# There is a known memory leak in Tensorflow when used with generators,
# so we pass this callback to keep the memory from overflowing.
def reset_memory():
    # tf.keras.backend.clear_session()
    gc.collect()


class MemoryCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(MemoryCallback, self).__init__()
        os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    def on_train_batch_end(self, batch, logs=None):
        reset_memory()

    def on_test_batch_end(self, batch, logs=None):
        reset_memory()

    def on_predict_batch_end(self, batch, logs=None):
        reset_memory()

    def on_epoch_end(self, epoch, logs=None):
        reset_memory()


# The data is too large to be loaded into memory all at once, or to store as one file.
#   So, we instead save the locations of each tile and get each tile from the location
#   data on a need to know basis.
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, indices, *arrays, batch_size, shuffle=True, aggregate=False):
        self.indices = indices
        # self.arrays[0] = input data
        # self.arrays[1] = ground truth
        self.arrays = arrays
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng()
        self.aggregate = aggregate
        self.n = len(self.indices)

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __getitem__(self, index):
        start_index = index * self.batch_size
        should_flip_batch = random.random() > 0.5
        rotation_factor = random.choice([0, 1, 2, 3])

        def augment_image(image):
            if self.shuffle:
                if should_flip_batch:
                    image = np.flip(image, 2)
                image = np.rot90(image, rotation_factor, axes=(1, 2))
            return image

        slices = list(
            np.asanyarray(
                [
                    (
                        augment_image(
                            get_slice(
                                array,
                                self.indices[i],
                            ),
                        )
                        if len(np.asanyarray(array[0]).shape) > 2
                        # If array cannot be sliced, it must be the array of weights
                        else array[i]
                    )
                    for i in range(start_index, start_index + self.batch_size)
                ]
            )
            for array in self.arrays
        )
        # There must necessarily be 0 landslide chance in the first image in the
        # stack, since there is no "before" reference image
        slices[1][:, 0] = np.zeros_like(slices[1][:, 0])
        if self.aggregate:
            slices[1] = np.where(
                (
                    slices[1].sum(axis=(2, 3))
                    / (slices[0].shape[-1] * slices[0].shape[-1])
                )
                > 0.05,
                1,
                0,
            )
            slices.append(np.where(slices[1] > 0, 0.95, 0.05))
            slices[2] = np.expand_dims(slices[2], 2)
        return slices

    def __len__(self):
        return self.n // self.batch_size


def get_empty_sample_mask(locations, array):
    location_stats = np.zeros(len(locations), dtype=np.uint8)
    squashed = []

    for i in range(len(array)):
        array_to_squash = array[i]

        def get_squashed_array():
            return np.count_nonzero(np.asanyarray(array_to_squash), axis=-1)

        squashed.append(np.count_nonzero(np.asanyarray(array_to_squash), axis=-1))
    pixels_per_patch = (
        (locations[0][2] - locations[0][1])
        * (locations[0][4] - locations[0][3])
        * (locations[0][6] - locations[0][5])
    )
    for i in range(len(locations)):
        location = locations[i]
        patch = get_slice(squashed, location)
        if np.count_nonzero(patch) > pixels_per_patch * 0.8:
            location_stats[i] = 1
    return location_stats
