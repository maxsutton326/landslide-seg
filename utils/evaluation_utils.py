import csv
from datetime import date
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from sklearn.metrics import precision_recall_curve, auc
import tensorflow as tf
from utils.data_utils import reset_memory, DataGenerator, get_slice
import os
import rasterio
from rasterio.plot import reshape_as_raster

"""
This file contains utility functions used in the python-planet-tile respository. 
These functions are all used to help evaluate and make predictions with a fully
trained machine learning model.
"""


def transform_to_image(array):
    return np.rint(array * 255).clip(0, 255).astype(np.uint8)


# plot_metrics_curves
#
# plots ROC and precision recall metrics curves
#
# arg labels: numpy array with ground truth labels of tiles
# arg predictions: numpy array of predicted labels of tiles
# arg file_path: file path to save images to
# arg dataset_name: name associated with dataset, used for titling image and file path
def plot_metrics_curves(labels, predictions, file_path="", dataset_name=""):
    labels = labels.flatten()
    predictions = predictions.flatten()
    # Compute precision and recall
    print("getting precision recall", flush=True)
    precision, recall, thresholds = precision_recall_curve(
        y_true=labels, probas_pred=predictions
    )
    print("finished precision recall, getting indices", flush=True)
    print("finished indices, getting auc", flush=True)
    model_auc = round(auc(recall, precision), 5)
    print(f"auc: {model_auc}", flush=True)
    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, label=f"AUC: {model_auc}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tick_params(top=False, right=False)
    plt.title(f"Precision-Recall Curve")
    plt.gca().set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
    plt.savefig(f"{file_path}_{dataset_name}_PR.png")
    print(f"PR AUC: {model_auc}")


# get_predictions_from_locations
#
# given a set of locations to predict on, generates numpy arrays of both
#   predictions and ground truth labels corresponding to those locations.
#   These predicions and labels can then be used to evaluate the model
#   (see methods such as of predicted labels of tiles, above).
#
# arg model: trained Keras model used to predict
# arg locations: numpy array of indices of the form [xstart, xstop, ystart, ystop, zstart, zstop]
# arg images: numpy array of raw data corresponding to locations
# arg labels: numpy array with ground truth labels corresponding to locations
# arg batch_size: size of batches used to make predictions
#
# returns (predictions, patch_labels)
#   predictions: numpy array of predicted labels of tiles
#   patch_labels: numpy array with ground truth labels of tiles
def get_predictions_from_locations(
    model, locations, images, labels, batch_size, buffer_ratio=1
):
    tile_size = (
        int(locations[0][2] - locations[0][1]),
        int((locations[0][4] - locations[0][3]) / buffer_ratio),
        int((locations[0][6] - locations[0][5]) / buffer_ratio),
    )
    stride_size = int(tile_size[1] / buffer_ratio) if buffer_ratio != 1 else 0
    y_pred_probs, patch_labels = (
        np.zeros((len(locations), *tile_size)) for array in (images, labels)
    )

    reset_memory()
    dataset = DataGenerator(
        locations,
        images,
        labels,
        batch_size=batch_size,
        shuffle=False,
    )
    total = dataset.__len__()

    for i in range(total):
        batch = dataset.__getitem__(i)
        patch_labels[i : i + batch_size] = batch[1][
            :,  # batch samples
            :,  # time slices
            stride_size : (
                -stride_size if buffer_ratio != 1 else tile_size[1]
            ),  # length
            stride_size : (
                -stride_size if buffer_ratio != 1 else tile_size[2]
            ),  # width
        ]
        predictions = model(batch[0], training=False)
        y_pred_probs[i : i + batch_size] = tf.squeeze(
            predictions[
                :,
                :,
                stride_size : (
                    -stride_size if buffer_ratio != 1 else tile_size[1]
                ),  # length,
                stride_size : (
                    -stride_size if buffer_ratio != 1 else tile_size[1]
                ),  # width,
            ]
        )
        reset_memory()
        if i % 100 == 0:
            print(f"{i} batch of {total} complete", flush=True)
    # There must necessarily be 0 landslide chance in the first image in the
    # stack, since there is no "before" reference image
    y_pred_probs[:, 0] = np.zeros_like(y_pred_probs[:, 0])
    return y_pred_probs, patch_labels


def get_total_predictions(model, locations, images, labels, batch_size):
    y_pred_probs = np.zeros(
        (
            len(locations),
            int(locations[0][2] - locations[0][1]),
            int(locations[0][4] - locations[0][3]),
            int(locations[0][6] - locations[0][5]),
        )
    )
    total = len(locations) - batch_size
    for i in range(0, total, batch_size):
        y_pred_probs[i : i + batch_size] = tf.squeeze(
            model(
                np.asanyarray(
                    [get_slice(images, locations[j]) for j in range(i, i + batch_size)]
                ),
                training=False,
            )
        )
        if i % 100 == 0:
            print(f"{i} batch of {total} complete", flush=True)
    return y_pred_probs


# generate_prediction_maps
#
# reverse maps predicted probabilities of tiles to create geotiff images
#   of predicted labels
#
# arg path: file path to folder with input data
# arg patch_loc: numpy array of location of tiles in original image
# arg ground_truth_mask: numpy array with ground truth labels of tiles
# arg y_pred_probs: numpy array of predicted labels of tiles


def generate_prediction_maps(
    path,
    patch_loc,
    ground_truth_mask,
    y_pred_probs,
    buffer_ratio=1,
):
    print(f"Getting images...")

    def save_tiff(file_name, image):
        with rasterio.open(file_name, "w", **profile) as out:
            out.write(reshape_as_raster(image))

    # ground_truth_mask = np.where(ground_truth_mask > 0, 0, 1)
    with open(f"{path}/dates_{os.path.split(path)[1]}.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for line in csv_reader:
            image_date = date.fromisoformat(line[0])
            image_path = (
                f"{os.path.split(path)[1]}_{image_date.year}_{image_date.month}.tif"
            )
        num_images = csv_reader.line_num
    with rasterio.open(f"{path}/{image_path}") as src:
        src_transform = src.transform
        src_crs = src.crs
        rows, cols = src.shape
        profile = {
            "driver": "GTiff",
            "width": cols,
            "height": rows,
            "count": 4,
            "dtype": rasterio.uint8,
            "crs": src_crs,
            "transform": src_transform,
        }
    predictions = np.zeros_like(ground_truth_mask[0], dtype=y_pred_probs.dtype)
    tile_length = int((patch_loc[0][4] - patch_loc[0][3]) / buffer_ratio)
    stride_length = int(tile_length / 2) if buffer_ratio != 1 else 0
    for i in range(len(patch_loc)):
        predictions[
            patch_loc[i][1] : patch_loc[i][2],
            patch_loc[i][3] + stride_length : patch_loc[i][4] - stride_length,
            patch_loc[i][5] + stride_length : patch_loc[i][6] - stride_length,
        ] = tf.squeeze(y_pred_probs[i])
    for i in range(num_images):
        predicted = predictions[i]
        plt.imsave(
            f"{path}/output_predictions/output_{i}.png",
            transform_to_image(predicted),
            cmap="viridis",
        )
        # save_tiff(
        #     f"{path}/output_predictions/output_{i}.tif",
        #     plt.get_cmap("viridis")(predicted, bytes=True),
        # )
        labels = ground_truth_mask[0][i]
        labels = transform_to_image(labels)
        # if np.min(labels) == 0:
        #     PIL.Image.fromarray(labels, mode="L").save(
        #         f"{path}/output_predictions/labels.png"
        #     )
