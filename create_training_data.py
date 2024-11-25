###################
## create_training_data.py
## Written pre-CS230
## Prepares data for use in machine learning model.
## - Converts Geotiff satellite data into numpy format
## - Creates numpy mask of ground truth landslides based on shapefile
## - Divides area of interest into tiles/patches. Because these tiles
##      overlap, the total size of the tiles is larger than the raw data.
##      To avoid memory issues, I instead record and save the indices of
##      tiles, so that they can easily be referenced later.
import csv
from datetime import date
import geopandas as gpd
import numpy as np
import os
import pandas as pd
import rasterio
from rasterio.features import geometry_mask, sieve
from rasterio.enums import Resampling
from rasterio.plot import reshape_as_image
from sklearn.preprocessing import OneHotEncoder
import sys

scratch_dir = os.getenv("GROUP_SCRATCH")


def read_image_stack(image_paths):
    # Read satellite images and stack them into a 3D array
    # image_paths: list of paths to satellite images sorted by time
    with rasterio.open(image_paths[0]) as src:
        rows, cols = src.shape
        time_bands = len(image_paths)
        image_bands = src.count
        image_stack = np.zeros(
            (time_bands, rows, cols, image_bands),
            dtype=np.float16,
        )
        transform = src.transform
        crs = src.crs

    for i, image_path in enumerate(image_paths):
        with rasterio.open(image_path) as src:
            image_stack[i] = reshape_as_image(src.read())

    print(f"Read image stack, shape {image_stack.shape}")
    np.save(f"{os.path.dirname(path)}/images.npy", image_stack)

    return image_stack, transform, crs


def create_ground_truth_mask(image_stack, image_dates, polygon_data, transform):
    # Create a 3D ground truth mask from polygon data
    #
    # image_stack: 3D array of satellite images
    # polygon_data: geopandas dataframe containing landslide polygons
    # landslide_time: integer value representing the index in the image stack that the landslide corresponds to.
    # transform: Affine transform of images in image_stack
    should_remove_small_shapes = False
    time_bands, rows, cols, image_bands = image_stack.shape
    ground_truth_mask = np.zeros((time_bands, rows, cols), dtype=np.int8)

    def get_closest_date_index(event_date):
        deltas = np.zeros(time_bands)
        for i in range(time_bands):
            deltas[i] = int((date.fromisoformat(event_date) - image_dates[i]).days)
            if i != 0 and (
                (deltas[i] * deltas[i - 1]) < 0 or (deltas[i] * deltas[i - 1]) == 0
            ):
                return i
        return 0

    event_indices = np.array(
        [get_closest_date_index(event_date) for event_date in polygon_data["ev_date"]]
    )

    for i in range(time_bands):
        monthly_polygon_data = polygon_data[event_indices == i]
        if len(monthly_polygon_data) > 1:
            shape_mask = geometry_mask(
                monthly_polygon_data.geometry,
                out_shape=(rows, cols),
                transform=transform,
                invert=True,
                all_touched=True,
            )
            if should_remove_small_shapes:
                shape_mask = sieve(shape_mask.astype(rasterio.int8), 4).astype(bool)
            ground_truth_mask[i] += shape_mask

    return ground_truth_mask


# def create_patch_list
#
# Creates a list of smaller patches from input data for use in a machine learning model
#
# arg input_data: 3d np array of input data
# arg patch_size: tuple of ints representing how big each patch is in each dimension
# arg stride_size: tuple of ints representing how big each stride is in each dimension
#   (how much the window "slides" over the image in each iteration)
# arg output_name: name of output file in which to save patch information
# arg should_classify: if true, gives one label for entire tile of whether feature is present,
#   rather than copying over pixels.
# arg should_partition: whether to partition into different files for eaiser storage
def create_patch_list(
    input_data,
    output_name,
    patch_size=(10, 64, 64),
    stride_size=(4, 64, 64),
):
    num_files_saved = 0
    should_remove_empty_patches = True

    def save_patches(patch_list, file_suffix=""):
        # Convert the list to numpy arrays
        patch_list = np.array(patch_list)
        # Print the shape of the extracted patches and labels
        print(f"{output_name}{file_suffix} shape: {patch_list.shape}")
        # Save the patches and labels to file or use them directly for training your 3D CNN
        np.save(f"{output_name}{file_suffix}.npy", patch_list)

    # Initialize list to store patches
    patch_loc = []
    # Slide the patch window across the image.
    for y in range(0, input_data.shape[1] - patch_size[1] + 1, stride_size[1]):
        for z in range(0, input_data.shape[0] - patch_size[0] + 1, stride_size[0]):
            for x in range(0, input_data.shape[2] - patch_size[2] + 1, stride_size[2]):
                # Iterate over the image stack and extract patches
                (zindex, yindex, xindex) = (
                    slice(z, z + patch_size[0]),
                    slice(y, y + patch_size[1]),
                    slice(x, x + patch_size[2]),
                )
                patch = input_data[(zindex, yindex, xindex)]
                if not should_remove_empty_patches or np.any(patch):
                    patch_loc.append(
                        [
                            zindex.start,
                            zindex.stop,
                            yindex.start,
                            yindex.stop,
                            xindex.start,
                            xindex.stop,
                        ]
                    )
    save_patches(patch_loc, file_suffix="_loc_384")


# Command line arguments:
#
# [1]: File path to csv file containing list of images. The images must be in the same folder as
#   the csv file. A line in the csv containing a second row item is assumed to be the date of
#   landslide occurance.
# [2]: Output file path for patches and labels files
# [3+]: File paths to shapefiles containing landslide shapefiles
if __name__ == "__main__":
    # Initialize variables
    landslide_time = 0
    image_paths = []
    image_dates = []
    image_info = sys.argv[1]
    path = f"{sys.argv[2]}"

    # Read image paths and polygon data
    with open(image_info) as csv_file:
        base_name = os.path.basename(path)
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            image_date = date.fromisoformat(row[0])
            file_name = f"{base_name}_{image_date.year}_{image_date.month}.tif"
            image_paths.append(os.path.join(os.path.dirname(path), file_name))
            image_dates.append(image_date)

    # Read and preprocess the data
    image_stack, image_transform, image_crs = read_image_stack(image_paths)

    # Define parameters
    patches = (10, 384, 384)  # Size of the patches (width, height)
    stride = (1, 128, 128)  # Stride between consecutive patches (width, height)

    # Create patches for input data
    create_patch_list(
        image_stack,
        f"{os.path.dirname(path)}/patches",
        patch_size=patches,
        stride_size=stride,
    )

    for i in range(3, len(sys.argv)):
        data = gpd.read_file(sys.argv[i]).to_crs(image_crs)
        data = data[
            data.geometry.apply(
                lambda x: x.geom_type == "MultiPolygon" or x.geom_type == "Polygon"
            )
        ]
        if data.size == 0:
            raise (
                BaseException(
                    "Landslide data should contain polygons (Is your data series only points?)"
                )
            )
        if i == 3:
            polygon_data = data
        else:
            polygon_data = pd.concat([polygon_data, data])

    # Create labels
    # labels = create_ground_truth_mask(
    #     image_stack, image_dates, polygon_data, image_transform
    # ).astype(np.int8)
    # np.save(f"{os.path.dirname(path)}/labels.npy", labels)

    # # Create patches for labels
    # print(f"Created ground truth mask, shape {labels.shape}")
