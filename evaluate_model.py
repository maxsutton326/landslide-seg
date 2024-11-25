import argparse
import os
import tensorflow as tf
from utils.data_utils import (
    load_data_slices,
    training_validation_split,
    get_empty_sample_mask,
)
from utils.evaluation_utils import (
    plot_metrics_curves,
    get_predictions_from_locations,
    get_total_predictions,
    generate_prediction_maps,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        required=True,
        help="Filepaths to numpy files with patches and labels.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Filepath to model to evaluate.",
    )
    parser.add_argument(
        "--should-generate-map",
        default=False,
        type=bool,
        help="Whether to generate full map of model predictions.",
    )
    return parser.parse_args()


# Evaluates trained machine learning model and prints statistics
if __name__ == "__main__":
    # Command line arguments:
    #
    # [1] relative file path to model
    # [2] file path to folder with test data
    args = get_args()
    print(args)
    use_locations = True
    model = tf.keras.models.load_model(args.model)
    path = args.path

    # Generate predictions on the validation data
    print("Predicting...", flush=True)
    batch_size = 8
    image_stack, ground_truth_mask, locs = load_data_slices([path], suffix="_128")
    # Since the satellite images are not rectangles, there are some locations that
    # without RGBA data. We exclude those locations.
    mask = get_empty_sample_mask(locs, image_stack)
    loc_total = locs[mask != 0]
    if args.should_generate_map:
        y_pred_probs = get_total_predictions(
            model=model,
            locations=loc_total,
            images=image_stack,
            labels=ground_truth_mask,
            batch_size=batch_size,
        )
        print("Generating prediction maps...", flush=True)
        generate_prediction_maps(
            path,
            loc_total,
            ground_truth_mask,
            y_pred_probs,
            buffer_ratio=2,
        )
        exit(0)
    else:
        loc_train, loc_val, loc_test = training_validation_split(
            loc_total, proportion=0.7
        )
        loc_total = loc_test

        y_pred_probs, labels = get_predictions_from_locations(
            model=model,
            locations=loc_total,
            images=image_stack,
            labels=ground_truth_mask,
            batch_size=batch_size,
            buffer_ratio=2,
        )
    output_plot_path = f"{os.path.splitext(args.model)[0]}"
    print("Plotting metrics...", flush=True)
    plot_metrics_curves(
        labels=labels,
        predictions=y_pred_probs,
        file_path=output_plot_path,
        dataset_name=os.path.basename(path).capitalize(),
    )
