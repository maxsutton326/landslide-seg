import argparse
import numpy as np
import os
from utils.data_utils import (
    positive_negative_split,
    load_data_slices,
    get_slice,
    MemoryCallback,
    DataGenerator,
    training_validation_split,
    random_samples,
)
import tensorflow as tf
from utils.model import get_unet


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        nargs="*",
        required=True,
        help="Filepaths to numpy files with patches and labels.",
    )
    parser.add_argument(
        "--epochs", default=256, type=int, help="number of epochs to train model for."
    )
    parser.add_argument(
        "--alpha",
        default=0.85,
        type=float,
        help="value of alpha to use in binary crossentropy loss, used in training the model.",
    )
    parser.add_argument("--name", required=True, help="File to save model in.")
    parser.add_argument(
        "--pos-split",
        default=1,
        type=float,
        help="Proportion of positive samples to use in training.",
    )
    parser.add_argument(
        "--neg-split",
        default=1,
        type=float,
        help="Number of negative samples to use in training, calculated as a proportion of positive samples.",
    )
    parser.add_argument(
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--model",
        required=False,
        help="Path to existing model to retrain",
    )
    parser.add_argument(
        "--training-proportion",
        default=0.5,
        type=float,
        help="What fraction of the dataset to use for training, default=0.5",
    )
    return parser.parse_args()


BATCH_SIZE = 8

# Trains a machine learning model
if __name__ == "__main__":
    args = get_args()
    model_path = f"{args.name}.keras"
    print(args)
    # Load the stack of images
    image_stack, ground_truth_mask, locations = load_data_slices(args.data, suffix="")
    loc_train, loc_val, loc_test = training_validation_split(
        locations, proportion=args.training_proportion
    )

    # Normalize by size of dataset, when training with multiple landslide inventories
    num_inventories = np.max(loc_train[:, 0]) + 1
    data_lengths = np.array(
        [len(loc_train[loc_train[:, 0] == i]) for i in range(num_inventories)]
    )
    smallest_inventory = int(np.sort(data_lengths)[0])
    loc_train, _ = positive_negative_split(
        loc_train,
        np.asanyarray(
            [
                get_slice(ground_truth_mask, location).any(axis=1)
                for location in loc_train
            ]
        ),
        proportion_n=args.neg_split,
        proportion_p=args.pos_split,
    )
    loc_test = np.concatenate(
        [
            np.array(
                random_samples(
                    loc_test[loc_test[:, 0] == i],
                    num_samples=smallest_inventory,
                )
            )[0]
            for i in range(num_inventories)
        ]
    )
    print(f"num validation points: {len(loc_test)}")

    # Split the data into training and validation sets
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")
    with strategy.scope():
        if args.model:
            model = tf.keras.models.load_model(args.model, compile=False)
        else:
            # Define the 3D-CNN model
            sample_shape = (
                int(loc_train[0][2] - loc_train[0][1]),
                int(loc_train[0][4] - loc_train[0][3]),
                int(loc_train[0][6] - loc_train[0][5]),
                *np.asanyarray(image_stack[0]).shape[3:],
            )
            model = get_unet(input_shape=sample_shape)

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=tf.keras.losses.BinaryFocalCrossentropy(
                apply_class_balancing=True, alpha=args.alpha
            ),
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.AUC(curve="PR"),
            ],
            steps_per_execution=8,
        )

        # Define model parameters
        num_val = int(len(loc_test) / BATCH_SIZE)
        num_train = int(len(loc_train) / BATCH_SIZE)
        model_params = {
            "workers": 16,
            "use_multiprocessing": True,
            "callbacks": [
                MemoryCallback(),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_auc",
                    patience=4,
                    factor=0.4,
                    min_lr=1e-07,
                    min_delta=1e-3,
                    mode="max",
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    model_path,
                    monitor="val_loss",
                    verbose=0,
                    save_best_only=False,
                    save_weights_only=False,
                    mode="auto",
                    save_freq="epoch",
                    initial_value_threshold=None,
                ),
            ],
        }

        # Train the model
    history = model.fit(
        x=DataGenerator(
            loc_train,
            image_stack,
            ground_truth_mask,
            batch_size=BATCH_SIZE,
        ),
        steps_per_epoch=num_train,
        epochs=args.epochs,
        validation_data=DataGenerator(
            loc_test,
            image_stack,
            ground_truth_mask,
            batch_size=BATCH_SIZE,
        ),
        validation_steps=num_val,
        verbose=2,
        **model_params,
    )

    model.save(model_path)
