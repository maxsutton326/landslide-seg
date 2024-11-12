import tensorflow as tf


def get_unet(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    # Encoder: Contracting Path
    # Block 1
    conv1 = tf.keras.layers.Conv3D(
        64,
        (3, 3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(0.005),
    )(inputs)
    conv1 = tf.keras.layers.Conv3D(
        64,
        (3, 3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(0.005),
    )(conv1)
    pool1 = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(conv1)
    pool1 = tf.keras.layers.Dropout(0.2)(pool1)

    # Block 2
    conv2 = tf.keras.layers.Conv3D(
        128,
        (3, 3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(0.005),
    )(pool1)
    conv2 = tf.keras.layers.Conv3D(
        128,
        (3, 3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(0.005),
    )(conv2)
    pool2 = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(conv2)
    pool2 = tf.keras.layers.Dropout(0.2)(pool2)

    # Block 3
    conv3 = tf.keras.layers.Conv3D(
        256,
        (3, 3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(0.005),
    )(pool2)
    conv3 = tf.keras.layers.Conv3D(
        256,
        (3, 3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(0.005),
    )(conv3)
    pool3 = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(conv3)
    pool3 = tf.keras.layers.Dropout(0.2)(pool3)

    # Block 4 (Bottleneck)
    conv4 = tf.keras.layers.Conv3D(
        512,
        (3, 3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(0.005),
    )(pool3)
    conv4 = tf.keras.layers.Conv3D(
        512,
        (3, 3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(0.005),
    )(conv4)
    conv4 = tf.keras.layers.Dropout(0.2)(conv4)

    # Decoder: Expanding Path
    # Block 5
    up5 = tf.keras.layers.Conv3DTranspose(
        256, (3, 3, 3), strides=(1, 2, 2), activation="relu", padding="same"
    )(conv4)
    up5 = tf.keras.layers.Conv3D(
        256,
        (3, 3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(0.005),
    )(up5)
    merge5 = tf.keras.layers.concatenate([conv3, up5], axis=-1)
    conv5 = tf.keras.layers.Conv3D(
        256,
        (3, 3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(0.005),
    )(merge5)
    conv5 = tf.keras.layers.Conv3D(
        256,
        (3, 3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(0.005),
    )(conv5)
    conv5 = tf.keras.layers.Dropout(0.2)(conv5)

    # Block 6
    up6 = tf.keras.layers.Conv3DTranspose(
        128, (3, 3, 3), strides=(1, 2, 2), activation="relu", padding="same"
    )(conv5)
    up6 = tf.keras.layers.Conv3D(
        128,
        (3, 3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(0.005),
    )(up6)
    merge6 = tf.keras.layers.concatenate([conv2, up6], axis=-1)
    conv6 = tf.keras.layers.Conv3D(
        128,
        (3, 3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(0.005),
    )(merge6)
    conv6 = tf.keras.layers.Conv3D(
        128,
        (3, 3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(0.005),
    )(conv6)
    conv6 = tf.keras.layers.Dropout(0.2)(conv6)

    # Block 7
    up7 = tf.keras.layers.Conv3DTranspose(
        64,
        (3, 3, 3),
        strides=(1, 2, 2),
        activation="relu",
        padding="same",
    )(conv6)
    up7 = tf.keras.layers.Conv3D(
        64,
        (3, 3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(0.005),
    )(up7)
    merge7 = tf.keras.layers.concatenate([conv1, up7], axis=-1)
    conv7 = tf.keras.layers.Conv3D(
        64,
        (3, 3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(0.005),
    )(merge7)
    conv7 = tf.keras.layers.Conv3D(
        64,
        (3, 3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(0.005),
    )(conv7)
    conv7 = tf.keras.layers.Dropout(0.2)(conv7)
    outputs = tf.keras.layers.Conv3D(
        1,
        1,
        activation="sigmoid",
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(0.005),
    )(conv7)

    # Define the model
    model = tf.keras.Model(inputs, outputs)
    return model
