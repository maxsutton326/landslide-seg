import tensorflow as tf


def get_unet_5(input_shape, max_filters=512):
    inputs = tf.keras.Input(shape=input_shape)
    # Encoder: Contracting Path
    # Block 1

    # Block 2
    conv2 = tf.keras.layers.Conv3D(
        max_filters / 4, (3, 3, 3), activation="relu", padding="same"
    )(inputs)
    conv2 = tf.keras.layers.Conv3D(
        max_filters / 4, (3, 3, 3), activation="relu", padding="same"
    )(conv2)
    pool2 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    # pool2 = tf.keras.layers.Dropout(0.8)(pool2)

    # Block 3
    conv3 = tf.keras.layers.Conv3D(
        max_filters / 2, (3, 3, 3), activation="relu", padding="same"
    )(pool2)
    conv3 = tf.keras.layers.Conv3D(
        max_filters / 2, (3, 3, 3), activation="relu", padding="same"
    )(conv3)
    pool3 = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(conv3)
    # pool3 = tf.keras.layers.Dropout(0.8)(pool3)

    # Block 4 (Bottleneck)
    conv4 = tf.keras.layers.Conv3D(
        max_filters, (3, 3, 3), activation="relu", padding="same"
    )(pool3)
    conv4 = tf.keras.layers.Conv3D(
        max_filters, (3, 3, 3), activation="relu", padding="same"
    )(conv4)
    # conv4 = tf.keras.layers.Dropout(0.8)(conv4)

    # Decoder: Expanding Path
    # Block 5
    up5 = tf.keras.layers.Conv3DTranspose(
        max_filters / 2, (3, 3, 3), strides=(1, 2, 2), activation="relu", padding="same"
    )(conv4)
    up5 = tf.keras.layers.Conv3D(
        max_filters / 2, (3, 3, 3), activation="relu", padding="same"
    )(up5)
    merge5 = tf.keras.layers.concatenate([conv3, up5], axis=-1)
    conv5 = tf.keras.layers.Conv3D(
        max_filters / 2, (3, 3, 3), activation="relu", padding="same"
    )(merge5)
    conv5 = tf.keras.layers.Conv3D(
        max_filters / 2,
        (3, 3, 3),
        activation="relu",
        padding="same",
        #
    )(conv5)

    # Block 6
    up6 = tf.keras.layers.Conv3DTranspose(
        max_filters / 4, (3, 3, 3), strides=(2, 2, 2), activation="relu", padding="same"
    )(conv5)
    up6 = tf.keras.layers.Conv3D(
        max_filters / 4,
        (3, 3, 3),
        activation="relu",
        padding="same",
        #
    )(up6)
    merge6 = tf.keras.layers.concatenate([conv2, up6], axis=-1)
    conv6 = tf.keras.layers.Conv3D(
        max_filters / 4,
        (3, 3, 3),
        activation="relu",
        padding="same",
        #
    )(merge6)
    conv6 = tf.keras.layers.Conv3D(
        max_filters / 4, (3, 3, 3), activation="relu", padding="same"
    )(conv6)

    outputs = tf.keras.layers.Conv3D(1, 1, activation="sigmoid", padding="same")(conv6)

    # Define the model
    model = tf.keras.Model(inputs, outputs)
    return model


def get_unet_7(input_shape, max_filters=512):
    inputs = tf.keras.Input(shape=input_shape)
    # Encoder: Contracting Path
    # Block 1
    conv1 = tf.keras.layers.Conv3D(
        max_filters / 8, (3, 3, 3), activation="relu", padding="same"
    )(inputs)
    conv1 = tf.keras.layers.Conv3D(
        max_filters / 8, (3, 3, 3), activation="relu", padding="same"
    )(conv1)
    pool1 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    pool1 = tf.keras.layers.Dropout(0.8)(pool1)

    # Block 2
    conv2 = tf.keras.layers.Conv3D(
        max_filters / 4, (3, 3, 3), activation="relu", padding="same"
    )(pool1)
    conv2 = tf.keras.layers.Conv3D(
        max_filters / 4, (3, 3, 3), activation="relu", padding="same"
    )(conv2)
    pool2 = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(conv2)
    pool2 = tf.keras.layers.Dropout(0.8)(pool2)

    # Block 3
    conv3 = tf.keras.layers.Conv3D(
        max_filters / 2, (3, 3, 3), activation="relu", padding="same"
    )(pool2)
    conv3 = tf.keras.layers.Conv3D(
        max_filters / 2, (3, 3, 3), activation="relu", padding="same"
    )(conv3)
    pool3 = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(conv3)
    pool3 = tf.keras.layers.Dropout(0.8)(pool3)

    # Block 4 (Bottleneck)
    conv4 = tf.keras.layers.Conv3D(
        max_filters, (3, 3, 3), activation="relu", padding="same"
    )(pool3)
    conv4 = tf.keras.layers.Conv3D(
        max_filters, (3, 3, 3), activation="relu", padding="same"
    )(conv4)
    conv4 = tf.keras.layers.Dropout(0.8)(conv4)

    # Decoder: Expanding Path
    # Block 5
    up5 = tf.keras.layers.Conv3DTranspose(
        max_filters / 2, (3, 3, 3), strides=(1, 2, 2), activation="relu", padding="same"
    )(conv4)
    up5 = tf.keras.layers.Conv3D(
        max_filters / 2, (3, 3, 3), activation="relu", padding="same"
    )(up5)
    merge5 = tf.keras.layers.concatenate([conv3, up5], axis=-1)
    conv5 = tf.keras.layers.Conv3D(
        max_filters / 2, (3, 3, 3), activation="relu", padding="same"
    )(merge5)
    conv5 = tf.keras.layers.Conv3D(
        max_filters / 2, (3, 3, 3), activation="relu", padding="same"
    )(conv5)
    conv5 = tf.keras.layers.Dropout(0.8)(conv5)

    # Block 6
    up6 = tf.keras.layers.Conv3DTranspose(
        max_filters / 4, (3, 3, 3), strides=(1, 2, 2), activation="relu", padding="same"
    )(conv5)
    up6 = tf.keras.layers.Conv3D(
        max_filters / 4, (3, 3, 3), activation="relu", padding="same"
    )(up6)
    merge6 = tf.keras.layers.concatenate([conv2, up6], axis=-1)
    conv6 = tf.keras.layers.Conv3D(
        max_filters / 4, (3, 3, 3), activation="relu", padding="same"
    )(merge6)
    conv6 = tf.keras.layers.Conv3D(
        max_filters / 4, (3, 3, 3), activation="relu", padding="same"
    )(conv6)
    conv6 = tf.keras.layers.Dropout(0.8)(conv6)

    # Block 7
    up7 = tf.keras.layers.Conv3DTranspose(
        max_filters / 8, (3, 3, 3), strides=(2, 2, 2), activation="relu", padding="same"
    )(conv6)
    up7 = tf.keras.layers.Conv3D(
        max_filters / 8, (3, 3, 3), activation="relu", padding="same"
    )(up7)
    merge7 = tf.keras.layers.concatenate([conv1, up7], axis=-1)
    conv7 = tf.keras.layers.Conv3D(
        max_filters / 8, (3, 3, 3), activation="relu", padding="same"
    )(merge7)
    conv7 = tf.keras.layers.Conv3D(
        max_filters / 8, (3, 3, 3), activation="relu", padding="same"
    )(conv7)
    conv7 = tf.keras.layers.Dropout(0.8)(conv7)

    outputs = tf.keras.layers.Conv3D(1, 1, activation="sigmoid", padding="same")(conv7)

    # Define the model
    model = tf.keras.Model(inputs, outputs)
    return model


def get_unet_9(input_shape, max_filters=512):
    inputs = tf.keras.Input(shape=input_shape)
    # Encoder: Contracting Path
    conv0 = tf.keras.layers.Conv3D(
        max_filters / 16, (3, 3, 3), activation="relu", padding="same"
    )(inputs)
    conv0 = tf.keras.layers.Conv3D(
        max_filters / 16, (3, 3, 3), activation="relu", padding="same"
    )(conv0)
    pool0 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv0)
    pool0 = tf.keras.layers.Dropout(0.8)(pool0)
    # Block 1
    conv1 = tf.keras.layers.Conv3D(
        max_filters / 8, (3, 3, 3), activation="relu", padding="same"
    )(pool0)
    conv1 = tf.keras.layers.Conv3D(
        max_filters / 8, (3, 3, 3), activation="relu", padding="same"
    )(conv1)
    pool1 = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(conv1)
    pool1 = tf.keras.layers.Dropout(0.8)(pool1)

    # Block 2
    conv2 = tf.keras.layers.Conv3D(
        max_filters / 4, (3, 3, 3), activation="relu", padding="same"
    )(pool1)
    conv2 = tf.keras.layers.Conv3D(
        max_filters / 4, (3, 3, 3), activation="relu", padding="same"
    )(conv2)
    pool2 = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(conv2)
    # pool2 = tf.keras.layers.Dropout(0.8)(pool2)

    # Block 3
    conv3 = tf.keras.layers.Conv3D(
        max_filters / 2, (3, 3, 3), activation="relu", padding="same"
    )(pool2)
    conv3 = tf.keras.layers.Conv3D(
        max_filters / 2, (3, 3, 3), activation="relu", padding="same"
    )(conv3)
    pool3 = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(conv3)
    # pool3 = tf.keras.layers.Dropout(0.8)(pool3)

    # Block 4 (Bottleneck)
    conv4 = tf.keras.layers.Conv3D(
        max_filters, (3, 3, 3), activation="relu", padding="same"
    )(pool3)
    conv4 = tf.keras.layers.Conv3D(
        max_filters, (3, 3, 3), activation="relu", padding="same"
    )(conv4)
    # conv4 = tf.keras.layers.Dropout(0.8)(conv4)

    # Decoder: Expanding Path
    # Block 5
    up5 = tf.keras.layers.Conv3DTranspose(
        max_filters / 2, (3, 3, 3), strides=(1, 2, 2), activation="relu", padding="same"
    )(conv4)
    up5 = tf.keras.layers.Conv3D(
        max_filters / 2, (3, 3, 3), activation="relu", padding="same"
    )(up5)
    merge5 = tf.keras.layers.concatenate([conv3, up5], axis=-1)
    conv5 = tf.keras.layers.Conv3D(
        max_filters / 2, (3, 3, 3), activation="relu", padding="same"
    )(merge5)
    conv5 = tf.keras.layers.Conv3D(
        max_filters / 2,
        (3, 3, 3),
        activation="relu",
        padding="same",
        #
    )(conv5)

    # Block 6
    up6 = tf.keras.layers.Conv3DTranspose(
        max_filters / 4, (3, 3, 3), strides=(1, 2, 2), activation="relu", padding="same"
    )(conv5)
    up6 = tf.keras.layers.Conv3D(
        max_filters / 4,
        (3, 3, 3),
        activation="relu",
        padding="same",
        #
    )(up6)
    merge6 = tf.keras.layers.concatenate([conv2, up6], axis=-1)
    conv6 = tf.keras.layers.Conv3D(
        max_filters / 4,
        (3, 3, 3),
        activation="relu",
        padding="same",
        #
    )(merge6)
    conv6 = tf.keras.layers.Conv3D(
        max_filters / 4, (3, 3, 3), activation="relu", padding="same"
    )(conv6)

    # Block 7
    up7 = tf.keras.layers.Conv3DTranspose(
        max_filters / 8, (3, 3, 3), strides=(1, 2, 2), activation="relu", padding="same"
    )(conv6)
    up7 = tf.keras.layers.Conv3D(
        max_filters / 8, (3, 3, 3), activation="relu", padding="same"
    )(up7)
    merge7 = tf.keras.layers.concatenate([conv1, up7], axis=-1)
    conv7 = tf.keras.layers.Conv3D(
        max_filters / 8, (3, 3, 3), activation="relu", padding="same"
    )(merge7)
    conv7 = tf.keras.layers.Conv3D(
        max_filters / 8, (3, 3, 3), activation="relu", padding="same"
    )(conv7)

    up8 = tf.keras.layers.Conv3DTranspose(
        max_filters / 16,
        (3, 3, 3),
        strides=(2, 2, 2),
        activation="relu",
        padding="same",
    )(conv7)
    up8 = tf.keras.layers.Conv3D(
        max_filters / 16, (3, 3, 3), activation="relu", padding="same"
    )(up8)
    merge8 = tf.keras.layers.concatenate([conv0, up8], axis=-1)
    conv8 = tf.keras.layers.Conv3D(
        max_filters / 16, (3, 3, 3), activation="relu", padding="same"
    )(merge8)
    conv8 = tf.keras.layers.Conv3D(
        max_filters / 16, (3, 3, 3), activation="relu", padding="same"
    )(conv8)
    outputs = tf.keras.layers.Conv3D(1, 1, activation="sigmoid", padding="same")(conv8)

    # Define the model
    model = tf.keras.Model(inputs, outputs)
    return model


def get_unet_11(input_shape, max_filters=512):
    inputs = tf.keras.Input(shape=input_shape)
    # Encoder: Contracting Path
    conv00 = tf.keras.layers.Conv3D(
        max_filters / 32, (3, 3, 3), activation="relu", padding="same"
    )(inputs)
    conv00 = tf.keras.layers.Conv3D(
        max_filters / 32, (3, 3, 3), activation="relu", padding="same"
    )(conv00)
    pool00 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv00)
    pool00 = tf.keras.layers.Dropout(0.8)(pool00)

    conv0 = tf.keras.layers.Conv3D(
        max_filters / 16, (3, 3, 3), activation="relu", padding="same"
    )(pool00)
    conv0 = tf.keras.layers.Conv3D(
        max_filters / 16, (3, 3, 3), activation="relu", padding="same"
    )(conv0)
    pool0 = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(conv0)
    pool0 = tf.keras.layers.Dropout(0.8)(pool0)
    # Block 1
    conv1 = tf.keras.layers.Conv3D(
        max_filters / 8, (3, 3, 3), activation="relu", padding="same"
    )(pool0)
    conv1 = tf.keras.layers.Conv3D(
        max_filters / 8, (3, 3, 3), activation="relu", padding="same"
    )(conv1)
    pool1 = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(conv1)
    # pool1 = tf.keras.layers.Dropout(0.8)(pool1)

    # Block 2
    conv2 = tf.keras.layers.Conv3D(
        max_filters / 4, (3, 3, 3), activation="relu", padding="same"
    )(pool1)
    conv2 = tf.keras.layers.Conv3D(
        max_filters / 4, (3, 3, 3), activation="relu", padding="same"
    )(conv2)
    pool2 = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(conv2)
    # pool2 = tf.keras.layers.Dropout(0.8)(pool2)

    # Block 3
    conv3 = tf.keras.layers.Conv3D(
        max_filters / 2, (3, 3, 3), activation="relu", padding="same"
    )(pool2)
    conv3 = tf.keras.layers.Conv3D(
        max_filters / 2, (3, 3, 3), activation="relu", padding="same"
    )(conv3)
    pool3 = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(conv3)
    # pool3 = tf.keras.layers.Dropout(0.8)(pool3)

    # Block 4 (Bottleneck)
    conv4 = tf.keras.layers.Conv3D(
        max_filters, (3, 3, 3), activation="relu", padding="same"
    )(pool3)
    conv4 = tf.keras.layers.Conv3D(
        max_filters, (3, 3, 3), activation="relu", padding="same"
    )(conv4)
    # conv4 = tf.keras.layers.Dropout(0.8)(conv4)

    # Decoder: Expanding Path
    # Block 5
    up5 = tf.keras.layers.Conv3DTranspose(
        max_filters / 2, (3, 3, 3), strides=(1, 2, 2), activation="relu", padding="same"
    )(conv4)
    up5 = tf.keras.layers.Conv3D(
        max_filters / 2, (3, 3, 3), activation="relu", padding="same"
    )(up5)
    merge5 = tf.keras.layers.concatenate([conv3, up5], axis=-1)
    conv5 = tf.keras.layers.Conv3D(
        max_filters / 2, (3, 3, 3), activation="relu", padding="same"
    )(merge5)
    conv5 = tf.keras.layers.Conv3D(
        max_filters / 2,
        (3, 3, 3),
        activation="relu",
        padding="same",
        #
    )(conv5)

    # Block 6
    up6 = tf.keras.layers.Conv3DTranspose(
        max_filters / 4, (3, 3, 3), strides=(1, 2, 2), activation="relu", padding="same"
    )(conv5)
    up6 = tf.keras.layers.Conv3D(
        max_filters / 4,
        (3, 3, 3),
        activation="relu",
        padding="same",
        #
    )(up6)
    merge6 = tf.keras.layers.concatenate([conv2, up6], axis=-1)
    conv6 = tf.keras.layers.Conv3D(
        max_filters / 4,
        (3, 3, 3),
        activation="relu",
        padding="same",
        #
    )(merge6)
    conv6 = tf.keras.layers.Conv3D(
        max_filters / 4, (3, 3, 3), activation="relu", padding="same"
    )(conv6)

    # Block 7
    up7 = tf.keras.layers.Conv3DTranspose(
        max_filters / 8, (3, 3, 3), strides=(1, 2, 2), activation="relu", padding="same"
    )(conv6)
    up7 = tf.keras.layers.Conv3D(
        max_filters / 8, (3, 3, 3), activation="relu", padding="same"
    )(up7)
    merge7 = tf.keras.layers.concatenate([conv1, up7], axis=-1)
    conv7 = tf.keras.layers.Conv3D(
        max_filters / 8, (3, 3, 3), activation="relu", padding="same"
    )(merge7)
    conv7 = tf.keras.layers.Conv3D(
        max_filters / 8, (3, 3, 3), activation="relu", padding="same"
    )(conv7)

    up8 = tf.keras.layers.Conv3DTranspose(
        max_filters / 16,
        (3, 3, 3),
        strides=(1, 2, 2),
        activation="relu",
        padding="same",
    )(conv7)
    up8 = tf.keras.layers.Conv3D(
        max_filters / 16, (3, 3, 3), activation="relu", padding="same"
    )(up8)
    merge8 = tf.keras.layers.concatenate([conv0, up8], axis=-1)
    conv8 = tf.keras.layers.Conv3D(
        max_filters / 16, (3, 3, 3), activation="relu", padding="same"
    )(merge8)
    conv8 = tf.keras.layers.Conv3D(
        max_filters / 16, (3, 3, 3), activation="relu", padding="same"
    )(conv8)

    up9 = tf.keras.layers.Conv3DTranspose(
        max_filters / 32,
        (3, 3, 3),
        strides=(2, 2, 2),
        activation="relu",
        padding="same",
    )(conv8)
    up9 = tf.keras.layers.Conv3D(
        max_filters / 32, (3, 3, 3), activation="relu", padding="same"
    )(up9)
    merge9 = tf.keras.layers.concatenate([conv00, up9], axis=-1)
    conv9 = tf.keras.layers.Conv3D(
        max_filters / 32, (3, 3, 3), activation="relu", padding="same"
    )(merge9)
    conv9 = tf.keras.layers.Conv3D(
        max_filters / 32, (3, 3, 3), activation="relu", padding="same"
    )(conv9)

    outputs = tf.keras.layers.Conv3D(1, 1, activation="sigmoid", padding="same")(conv9)
    # Define the model
    model = tf.keras.Model(inputs, outputs)
    return model
