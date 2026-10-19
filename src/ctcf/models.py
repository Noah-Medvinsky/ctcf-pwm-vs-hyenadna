"""Baseline CNN from Tang et al. (2025), Task 2 (ChIP-seq).

Copied from `chip_cnn` in amberT15/LLM_eval, chip-clip-seq/representation_perf.py
(MIT License). Uses tf_keras so the Keras 2 behavior of their TensorFlow 2.12
setup is kept.

Reproduction note: the original builds an optional BatchNorm + 1x1 Conv1D
(512 filters) for embedding inputs, but its first conv layer is applied to
`input` rather than to that branch's output, so the branch never reaches the
model's output. It is omitted here; the resulting network is identical to the
one the authors trained, for both one-hot and embedding inputs.
"""

import tf_keras as keras


def chip_cnn(input_shape, output_shape=1):
    """input_shape is (length, channels): (200, 4) for one-hot, (L, 256) for HyenaDNA."""
    initializer = keras.initializers.HeUniform()
    inputs = keras.Input(shape=input_shape)

    # first conv layer
    nn = keras.layers.Conv1D(filters=64, kernel_size=7, padding="same",
                             kernel_initializer=initializer)(inputs)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation("relu")(nn)
    nn = keras.layers.MaxPooling1D(4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    # second conv layer
    nn = keras.layers.Conv1D(filters=96, kernel_size=5, padding="same",
                             kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation("relu")(nn)
    nn = keras.layers.MaxPooling1D(4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    # third conv layer
    nn = keras.layers.Conv1D(filters=128, kernel_size=5, padding="same",
                             kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation("relu")(nn)
    nn = keras.layers.MaxPooling1D(2)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    # dense layer
    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(256)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation("relu")(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    # output layer
    logits = keras.layers.Dense(output_shape)(nn)
    outputs = keras.layers.Activation("sigmoid")(logits)
    return keras.Model(inputs=inputs, outputs=outputs)
