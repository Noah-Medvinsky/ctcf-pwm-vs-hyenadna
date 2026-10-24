"""Check that ctcf.models.chip_cnn matches the authors' original network."""

import numpy as np
import pytest
import tf_keras as keras

from ctcf.models import chip_cnn


def original_chip_cnn(input_shape, output_shape, model_name):
    """Verbatim copy of chip_cnn from LLM_eval chip-clip-seq/representation_perf.py
    (MIT License, Copyright (c) 2024 amberT15; see THIRD_PARTY_NOTICES.md)."""
    initializer = keras.initializers.HeUniform()
    input = keras.Input(shape=input_shape)
    if model_name != 'GPN' and model_name != 'CNN':
        nn = keras.layers.BatchNormalization()(input)
    else:
        nn = input
    nn = keras.layers.Conv1D(filters=512, kernel_size=1, kernel_initializer=initializer)(nn)
    nn = keras.layers.Conv1D(filters=64, kernel_size=7, padding='same', kernel_initializer=initializer)(input)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPooling1D(4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)
    nn = keras.layers.Conv1D(filters=96, kernel_size=5, padding='same', kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPooling1D(4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)
    nn = keras.layers.Conv1D(filters=128, kernel_size=5, padding='same', kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPooling1D(2)(nn)
    nn = keras.layers.Dropout(0.2)(nn)
    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(256)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.5)(nn)
    logits = keras.layers.Dense(output_shape)(nn)
    output = keras.layers.Activation('sigmoid')(logits)
    return keras.Model(inputs=input, outputs=output)


@pytest.mark.parametrize("input_shape,model_name", [((200, 4), "CNN"), ((201, 256), "Hyena")])
def test_same_architecture_as_original(input_shape, model_name):
    ours = chip_cnn(input_shape, 1)
    theirs = original_chip_cnn(input_shape, 1, model_name)
    assert [w.shape for w in ours.weights] == [w.shape for w in theirs.weights]
    assert ours.count_params() == theirs.count_params()


def test_output_is_probability():
    model = chip_cnn((200, 4), 1)
    x = np.random.default_rng(0).random((8, 200, 4)).astype("float32")
    prob = model.predict(x, verbose=0)
    assert prob.shape == (8, 1)
    assert ((prob >= 0) & (prob <= 1)).all()
