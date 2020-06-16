#!/usr/bin/env python3

import os
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf
from IPython.display import Video
from sklearn.model_selection import train_test_split

import utils
from model import *

path_to_zip = tf.keras.utils.get_file(
    "tur-eng.zip",
    origin="https://github.com/haluk/NLP_course_materials/blob/master/hw4/tur-eng.zip?raw=true",
    extract=True,
)
path_to_file = os.path.dirname(path_to_zip) + "/tur.txt"
en, tr, _ = utils.create_dataset(path_to_file, None)

# Try experimenting with the size of that dataset
num_examples = 30000
input_tensor, target_tensor, inp_lang, targ_lang = utils.load_dataset(
    path_to_file, num_examples
)

# Calculate max_length of the target tensors
max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

# Creating training and validation sets using an 80-20 split
(
    input_tensor_train,
    input_tensor_val,
    target_tensor_train,
    target_tensor_val,
) = train_test_split(input_tensor, target_tensor, test_size=0.2)

# Show length
print(
    len(input_tensor_train),
    len(target_tensor_train),
    len(input_tensor_val),
    len(target_tensor_val),
)

print(
    "{:20s} => {:20s}: {} {:20s}: {}".format(
        "Input language",
        "Training size",
        len(input_tensor_train),
        "Validation size",
        len(input_tensor_val),
    )
)
print(
    "{:20s} => {:20s}: {} {:20s}: {}".format(
        "Target language",
        "Training size",
        len(target_tensor_train),
        "Validation size",
        len(target_tensor_val),
    )
)


# print("Input Language; index to word mapping")
# utils.convert(inp_lang, input_tensor_train[0])
# print()
# print("Target Language; index to word mapping")
# utils.convert(targ_lang, target_tensor_train[0])

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word_index) + 1
vocab_tar_size = len(targ_lang.word_index) + 1

dataset = tf.data.Dataset.from_tensor_slices(
    (input_tensor_train, target_tensor_train)
).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


checkpoint_dir = "./training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
attention_layer = BahdanauAttention(10)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)


@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([targ_lang.word_index["<start>"]] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = loss / int(targ.shape[1])

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


EPOCHS = 10

for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print(
                "Epoch {} Batch {} Loss {:.4f}".format(
                    epoch + 1, batch, batch_loss.numpy()
                )
            )
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

    print("Epoch {} Loss {:.4f}".format(epoch + 1, total_loss / steps_per_epoch))
    print("Time taken for 1 epoch {} sec\n".format(time.time() - start))
