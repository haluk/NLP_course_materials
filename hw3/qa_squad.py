#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

import arch_utils
import utils
from layers import *

CWD = Path.cwd().as_posix()

squad_data, info = tfds.load("squad", data_dir=CWD, with_info=True)
squad_train = squad_data["train"]
squad_validation = squad_data["validation"]
print(info.features)

context_tr, question_tr, answer_text_tr, answer_start_tr = utils.split_info(squad_train)

padding_type = "post"
oov_token = "<OOV>"

tokenizer = Tokenizer(oov_token=oov_token)
tokenizer.fit_on_texts(context_tr)
word_index = tokenizer.word_index
num_words = len(word_index.keys())
print("{:50s}: {}".format("Total number of words", num_words))

sequences = tokenizer.texts_to_sequences(context_tr)
context_len = max(map(len, sequences))
print("{:50s}: {}".format("Max length of a context vector", context_len))
context_padded = pad_sequences(sequences, maxlen=context_len, padding=padding_type)

sequences = tokenizer.texts_to_sequences(question_tr)
question_len = max(map(len, sequences))
print("{:50s}: {}".format("Max length of a question vector", question_len))
question_padded = pad_sequences(sequences, maxlen=question_len, padding=padding_type)

answer_token = tokenizer.texts_to_sequences(answer_text_tr)

y_train_tup = []  # (start,end)
selected = []  # contexts which contain exact answer
WINDOW = 10

for i in range(len(answer_start_tr)):
    start_char_idx = answer_start_tr[i]
    start = len(context_tr[i][0 : start_char_idx - 1].replace("-", " ").split()) + 1
    answer_len = len(answer_text_tr[i].replace("-", " ").split())
    end = start + answer_len

    for j in range(start - WINDOW, start + WINDOW):
        if np.array_equal(context_padded[i][j : j + answer_len], answer_token[i]):
            start = j
            end = j + answer_len - 1
            y_train_tup.append((start, end))
            selected.append(i)
            break

context_padded_clean = context_padded[selected]
question_padded_clean = question_padded[selected]
answer_text_clean = answer_text_tr[selected]

num_train_data = context_padded_clean.shape[0]
print("{:50s}: {}".format("Number of training samples after cleaning", num_train_data))

y_train = []
for i in range(len(context_padded_clean)):
    s = np.zeros(context_len, dtype="float32")
    e = np.zeros(context_len, dtype="float32")

    s[y_train_tup[i][0]] = 1
    e[y_train_tup[i][1]] = 1

    y_train.append(np.concatenate((s, e)))
y_train = np.array(y_train)

emb_dim = 50  # glove.6B.50d
embeddings_mat = arch_utils.load_embeddings(
    "glove.6B.50d.txt", num_words, emb_dim, word_index
)

# Functional model
units = 128
context_input = Input(shape=(context_len,))
context_emb = Embedding(num_words, embeddings_mat, emb_dim)(context_input)
context_lstm = BiLSTM(units)(context_emb)

question_input = Input(shape=(question_len,))
question_emb = Embedding(num_words, embeddings_mat, emb_dim)(question_input)
question_lstm = BiLSTM(units)(question_emb)

y_prob = BiLinear_Layer(2 * units, question_len)(context_lstm, question_lstm)

model = Model(inputs=[context_input, question_input], outputs=y_prob)
model.summary()

optimizer = tf.keras.optimizers.Adam(lr=1e-3)
model.compile(
    optimizer=optimizer,
    loss=arch_utils.loss,
    metrics=[arch_utils.Custom_Accuracy(num_train_data)],
)

context_padded_jupyter = context_padded_clean[:1000]
question_padded_jupyter = question_padded_clean[:1000]
y_train_jupyter = y_train[:1000]

# model.load_weights("epochs_1000")
init_epoch = 0
num_epochs = 2
batch_size = 128

# early stopping will depend on the validation loss
# patience parameter determines how many epochs with no improvement
# in validation loss will be tolerated
# before training is terminated.
earlystopping = EarlyStopping(monitor="val_loss", patience=2)

filepath = "epochs_{epoch:03d}"
checkpoint = ModelCheckpoint(filepath, save_weights_only=True)

callbacks = [earlystopping, checkpoint]

history = model.fit(
    x=[context_padded_jupyter, question_padded_jupyter],
    y=y_train_jupyter,
    # keep 10% of the training data for validation
    validation_split=0.1,
    initial_epoch=init_epoch,
    epochs=num_epochs,
    callbacks=callbacks,
    verbose=2,  # Logs once per epoch.
    batch_size=batch_size,
    # Our neural network will be trained
    # with stochastic (mini-batch) gradient descent.
    # It is important that we shuffle our input.
    shuffle=True,  # set to True by default
)

# Print training history
history = history.history
print(
    "\nValidation accuracy: {acc}, loss: {loss}".format(
        acc=history["val_custom_accuracy"][-1], loss=history["val_loss"][-1]
    )
)

# Testing
context_padded_test_jupyter = context_padded_clean[1000:2000]
question_padded_test_jupyter = question_padded_clean[1000:2000]
y_test_jupyter = y_train[1000:2000]

print("\nTesting...")
model.evaluate(
    [context_padded_test_jupyter, question_padded_test_jupyter],
    y_test_jupyter,
    batch_size=batch_size,
    verbose=1,
)
