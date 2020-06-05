#!/usr/bin/env python3

from pathlib import Path

import contractions
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import utils
from layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

CWD = Path.cwd().as_posix()


def split_info(data):
    def to_numpy(*args):
        return [np.array(lst) for lst in args]

    context = []
    question = []
    answer_text = []
    answer_start = []

    for q in data:
        context.append(contractions.fix(q["context"].numpy().decode("UTF-8")))
        question.append(contractions.fix(q["question"].numpy().decode("UTF-8")))
        answer_text.append(
            contractions.fix(q["answers"]["text"].numpy()[0].decode("UTF-8"))
        )
        answer_start.append(q["answers"]["answer_start"].numpy()[0])

    return to_numpy(context, question, answer_text, answer_start)


squad_data, info = tfds.load("squad", data_dir=CWD, with_info=True)
squad_train = squad_data["train"]
squad_validation = squad_data["validation"]
print(info.features)

context_tr, question_tr, answer_text_tr, answer_start_tr = split_info(squad_train)
context_val, question_val, answer_text_val, answer_start_val = split_info(
    squad_validation
)

padding_type = "post"
oov_token = "<OOV>"

tokenizer = Tokenizer(oov_token=oov_token)
tokenizer.fit_on_texts(context_tr)
word_index = tokenizer.word_index
num_words = len(word_index.keys())
print("{:30s}: {}".format("Total number of words", num_words))

sequences = tokenizer.texts_to_sequences(context_tr)
con_len = max(map(len, sequences))
print("Max length of a context vector: {}".format(con_len))
context_padded = pad_sequences(sequences, maxlen=con_len, padding=padding_type)

sequences = tokenizer.texts_to_sequences(question_tr)
que_len = max(map(len, sequences))
print("Max length of a question vector: {}".format(que_len))
question_padded = pad_sequences(sequences, maxlen=que_len, padding=padding_type)

answer_token = tokenizer.texts_to_sequences(answer_text_tr)

sequences_val = tokenizer.texts_to_sequences(context_val)
context_val_padded = pad_sequences(sequences, maxlen=con_len, padding=padding_type)
sequences_val = tokenizer.texts_to_sequences(question_val)
question_val_padded = pad_sequences(sequences, maxlen=que_len, padding=padding_type)
answer_token_val = tokenizer.texts_to_sequences(answer_text_val)

y_train_i = []  # (start, end)
selected_i = []
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
            y_train_i.append((start, end))
            selected_i.append(i)
            break

context_padded_clean = context_padded[selected_i]
question_padded_clean = question_padded[selected_i]
answer_text_clean = answer_text_tr[selected_i]

num_train_data = context_padded_clean.shape[0]
print(
    "Number of samples in training set after preprocessing: {}".format(num_train_data)
)

y_train = []

for i in range(len(context_padded_clean)):
    s = np.zeros(con_len, dtype="float32")
    e = np.zeros(con_len, dtype="float32")

    s[y_train_i[i][0]] = 1
    e[y_train_i[i][1]] = 1

    y_train.append(np.concatenate((s, e)))

embeddings = {}

with open("glove.6B.50d.txt") as fd:
    for line in fd:
        line = line.strip().split()
        embeddings[line[0]] = np.asarray(line[1:], dtype="float32")

embeddings_mat = np.zeros((num_words + 1, 50))
for word, i in word_index.items():
    emb = embeddings.get(word)
    if emb is not None:
        embeddings_mat[i] = emb

units = 128
emb_dim = 50

# Functional model
context_input = Input(shape=(con_len,))
context_emb = Embedding(num_words, embeddings_mat, emb_dim)(context_input)
context_lstm = BiLSTM(units)(context_emb)

question_input = Input(shape=(que_len,))
question_emb = Embedding(num_words, embeddings_mat, emb_dim)(question_input)
question_lstm = BiLSTM(units)(question_emb)

y_prob = BiLinear_Layer(2 * units, que_len)(context_lstm, question_lstm)

model = Model(inputs=[context_input, question_input], outputs=y_prob)
model.summary()

context_padded_ = np.array(context_padded_clean)
question_padded_ = np.array(question_padded_clean)
y_train_ = np.array(y_train)

train_dataset = tf.data.Dataset.from_tensor_slices(
    ({"input_1": context_padded_, "input_2": question_padded_}, y_train_)
)

BATCH_SIZE = 128

train_dataset = train_dataset.batch(BATCH_SIZE)

utils.con_len = con_len
model.compile(
    optimizer="adam", loss=utils.loss, metrics=[utils.Custom_Accuracy(num_train_data)]
)

filepath = "epochs_{epoch:03d}"
checkpoint = ModelCheckpoint(filepath, save_weights_only=True)
callbacks_list = [checkpoint]

model.load_weights("epochs:1000")

num_epochs = 1500
init_epoch = 1000

history = model.fit(
    train_dataset,
    initial_epoch=init_epoch,
    epochs=num_epochs,
    callbacks=callbacks_list,
)
