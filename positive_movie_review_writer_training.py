import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

POSITIVE_VAL = 1

training_data = tfds.load(name="imdb_reviews", split="train", batch_size=-1, as_supervised=True)

training_data, training_labels = tfds.as_numpy(training_data)

training_df = pd.DataFrame({'review-text': training_data, 'sentiment-value': training_labels})

training_data_by_sentiment_val = training_df[training_df['sentiment-value'] == POSITIVE_VAL]
review_text_by_sentiment_val = training_data_by_sentiment_val['review-text']

review_text_by_sentiment_str = ''
review_text_by_sentiment_len = len(review_text_by_sentiment_val)

# Access elements of Series converted to array to avoid error message.
review_text_by_sentiment_arr = review_text_by_sentiment_val.values

for i in range(review_text_by_sentiment_len):
    review_text_by_sentiment_element_str = review_text_by_sentiment_arr[i].decode()

    # Literal line break tags have to be removed from string.
    review_text_by_sentiment_str += review_text_by_sentiment_element_str.replace('<br /><br />', '\n')

    if i < (review_text_by_sentiment_len - 1):
        review_text_by_sentiment_str += '\n\n'

vocab = sorted(set(review_text_by_sentiment_str))
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)


def text_to_int(text):
    return np.array([char2idx[c] for c in text])


text_as_int = text_to_int(review_text_by_sentiment_str)

# Creating training examples.

seq_length = 100  # Length of sequence for training example.
examples_per_epoch = len(review_text_by_sentiment_str)//(seq_length + 1)
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset = sequences.map(split_input_target)

BATCH_SIZE = 64
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 256
RNN_UNITS = 1024

BUFFER_SIZE = 10000

data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    temp_model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return temp_model


model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
model.summary()


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


model.compile(optimizer='adam', loss=loss)

# Creating checkpoints.

checkpoint_dir = './training_checkpoints'

# Name of checkpoint files.
checkpoint_prefix = os.path.join(checkpoint_dir, 'checkpoint_{epoch}')

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)

history = model.fit(data, epochs=40, callbacks=[checkpoint_callback])

model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))


def generate_text(model_arg, start_string):
    num_generate = 800

    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    # Low temperatures result in more predictable text and vice-versa.
    temperature = 1.0
    # Batch size = 1 here.
    model_arg.reset_states()

    for iteration in range(num_generate):
        predictions = model_arg(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        """ Pass predicted character as next input to model
        along with previous hidden state. """
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)


inp = input("Type a starting string: ")
print(generate_text(model, inp))
