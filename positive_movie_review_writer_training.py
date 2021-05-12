import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

POSITIVE_VAL = 1

BATCH_SIZE = 64
EMBEDDING_DIM = 256
RNN_UNITS = 1024

""" Buffer size to shuffle dataset.
(TensorFlow data is designed to work with potentially infinite sequences.
So it does not try to shuffle entire sequence in memory.
Rather, it maintains buffer in which it shuffles elements.) """
BUFFER_SIZE = 10000

training_data = tfds.load(name="imdb_reviews", split="train", batch_size=-1, as_supervised=True)

training_data, training_labels = tfds.as_numpy(training_data)
training_df = pd.DataFrame({'review-text': training_data, 'sentiment-value': training_labels})
training_data_by_positive_val = training_df[training_df['sentiment-value'] == POSITIVE_VAL]
review_text_by_positive_val = training_data_by_positive_val['review-text']

review_text_by_sentiment_str = ''

for i in range(len(review_text_by_positive_val)):
    # Access elements of Series converted to array to avoid error message.
    review_text_by_sentiment_str += review_text_by_positive_val.array[i].decode(encoding='utf-8')

# Call ascii to prevent non-ascii characters from being included in output.
review_text_by_sentiment_str = ascii(review_text_by_sentiment_str)

vocab = sorted(set(review_text_by_sentiment_str))
char_to_idx = {u: i for i, u in enumerate(vocab)}
vocab_arr = np.array(vocab)


def text_to_int(text):
    return np.array([char_to_idx[c] for c in text])


text_as_int = text_to_int(review_text_by_sentiment_str)

# Creating training examples.

seq_len = 100  # Length of sequence for training example.
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_len + 1, drop_remainder=True)


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset = sequences.map(split_input_target)

VOCAB_SIZE = len(vocab)

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

model.fit(data, epochs=40, callbacks=[checkpoint_callback])
model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))


def generate_text(model_arg, start_str):
    num_chars_to_generate = 800

    input_eval = [char_to_idx[s] for s in start_str]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    # Low temperatures result in more predictable text and vice-versa.
    temperature = 1.0
    # Batch size = 1 here.
    model_arg.reset_states()

    for iteration in range(num_chars_to_generate):
        predictions = model_arg(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature

        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        """ Pass predicted character as next input to model
        along with previous hidden state. """
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(vocab_arr[predicted_id])

    return start_str + ''.join(text_generated)


inp = input("Please type a starting string: ")
print(generate_text(model, inp))
