import os
import sys

import matplotlib.pyplot as plt
import numpy
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, Dropout, LSTM

from model.attention_decoder import Attention


def task_add_two_numbers_after_delimiter(n: int, seq_length: int, delimiter: float = 0.0,
                                         index_1: int = None, index_2: int = None) -> (np.array, np.array):
    """
    Task: Add the two numbers that come right after the delimiter.
    x = [1, 2, 3, 0, 4, 5, 6, 0, 7, 8]. Result is y = 4 + 7 = 11.
    @param n: number of samples in (x, y).
    @param seq_length: length of the sequence of x.
    @param delimiter: value of the delimiter. Default is 0.0
    @param index_1: index of the number that comes after the first 0.
    @param index_2: index of the number that comes after the second 0.
    @return: returns two numpy.array x and y of shape (n, seq_length, 1) and (n, 1).
    """
    x = np.random.uniform(0, 1, (n, seq_length))
    y = np.zeros(shape=(n, 1))
    for i in range(len(x)):
        if index_1 is None and index_2 is None:
            a, b = np.random.choice(range(1, len(x[i])), size=2, replace=False)
        else:
            a, b = index_1, index_2
        y[i] = 0.5 * x[i, a:a + 1] + 0.5 * x[i, b:b + 1]
        x[i, a - 1:a] = delimiter
        x[i, b - 1:b] = delimiter
    x = np.expand_dims(x, axis=-1)
    return x, y


def main():
    numpy.random.seed(7)

    # data. definition of the problem.
    seq_length = 48
    x_train, y_train = task_add_two_numbers_after_delimiter(20_000, seq_length)
    x_val, y_val = task_add_two_numbers_after_delimiter(4_000, seq_length)

    # just arbitrary values. it's for visual purposes. easy to see than random values.
    test_index_1 = 4
    test_index_2 = 9
    x_test, _ = task_add_two_numbers_after_delimiter(10, seq_length, 0, test_index_1, test_index_2)
    # x_test_mask is just a mask that, if applied to x_test, would still contain the information to solve the problem.
    # we expect the attention map to look like this mask.
    x_test_mask = np.zeros_like(x_test[..., 0])
    x_test_mask[:, test_index_1:test_index_1 + 1] = 1
    x_test_mask[:, test_index_2:test_index_2 + 1] = 1

    model = Sequential([
        LSTM(100, input_shape=(seq_length, 1), return_sequences=True),
        Attention(name='attention_weight'),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])

    model.compile(loss='mse', optimizer='adam')
    print(model.summary())

    output_dir = 'task_add_two_numbers'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    max_epoch = int(sys.argv[1]) if len(sys.argv) > 1 else 200

    class VisualiseAttentionMap(Callback):

        def on_epoch_end(self, epoch, logs=None):
            attention_map = get_activations(model, x_test, layer_names='attention_weight')['attention_weight']

            # top is attention map.
            # bottom is ground truth.
            plt.imshow(np.concatenate([attention_map, x_test_mask]), cmap='hot')

            iteration_no = str(epoch).zfill(3)
            plt.axis('off')
            plt.title(f'Iteration {iteration_no} / {max_epoch}')
            plt.savefig(f'{output_dir}/epoch_{iteration_no}.png')
            plt.close()
            plt.clf()

    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=max_epoch,
              batch_size=64, callbacks=[VisualiseAttentionMap()])


if __name__ == '__main__':
    main()