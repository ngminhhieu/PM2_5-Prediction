from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, Concatenate
from tensorflow.keras.models import Model
from model.attention import AttentionLayer
# from tensorflow.keras.utils import plot_model

def cnn_lstm_attention_construction(seq_len, input_dim, output_dim, rnn_units, dropout, optimizer, log_dir, is_training=True):
    encoder_inputs = Input(shape=(seq_len, input_dim), name='encoder_input')
    conv1d_layer = Conv1D(filters=64, kernel_size=3, activation='relu')(encoder_inputs)
    encoder = LSTM(rnn_units, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder(conv1d_layer)

    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, output_dim), name='decoder_input')
    decoder_lstm = LSTM(rnn_units, return_sequences=True, return_state=True)
    decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    # attention
    attn_layer = AttentionLayer(input_shape=([None, seq_len, rnn_units],
                                                [None, seq_len, rnn_units]),
                                name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])
    decoder_outputs = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])
    
    # output
    decoder_dense = Dense(output_dim, activation='relu')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # plot_model(model=model, to_file=log_dir + '/model.png', show_shapes=True)

    if is_training:
        return model
    else:
        print("Load model from: {}".format(log_dir))
        model.load_weights(log_dir + 'best_model.hdf5')
        model.compile(optimizer=optimizer, loss='mse')

        # Inference encoder_model
        encoder_model = Model(encoder_inputs, [encoder_outputs] + encoder_states)

        # Inference decoder_model
        decoder_state_input_h = Input(shape=(rnn_units,))
        decoder_state_input_c = Input(shape=(rnn_units,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]

        # attention
        encoder_inf_states = Input(shape=(seq_len, rnn_units),
                                    name='encoder_inf_states_input')
        attn_out, attn_states = attn_layer([encoder_inf_states, decoder_outputs])
        decoder_outputs = Concatenate(axis=-1, name='concat')([decoder_outputs, attn_out])

        # output
        decoder_outputs = decoder_dense(decoder_outputs)

        decoder_model = Model([encoder_inf_states, decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

        # plot_model(model=encoder_model, to_file=log_dir + '/encoder.png', show_shapes=True)
        # plot_model(model=decoder_model, to_file=log_dir + '/decoder.png', show_shapes=True)

        return model, encoder_model, decoder_model
