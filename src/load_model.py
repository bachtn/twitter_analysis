from keras.models import Model
from keras.layers import Embedding, LSTM, concatenate, RepeatVector, Input, Dense

def get_model(path):
    encoding_dim = 50
    input_size = 50
    embedding_dim = 200
    voc_size = 19635

    input_seq = Input(shape=(50,1))
    input_type = Input(shape=(4,))


    embedding_layer = Embedding(voc_size,
                            embedding_dim,
                            input_length=input_size,
                            trainable=False)(input_seq)

    encoded = Dense(encoding_dim, activation='relu',
                            trainable=False)(embedding_layer)

    repeat = RepeatVector(50)(input_type)
    concat = concatenate([encoded, repeat])

    rec = LSTM(128, input_shape=(None, 50), dropout=0., recurrent_dropout=0., go_backwards=True)(concat)

    out = Dense(4, activation="softmax")(rec)

    model = Model([input_seq, input_type], out)
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=["accuracy"])

    model.load_weights(path)

    return model
