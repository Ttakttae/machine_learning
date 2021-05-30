import pandas as pd
import tensorflow as tf

def test_machine_learning():
    data = pd.read_csv('https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv')
    print(data.shape)

    x_data = data[['온도']]
    y_data = data[['판매량']]
    print(x_data.shape, y_data.shape)

    X = tf.keras.layers.Input(shape=[1])
    Y = tf.keras.layers.Dense(1)(X)
    model = tf.keras.models.Model(X,Y)
    model.compile(loss='mse')

    model.fit(x_data, y_data, epochs=10000)

    model.predict(x_data)

    model.get_weights()

    판매량 =  1.9876952 * 온도 + 0.29019067

test_machine_learning()