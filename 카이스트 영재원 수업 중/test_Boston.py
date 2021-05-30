import pandas as pd
import tensorflow as tf
from tensorflow.python.util.nest import _yield_value

def test_machine_learning():
    data = pd.read_csv('카이스트 영재원 수업 중/boston.csv')
    print(data.shape)
    print(data.columns)
    x_data = data[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']]
    y_data = data[['medv']]
    print(x_data.shape, y_data.shape)
    X = tf.keras.layers.Input(shape=[13])
    Y = tf.keras.layers.Dense(1)(X)
    model = tf.keras.models.Model(X, Y)
    model.compile(loss='mse')
    model.fit(x_data, y_data, epochs = 10000)

test_machine_learning()