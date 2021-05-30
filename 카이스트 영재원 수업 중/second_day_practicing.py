import pandas as pd
import tensorflow as tf

#학습 자료 수집

data = pd.read_csv('https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv')
print(data.shape)

x_data = data[['온도']]
y_data = data[['판매량']]
print(x_data.shape, y_data.shape)


def machine_learning():
    #모델 설정 및 학습 및 출력
    X = tf.keras.layers.Input(shape=[1])
    Y = tf.keras.layers.Dense(1)(X)
    model = tf.keras.models.Model(X,Y)
    model.compile(loss='mse')
    model.fit(x_data, y_data, epochs=10000)
    model.predict(x_data)
    model.get_weights()

machine_learning()