import pandas as pd
import tensorflow as tf

#학습 자료 수집
data = pd.read_csv('카이스트 영재원 수업 중/boston.csv')
print(data.shape)
print(data.columns)
x_data = data[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']]
y_data = data[['medv']]
print(x_data.shape, y_data.shape)
   
def machine_learning():
    #모델 설정 및 학습 및 출력
    a = 10
    while a >= 0:
        X = tf.keras.layers.Input(shape=[13])
        H1 = tf.keras.layers.Dense(5, activation='swish')(X)
        H2 = tf.keras.layers.Dense(5, activation='swish')(H1)
        H3 = tf.keras.layers.Dense(5, activation='swish')(H2)
        H4 = tf.keras.layers.Dense(5, activation='swish')(H3)
        H5 = tf.keras.layers.Dense(5, activation='swish')(H4)
        H6 = tf.keras.layers.Dense(5, activation='swish')(H5)
        Y = tf.keras.layers.Dense(1)(H6)
        model = tf.keras.models.Model(X, Y)
        model.compile(loss='mse')
        model.fit(x_data, y_data, epochs = 10000)
        a -= 1
    model.predict(x_data[0:5])
    print(data)
    model.get_weights()

machine_learning()