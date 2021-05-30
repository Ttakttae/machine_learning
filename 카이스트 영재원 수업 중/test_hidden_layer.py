import pandas as pd
import tensorflow as tf

#학습 자료 수집

data = pd.read_csv('카이스트 영재원 수업 중/boston.csv')
print(data.shape)
print(data.columns)
x_data = data[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']]
y_data = data[['medv']]
print(x_data.shape, y_data.shape)
   
def test_machine_learning():
    #모델 설정 및 학습 및 출력
    X = tf.keras.layers.Input(shape=[13])
    H = tf.keras.layers.Dense(5, activation='swish')(X)
    Y = tf.keras.layers.Dense(1)(H)
    model = tf.keras.models.Model(X, Y)
    model.compile(loss='mse')
    model.fit(x_data, y_data, epochs = 10000)
    model.predict(x_data[0:5])
    print(data)
    model.get_weights()

test_machine_learning()