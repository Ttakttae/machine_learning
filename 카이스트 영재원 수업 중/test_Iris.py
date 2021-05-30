import pandas as pd
import tensorflow as tf

#학습 자료 수집

data = pd.read_csv('카이스트 영재원 수업 중/iris.csv')
onehot = pd.get_dummies(data)

x_data = onehot[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
y_data = onehot[['품종_setosa', '품종_versicolor', '품종_virginica']]
print(x_data.shape, y_data.shape)

def test_machine_learning():
    #모델 설정 및 학습 및 출력
    X = tf.keras.layers.Input(shape=[4])
    Y = tf.keras.layers.Dense(3, activation='softmax')(X)
    model = tf.keras.models.Model(X,Y)
    model.compile(loss='categorical_crossentropy', metrics='accuracy')
    model.fit(x_data, y_data, epochs = 10000)
    model.predict(x_data[-5:])
    print(y_data[-5:])

test_machine_learning()