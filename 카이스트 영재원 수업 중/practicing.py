import pandas as pd
import tensorflow as tf

#학습 자료 수잡

파일경로 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'
레모네이드 = pd.read_csv(파일경로)
독립 = 레모네이드[['온도']]
종속 = 레모네이드[['판매량']]
print(레모네이드.shape)

파일경로 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
보스턴 = pd.read_csv(파일경로)
print(보스턴)

파일경로 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv'
아이리스 = pd.read_csv(파일경로)
print(아이리스)

def machine_learning():
    #모델 설정 및 학습 및 출력
    X = tf.keras.layers.Input(shape=[1])
    Y = tf.keras.layers.Dense(1)(X)
    model = tf.keras.models.Model(X,Y)
    model.compile(loss='mse')
    model.fit(독립, 종속, epochs=10000)
    model.predict(독립)
    print(독립)

machine_learning()