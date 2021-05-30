import pandas as pd
import tensorflow as tf

def test_machine_learning():
    data = pd.read_csv('카이스트 영재원 수업 중/boston.csv')
    print(data.shape)
    print(data.columns)
    x_data = data[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']]
    y_data = data[['medv']]
    print(x_data.shape, y_data.shape)

test_machine_learning()