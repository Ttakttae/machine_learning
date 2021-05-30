import pandas as pd
import tensorflow as tf

def test_machine_learning():
    보스턴 = pd.read_csv('카이스트 영재원 수업 중/boston.csv')
    독립 = 보스턴[['crim', 'zn']]
    

test_machine_learning()