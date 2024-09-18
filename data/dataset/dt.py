import pandas as pd

x = pd.read_csv('my_project/data/KIMORE/Kimore_ex1/Train_X.csv', header=None).iloc[:, :].values

print(x.shape)