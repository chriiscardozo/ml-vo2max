import pandas as pd
import os
import random

file_name = 'Dados-medicos.txt'

data = []

with open(file_name) as f:
	for line in f.readlines()[1:]:
		sample = [float(x) for x in line.split()]
		data.append(sample)

dataset = pd.DataFrame(data=data, columns=['idade','peso','carga','vo2max'])

cut = 1000
mask = [1]*cut + [0]*(len(data)-cut)
random.seed(42)
random.shuffle(mask)

train_indexes = list(filter(lambda x: mask[x]==1, range(len(data))))
test_indexes = list(filter(lambda x: mask[x]==0, range(len(data))))

train = dataset.iloc[train_indexes]
test = dataset.iloc[test_indexes]

