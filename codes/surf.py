import matplotlib.pyplot as plt
import numpy as np
from six.moves import cPickle as pickle
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import random

with open(r"../random_forest.pickle", 'rb') as fl:
    data = pickle.load(fl)
    train_dataset = data['train_dataset']
    train_labels = data['train_labels']
    test_dataset = data['test_dataset']
    attributes =  data['attributes']
for i in range(len(attributes)):
    print(attributes[i]," ",i)
i = 29
j = 28
ran = random.sample(range(len(train_dataset)), 1000)

X = train_dataset[ran, i]
Y = train_dataset[ran, j]
Z = train_labels[ran]

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_trisurf(X, Y, Z, linewidth=0.2, antialiased=True)

plt.show()

