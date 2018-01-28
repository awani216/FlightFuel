import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from six.moves import cPickle as pickle
import numpy as np
import pandas as pd

with open(r"../random_forest.pickle", 'rb') as fl:
    data = pickle.load(fl)
    train_dataset = data['train_dataset']
    train_labels = data['train_labels']
    test_dataset = data['test_dataset']
    attributes =  data['attributes']

corrmat = np.corrcoef(train_dataset, rowvar=False)
corrmat = abs(corrmat)
ticklabels = attributes
plt.imshow(corrmat, cmap=cm.viridis)
plt.colorbar()
plt.xticks(np.arange(len(attributes)), attributes, rotation='vertical')
plt.yticks(np.arange(len(attributes)), attributes)
plt.show()
