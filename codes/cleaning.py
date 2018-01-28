# Removed few unwanted columns and replaced NaN with column mean

import pandas as pd
import numpy as np
import csv
from six.moves import cPickle as pickle

path = r"..\Flight_Data\CAX_Train_Takeoff\takeoff"
dest_path =  r"..\Clean_Data\Train\takeoff"

for i in range(5):
    data = pd.read_csv(path+str(i+1)+".csv")
    data.fillna(data.mean())
    data.to_csv(dest_path + str(i+1) + ".csv", encoding='utf-8')

test_dataset = []
test_labels = []

path = r"..\Flight_Data\CAX_Test_Takeoff\CAX_Test_Takeoff.csv"
dest_path =  r"..\Clean_Data\Test\takeoff.csv"
data = pd.read_csv(path)
data.fillna(data.mean())
data.to_csv(dest_path, encoding='utf-8')

