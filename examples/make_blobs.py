#!/usr/bin/env python3

from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from pandas import DataFrame

# generate 3d classification dataset
X, y = make_blobs(n_samples=1000, centers=4, n_features=3)
# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], z=X[:,2], label=y))
train, test = train_test_split(df, test_size=0.2)
# write to disk space-separated
train.to_csv('blob_train.csv', index=False, header=False, sep=' ')
test.to_csv('blob_test.csv', index=False, header=False, sep=' ')
