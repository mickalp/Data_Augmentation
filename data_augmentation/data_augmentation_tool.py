#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 20:44:46 2023

@author: michal
"""

import numpy as np
from queue import Queue
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import euclidean
from scipy.stats.qmc import PoissonDisk
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.decomposition import PCA, KernelPCA, SparsePCA, TruncatedSVD
from sklearn.metrics import r2_score
from boruta import BorutaPy
import pycaret.regression as py_reg
import pycaret.classification as py_clf
from imblearn.over_sampling import SMOTE
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from collections import deque
import umap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numba
from pycaret.regression import *
import lightgbm as lgb
from scipy import *
from sklearn import datasets


iris = datasets.load_iris()
data = iris.data, iris.feature_names
df = pd.DataFrame(iris.data, columns=iris.feature_names)



mapper = umap.UMAP(n_components=3, metric="minkowski", n_neighbors=20, min_dist=1.0)
mapping = mapper.fit_transform(df)



hull = ConvexHull(mapping)

for simplex in hull.simplices:
    plt.plot(mapping[simplex, 0], mapping[simplex, 1], 'k-')

def poisson_disc_sampling(points, radius=0.1):
    # Generate the Poisson disk samples within the convex hull
    tri = Delaunay(points)
    lb, ub = tri.min_bound, tri.max_bound
    size = np.abs(ub - lb)
    poisson = PoissonDisk(d=3, radius=radius)
    unit_samples = poisson.fill_space()
    samples = lb + size * unit_samples

    # Select the points that are within the convex hull
    simplex = tri.find_simplex(samples)
    samples = samples[simplex >= 0]
    return samples


# Generate random points in 3D space
points = mapping

# Compute the convex hull
hull = ConvexHull(points)

# Compute the Delaunay triangulation of the hull
tri = Delaunay(points)

# generate poisson disk samples within the convex hull
samples = poisson_disc_sampling(points, radius=0.1)

inverse_samples = mapper.inverse_transform(samples)
data_new = pd.concat([df, pd.DataFrame(inverse_samples, columns=df.columns)]).reset_index(drop=True)


py_reg.setup(df,
             target="petal width (cm)",
             preprocess=True,
             feature_selection=False,
             feature_selection_method="sequential",
             n_features_to_select=20,
             normalize=False,
             normalize_method="robust",
             remove_multicollinearity=True,
             multicollinearity_threshold=0.95,
             low_variance_threshold=0.1,
             pca=False,
             pca_method="linear",
             pca_components=6,
             profile=False)

best = py_reg.compare_models()

py_reg.setup(data_new,
             target="petal width (cm)",
             preprocess=True,
             feature_selection=False,
             feature_selection_method="sequential",
             n_features_to_select=20,
             normalize=False,
             normalize_method="robust",
             remove_multicollinearity=True,
             multicollinearity_threshold=0.95,
             low_variance_threshold=0.1,
             pca=False,
             pca_method="linear",
             pca_components=6,
             profile=False)

best = py_reg.compare_models()

