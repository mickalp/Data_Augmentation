#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 20:44:46 2023

@author: michal
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 19:43:55 2023

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
#%%
df = pd.read_excel("/Users/michal/OneDrive - University of Gdansk/OneDrive - University of Gdansk (for Students)/irini/data_with_descriptors.xlsx")
df.head()

#%%
df["cell viability _(%)"] = df["cell viability _(%)"].replace({"?": np.nan}).astype(float)
df.loc[df["cell viability _(%)"] > 130, "cell viability _(%)"] = 130  # limit values to max 130
df.dropna(subset=["cell viability _(%)"], inplace=True)
df_numeric = df.select_dtypes(exclude=[object])
df_numeric.drop(columns=["Exposure dose_PPM", "cell viability _(%)"], inplace=True)

decomposer = TruncatedSVD(n_components=5)
transformed_data = decomposer.fit_transform(df_numeric)
explained_variances = decomposer.explained_variance_ratio_
plt.plot(explained_variances)

df.select_dtypes(exclude=[object]).shape
df_numeric.describe()
df_numeric.describe().loc["std", :]
df.select_dtypes(exclude=[object]).head()

X = df.select_dtypes(exclude=[object])
y = X["cell viability _(%)"].values
X = X.drop(columns=["Exposure dose_PPM", "cell viability _(%)"])

selected_features = ["HOMO energy [eV]", "LUMO energy [eV]", "Band gap [eV]", "HOMO boundries [eV]", "Dipole moment", "Sp"] + ["Exposure dose_PPM"]

py_reg.setup(df[selected_features + ["cell viability _(%)"]],
             target="cell viability _(%)",
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
             profile=False,
             use_gpu=True)

X = py_reg.get_config("X")
y = py_reg.get_config("y")

X_train = py_reg.get_config("X_train")
y_train = py_reg.get_config("y_train")

X_val = py_reg.get_config("X_test")
y_val= py_reg.get_config("y_test")

print(f"{X_train.shape[1]} features after processing")

X_train.head()

#%%

mapper = umap.UMAP(n_components=3, metric="minkowski", n_neighbors=100, min_dist=1.0)
data = X.join(y)
mapping = mapper.fit_transform(data)











