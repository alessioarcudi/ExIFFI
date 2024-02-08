# uncompyle6 version 3.8.0
# Python bytecode 3.7.0 (3394)
# Decompiled from: Python 3.9.13 (main, May 24 2022, 21:28:12) 
# [Clang 12.0.0 (clang-1200.0.32.29)]
# Embedded file name: /Users/alessio/Documents/diffi-extended/diffi_extended/simulation_setup.py
# Compiled at: 2022-05-23 15:13:59
# Size of source mod 2**32: 10146 bytes
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from numpy.random import random_sample

class sample_generator:

    def __init__(self, n, dim, interval=None):
        self.dim = dim
        self.n = n
        self.interval = interval
        self.sample = None
        self.X = None

    def generate_sample(self, seed=np.random.seed(seed=0), points_type='random', radius=1, center=(0, 0), m=1, b=0):
        self.type = points_type
        try:
            if self.interval == None:
                self.a, self.b = (0, 1)
            else:
                if self.interval != None:
                    if type(self.interval) == np.ndarray or type(self.interval) == list:
                        self.a, self.b = self.interval[0], self.interval[1]
        except ValueError:
            print('Oops!  That was no valid option for the interval.  Try again with [a,b] in a list or array')

        if self.type == 'random':
            X = (self.b - self.a) * random_sample((self.n, self.dim)) + self.a
            self.ay = self.a
            self.by = self.b
            self.X = X
        else:
            if self.type == 'circular':
                alpha = 2 * np.pi * np.random.random(self.n)
                r = radius + np.random.normal(loc=0.0, scale=(radius / 10), size=(self.n))
                x = r * np.cos(alpha) + center[0]
                y = r * np.sin(alpha) + center[1]
                self.X = np.array([x, y]).T
                self.a = center[0] - radius - 2 * radius / 10
                self.b = center[0] + radius + 2 * radius / 10
                self.ay = center[1] - radius - 2 * radius / 10
                self.by = center[1] + radius + 2 * radius / 10
            else:
                if self.type == 'linear':
                    x = (self.b - self.a) * random_sample(self.n) + self.a
                    y = m * x + b + np.random.normal(loc=0.0, scale=(np.sqrt(10)), size=(self.n))
                    self.ay = m * self.a - b - 3 * np.sqrt(10)
                    self.by = m * self.b + b + 3 * np.sqrt(10)
                    self.X = np.array([x, y]).T

    def plot2D_(self):
        if self.X.shape[1] == 2:
            X_2dim = self.X
        else:
            pca = PCA(n_components=2)
            X_2dim = pca.fit_transform(self.X)
        plt.figure(figsize=(12, 12))
        plt.scatter((X_2dim.T[0]), (X_2dim.T[1]), alpha=0.6, s=10)
        plt.show()

    def add_anomaly_points(self, k):
        if self.type == 'random':
            Xa = (self.b + (self.b - self.a) / 2 - self.a + (self.b - self.a) / 2) * random_sample((k, self.dim)) + self.a - (self.b - self.a) / 2
        else:
            if self.type == 'linear' or self.type == 'circular':
                xa = (+self.b + 1 - (self.a - 1)) * random_sample(k) + self.a - 1
                ya = (self.by + 1 - (self.ay - 1)) * random_sample(k) + self.ay - 1
                Xa = np.array([xa, ya]).T
            else:
                print('error type not correct')
        self.X = np.concatenate((self.X, Xa), axis=0)


import time
from sklearn.datasets import make_moons, make_blobs, make_circles

class toy_examples:

    def __init__(self, n_samples, outlier_fraction):
        self.n_samples = n_samples
        self.outlier_fraction = outlier_fraction
        self.n_outliers = int(self.outlier_fraction * self.n_samples)
        self.n_inliers = self.n_samples - self.n_outliers

    def blobs_toy_dataset(self, random_state=0, n_features=2, centers=[[0, 0]], cluster_std=0.5, outlier_random_state=23, low=-6, high=6):
        rng = np.random.RandomState(outlier_random_state)
        self.X = make_blobs(n_samples=(self.n_inliers), centers=centers, cluster_std=cluster_std, random_state=random_state, n_features=n_features)[0]
        self.X = np.concatenate([self.X, rng.uniform(low=low, high=high, size=(self.n_outliers, n_features))], axis=0)
        self.y = [0] * int(self.n_inliers) + [1] * int(self.n_outliers)

    def moons_toy_dataset(self, random_state=0, n_features=2, noise=0.05, outlier_random_state=23, low=-6, high=6):
        rng = np.random.RandomState(outlier_random_state)
        self.X = make_moons(n_samples=(self.n_inliers), noise=noise, random_state=random_state)[0] - np.array([0.5, 0.25])
        self.X = np.concatenate([self.X, rng.uniform(low=low, high=high, size=(self.n_outliers, n_features))], axis=0)
        self.y = [0] * int(self.n_inliers) + [1] * int(self.n_outliers)

    def circles_toy_dataset(self, random_state=0, n_features=2, noise=0.05, factor=0.8, outlier_random_state=23, low=-6, high=6):
        rng = np.random.RandomState(outlier_random_state)
        self.X = make_circles(n_samples=(self.n_inliers), noise=noise, random_state=random_state, factor=factor)[0]
        self.X = np.concatenate([self.X, rng.uniform(low=low, high=high, size=(self.n_outliers, n_features))], axis=0)
        self.y = [0] * int(self.n_inliers) + [1] * int(self.n_outliers)

    def _plot2D(self):
        if self.X.shape[1] == 2:
            X_2dim = self.X
        else:
            pca = PCA(n_components=2)
            X_2dim = pca.fit_transform(self.X)
        plt.figure(figsize=(12, 12))
        plt.scatter((X_2dim.T[0]), (X_2dim.T[1]), alpha=0.6, s=10)
        plt.show()


import pandas as pd
from scipy.io import loadmat
from enum import Enum
import mat73
data = pd.DataFrame([['Lympho', 148, 18, 6, 0.041],
                     ['WBC', 278, 30, 21, 0.056],
                     ['Glass', 214, 9, 9, 0.042],
                     ['Vowels', 1456, 12, 50, 0.034],
                     ['Cardio', 1831, 21, 176, 0.096],
                     ['Thyroid', 3772, 6, 93, 0.025],
                     ['Musk', 3062, 166, 97, 0.032],
                     ['Satimage-2', 5803, 36, 71, 0.012],
                     ['Letter Recognition', 1600, 32, 100, 0.0625],
                     ['Speech', 3686, 400, 61, 0.0165],
                     ['Pima', 768, 8, 268, 0.35],
                     ['Satellite', 6435, 36, 2036, 0.32],
                     ['Shuttle', 49097, 9, 3511, 0.07],
                     ['BreastW', 683, 9, 239, 0.35],
                     ['Arrhythmia', 452, 274, 66, 0.15],
                     ['Ionosphere', 351, 33, 126, 0.36],
                     ['Mnist', 7603, 100, 700, 0.092],
                     ['Optdigits', 5216, 64, 150, 0.03],
                     ['Http (KDDCUP99)', 567479, 3, 2211, 0.004],
                     ['ForestCover', 286048, 10, 2747, 0.009],
                     ['Mulcross', 262144, 4, 26214, 0.1],
                     ['Smtp (KDDCUP99)', 95156, 3, 30, 0.0003],
                     ['Mammography', 11183, 6, 260, 0.0232],
                     ['Annthyroid', 7200, 6, 534, 0.0742],
                     ['Pendigits', 6870, 16, 156, 0.0227],
                     ['Ecoli', 336, 7, 9, 0.026],
                     ['Wine', 129, 13, 10, 0.077],
                     ['Vertebral', 240, 6, 30, 0.125],
                     ['Yeast', 1364, 8, 64, 0.047],
                     ['Seismic', 2584, 11, 170, 0.065],
                     ['Heart', 224, 44, 10, 0.044]],
  columns=['Dataset', 'points', 'n_features', 'outliers', '(%)'])


def generating_toy_examples():
    np.random.seed(0)
    blobs_6k_05_7dim = toy_examples(6000, 0.05)
    blobs_6k_05_7dim.blobs_toy_dataset(n_features=7, centers=[np.random.choice((range(10)), size=7, replace=False) for i in range(7)], cluster_std=(list(1.5 * np.random.random_sample(7) + 0.5)), low=([-15] + [-2] * 6), high=([15] + [2] * 6))
    blobs_6k_05_2dim = toy_examples(6000, 0.05)
    blobs_6k_05_2dim.blobs_toy_dataset(n_features=2, centers=[np.random.choice((range(3)), size=2, replace=False) for i in range(3)], cluster_std=(list(1.5 * np.random.random_sample(3) + 0.5)), low=(-6), high=6)
    moons_6k_05_y = toy_examples(6000, 0.05)
    moons_6k_05_y.moons_toy_dataset(n_features=2, noise=0.05, low=[-1, -6], high=[1, 6])
    moons_6k_05_x = toy_examples(6000, 0.05)
    moons_6k_05_x.moons_toy_dataset(n_features=2, noise=0.05, low=[-6, -1], high=[6, 1])
    circles_6k_05_y = toy_examples(6000, 0.05)
    circles_6k_05_y.circles_toy_dataset(low=[-1, -6], high=[1, 6])
    circles_6k_05_x = toy_examples(6000, 0.05)
    circles_6k_05_x.circles_toy_dataset(low=[-6, -1], high=[6, 1])
    return (
     blobs_6k_05_7dim, blobs_6k_05_2dim, moons_6k_05_y, moons_6k_05_x, circles_6k_05_y, circles_6k_05_x)
