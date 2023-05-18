import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import fsolve



def load_from_csv(filename:str) -> np.ndarray:
    data = np.loadtxt(filename, delimiter=',')
    return data

def predict_label(sim:np.ndarray, threshold:float) -> np.ndarray:
    label = np.ones_like(sim)
    label[sim>threshold] = 0.0
    return label

def show_plot(data:np.ndarray, label:np.ndarray, split:float) -> np.ndarray:
    fig, ax = plt.subplots()
    ax.hist(data[label==0.0], density=True, alpha=0.5, bins=100, label="matchable")
    ax.hist(data[label==1.0], density=True, alpha=0.5, bins=100, label="dangling")
    ax.axvline(x=split, color='r', linestyle='--', label="mean")
    plt.show()

def show_raw_plot(data:np.ndarray) -> np.ndarray:
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(data, density=True, alpha=0.5, bins=100, label="all")
    bin_centers = (bins[:-1] + bins[1:]) / 2
    density = n / np.sum(n)
    return bin_centers, density


def compare_predict(raw, predict):
    true_0_pred_0 = np.sum((raw==0.0) & (predict==0.0))
    true_1_pred_0 = np.sum((raw==1.0) & (predict==0.0))
    true_0_pred_1 = np.sum((raw==0.0) & (predict==1.0))
    true_1_pred_1 = np.sum((raw==1.0) & (predict==1.0))
    return true_0_pred_0, true_1_pred_0, true_0_pred_1, true_1_pred_1

def double_gaussian(x, a1,b1, c1,a2, b2, c2):
    return a1/(c1 * np.sqrt(2*np.pi)) * np.exp(-(x-b1)**2 / (2*c1**2)) + a2/(c2 * np.sqrt(2*np.pi))* np.exp(-(x-b2)**2 / (2*c2**2))

def gaussian(x, a1, b1, c1):
    return a1/(c1 * np.sqrt(2*np.pi)) * np.exp(-(x-b1)**2 / (2*c1**2))

def double_cauchy(x, a1, x0, gamma0, a2, x1, gamma1):

    return a1 / (np.pi * gamma0 * (1 + ((x - x0) / gamma0) ** 2)) + a2 / (np.pi * gamma1 * (1 + ((x - x1) / gamma1) ** 2))

def cauchy(x, a1, x0, gamma0):
    return a1 / (np.pi * gamma0 * (1 + ((x - x0) / gamma0) ** 2))

def laplace(x, a, mu, b):
    return a / (2 * b) * np.exp(-abs(x - mu) / b)

def double_laplace(x, a1, mu1, b1, a2, mu2, b2):
    return a1 / (2 * b1) * np.exp(-abs(x - mu1) / b1) + a2 / (2 * b2) * np.exp(-abs(x - mu2) / b2)

def h(x, func, a1, x0, gamma0, a2, x1, gamma1):
    return func(x, a1, x0, gamma0) - func(x, a2, x1, gamma1)


