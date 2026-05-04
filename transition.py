import numpy as np

def logistic_transition(z, gamma, c):
    return 1 / (1 + np.exp(-gamma * (z - c)))

def exponential_transition(z, gamma, c):
    return 1 - np.exp(-gamma * (z - c)**2)

def identity_transition(z, c):
    return (z > c).astype(float)
