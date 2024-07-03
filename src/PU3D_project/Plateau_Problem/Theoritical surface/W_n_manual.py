import numpy as np

import yaml
from yaml.loader import SafeLoader

with open(r"C:\Users\oudao\OneDrive\Documents\Montréal 4A\Les études\Chair_SCALE_AI\3dPU\paths\MRI_data.yaml", 'r') as f:
    data = yaml.load(f, Loader=SafeLoader)

# N = data["Plateau_meta"]["N"]
N = 1000
h = np.pi*2/N



def R(i,j,k):
    return 2*np.cos(k*h*(i-j))  - np.cos(k*h*(i-j-1)) - np.cos(k*h*(i-j+1))


def D(i,j) : 
    S = 0
    for k in range(1,1000):
        S += 2*R(i,j,k)/(k*k*k)
    return -2*np.log(2)*(h**2) - S


def S(i,j) :
    return D(i,j+1)/h - D(i,j)/h


def W(i,j):
    return (S(i,j) - S(i+1,j))/(4*h*np.pi)

if __name__ == '__main__':
    pass





