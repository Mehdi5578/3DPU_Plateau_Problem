from scipy.integrate import dblquad  # Module d'intégration "dblquad"
import numpy as np
from tqdm import tqdm
import yaml
from yaml.loader import SafeLoader

with open(r"C:\Users\oudao\OneDrive\Documents\Montréal 4A\Les études\Chair_SCALE_AI\3dPU\paths\MRI_data.yaml", 'r') as f:
    data = yaml.load(f, Loader=SafeLoader)


# Intervalle d'intégration
ax = 0
bx = 2*np.pi
ay = 0
by = 2*np.pi
N = data["Plateau_meta"]["N"]
A = (np.linspace(0,2*np.pi,N+19))
h = 2*np.pi/N

W_N = np.zeros((N,N))
  # Fonction à intégrer
for i in tqdm(range(N)):
  for j in tqdm(range(N)):
    def func(x, y):
        ph_i = 0
        ph_j = 0

        if x < A[i] and A[(i-1)] <= x :
          ph_i = -N/(2*np.pi)
        elif x < A[(i+1)] and A[i] <= x :
          ph_i = N/(2*np.pi)

        if y < A[j] and A[(j-1)] <= y :
          ph_j = -N/(2*np.pi)
        elif y < A[(j+1)] and A[j] <= y :
          ph_j = N/(2*np.pi)



        return np.log((np.sin(np.abs(x-y)/2)) + 0.0000000001 )*ph_i*ph_j


    # Calcul de l'intégrale
    res, err = dblquad(func, ax, bx, lambda x: ay, lambda x: by,)
    W_N[i,j] = res

# Affichage du résultat
print("Résultat de l'intégrale :", res)
print(err)