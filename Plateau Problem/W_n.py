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
A = (np.linspace(0,2*np.pi,N+1))
h = 2*np.pi/N

W_N = np.zeros((N,N))
  # Fonction à intégrer
for i in tqdm(range(N)):
  for j in tqdm(range(N)):
    def func(x, y):
        ph_i = 0
        ph_j = 0

        ph_i = 0
        x_i = A[i]
        x_i_ = x_i
        x_im = A[(i-1)]
        x_ip = A[(i+1)]
        if i == 0 : 
            x_i, x_ip, x_im  = A[0],A[1],A[-2]
            x_i_ = A[-1]
        if x < x_i_ and x_im <= x :
            ph_i = -N/(2*np.pi)
        elif x < x_ip and x_i <= x :
            ph_i = N/(2*np.pi)

        ph_i = 0
        x_j = A[j]
        x_j_ = x_j
        x_jm = A[(j-1)]
        x_jp = A[(j+1)]
        if j == 0 : 
            x_j, x_jp, x_jm  = A[0],A[1],A[-2]
            x_j_ = A[-1]
        if x < x_j_ and x_jm <= x :
            ph_j = -N/(2*np.pi)
        elif x < x_jp and x_j <= x :
            ph_j = N/(2*np.pi)



        return np.log((np.sin(np.abs(x-y)/2)) + 0.0000000001 )*ph_i*ph_j


    # Calcul de l'intégrale
    res, err = dblquad(func, ax, bx, lambda x: ay, lambda x: by,)
    W_N[i,j] = res

# Affichage du résultat
print("Résultat de l'intégrale :", res)
print(err)