
import numpy as np

d = 'cat_state_information_resources_exact_omega01.0_nL_200_nth_200'
Lam = np.loadtxt(f'{d}/coord_Lam.txt')
th  = np.loadtxt(f'{d}/coord_theta.txt')
d1  = np.loadtxt(f'{d}/delta1.txt')

L, T = np.meshgrid(Lam, th, indexing='ij')
lam1 = L * np.cos(T)

print(f'{"lam1":>6} {"n":>6} {"delta1 min":>12} {"delta1 max":>12} '
      f'{"Lam min":>9} {"Lam max":>9}')
for tgt in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    m = np.abs(lam1 - tgt) < 0.02
    if m.sum() == 0:
        continue
    print(f'{tgt:6.2f} {m.sum():6d} {d1[m].min():12.6f} {d1[m].max():12.6f} '
          f'{L[m].min():9.3f} {L[m].max():9.3f}')