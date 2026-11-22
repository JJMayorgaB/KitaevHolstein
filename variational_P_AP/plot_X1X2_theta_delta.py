# -*- coding: utf-8 -*-
"""
FIGURA: segundo momento cruzado <X1 X2> en el marco recentrado (X_i^(r) = X_i + λ_i/√2) del
estado base en el plano (θ, Δ/t), a partir de los datos (P+AP)S que genera X1X2_theta_delta.py
en Figures/paper_X1X2/data/.

Se cargan los dos valores (sector par y sector impar) y se escoge, punto a punto, el del
estado base con δE = E_e − E_o:  δE < 0 -> par,  δE > 0 -> impar.  La frontera δE = 0 se
dibuja con línea discontinua. Imprime además la tabla de validación si existe.

Uso:  python plot_X1X2_theta_delta.py                  (ε1 = ε2 = 1, Λ/√2 = 3)
      python plot_X1X2_theta_delta.py eps1 eps2 L      (L = Λ/√2, los mismos del cálculo)
"""

import os
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from X1X2_theta_delta import (DATA_DIR, THETA, DELTA, f_odd, f_even, f_val, _tag,
                              EPS1_DEF, EPS2_DEF, L_DEF)

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}\usepackage[utf8]{inputenc}',
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'font.size': 26,
    'axes.labelsize': 26,
    'axes.titlesize': 26,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 20,
    'figure.titlesize': 24,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'grid.alpha': 0.3,
    'grid.color': 'gray',
    'axes.axisbelow': True,
})

OUT_DIR  = os.path.join(HERE, 'Figures', 'paper_X1X2')
CMAP     = 'seismic_r'    # <X1 X2> < 0 (impar) en rojo, > 0 (par) en azul: mismos colores que la Fig. 2

if len(sys.argv) == 1:
    e1, e2, L = EPS1_DEF, EPS2_DEF, L_DEF                # ε1 = ε2 = 1, Λ/√2 = 3
elif len(sys.argv) < 4:
    sys.exit('uso: python plot_X1X2_theta_delta.py [eps1 eps2 L]    (L = Λ/√2)')
else:
    e1, e2, L = (float(v) for v in sys.argv[1:4])

# ══════════════════════════════════════════════════════════════════
#  CARGA
# ══════════════════════════════════════════════════════════════════
_needed = [f_odd(e1, e2, L), f_even('E', e1, e2, L), f_even('X1X2', e1, e2, L)]
_missing = [f for f in _needed if not os.path.exists(f)]
if _missing:
    sys.exit('Faltan datos (corre primero X1X2_theta_delta.py):\n  '
             + '\n  '.join(os.path.basename(f) for f in _missing))

_odd = np.loadtxt(f_odd(e1, e2, L))                      # columnas: θ, E_o, <X1 X2>_r,o
th, E_o, C_o = _odd[:, 0], _odd[:, 1], _odd[:, 2]
E_e = np.loadtxt(f_even('E', e1, e2, L))                 # filas = Δ, columnas = θ
C_e = np.loadtxt(f_even('X1X2', e1, e2, L))

dE  = E_e - E_o[None, :]
X12 = np.where(dE < 0.0, C_e, C_o[None, :])              # <X1 X2>_r del estado base

print(f'<X1 X2>_r del estado base ∈ [{X12.min():+.4e}, {X12.max():+.4e}]')
if os.path.exists(f_val(e1, e2, L)):
    R = np.atleast_2d(np.loadtxt(f_val(e1, e2, L)))
    print(f'Validación: max |col − lab| = {np.max(np.abs(R[:, 7] - R[:, 8])):.2e} (fórmula), '
          f'max |var − lab| = {np.max(np.abs(R[:, 6] - R[:, 8])):.2e} (ansatz), '
          f'max |lab − lab+| = {np.max(np.abs(R[:, 8] - R[:, 9])):.2e} (truncación)')

# ══════════════════════════════════════════════════════════════════
#  FIGURA
# ══════════════════════════════════════════════════════════════════
vmax = np.max(np.abs(X12))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

fig, ax = plt.subplots(figsize=(6.0, 4.0))
im = ax.imshow(X12, origin='lower', aspect='auto', cmap=CMAP, norm=norm,
               extent=[th[0], th[-1], DELTA[0], DELTA[-1]])
if dE.min() < 0.0 < dE.max():
    ax.contour(th, DELTA, dE, levels=[0.0], colors='0.15', linewidths=1.75, linestyles='--')

ax.set_xticks([0.0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2])
ax.set_xticklabels([r'$0$', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])
ax.set_xlim(th[0], th[-1])
ax.set_ylim(DELTA[0], DELTA[-1])
ax.set_yticks([v for v in np.arange(np.ceil(2*DELTA[0])/2, DELTA[-1] + 1e-9, 0.5)
               if v > DELTA[0]])                    # un solo 0 en la esquina (el del eje x)
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$\Delta/t$')
ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True,
               color='black', size=9, width=2.2)

cb = fig.colorbar(im, ax=ax, pad=0.03)
cb.set_label(r'$\langle\hat X_1\hat X_2\rangle$')
cb.ax.tick_params(direction='in')

NAME = f'X1X2_theta_delta_{_tag(e1, e2, L)}_PAPS'
os.makedirs(OUT_DIR, exist_ok=True)
fig.savefig(os.path.join(OUT_DIR, f'{NAME}.png'), bbox_inches='tight', transparent=False, dpi=300)
fig.savefig(os.path.join(OUT_DIR, f'{NAME}.svg'), bbox_inches='tight', transparent=False)
plt.show()
print(f'Figura guardada en: {os.path.join(OUT_DIR, NAME)}.png / .svg')
