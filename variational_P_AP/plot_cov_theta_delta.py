# -*- coding: utf-8 -*-
"""
FIGURA: covarianza Cov(X1, X2) del estado base en el plano (θ, Δ/t), a partir de los datos
(P+AP)S que genera cov_theta_delta.py en Figures/fig_X1X2/data/ (prefijo cov_).

Se cargan las dos covarianzas (sector par y sector impar) y se escoge, punto a punto, la del
estado base con δE = E_e − E_o:  δE < 0 -> par,  δE > 0 -> impar.  La frontera δE = 0 se
dibuja con línea discontinua. Imprime además el resumen de la validación si existe.
Las figuras van a Figures/fig_X1X2/CovX1X2/.

Uso:  python plot_cov_theta_delta.py                  (todos los (ε1, ε2, Λ/√2) de SWEEP)
      python plot_cov_theta_delta.py eps1 eps2 L      (L = Λ/√2, los mismos del cálculo)
"""

import os
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from X1X2_theta_delta import SWEEP
from cov_theta_delta import DELTA, f_odd, f_even, f_val, _tag

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

OUT_DIR = os.path.join(HERE, 'Figures', 'fig_X1X2', 'CovX1X2')
CMAP    = 'seismic_r'     # Cov < 0 (impar) en rojo, Cov > 0 (par) en azul: mismos colores que la Fig. 2


def _fmt(v):
    """Una cifra decimal como máximo: 0.5 -> '0.5', 1.0 -> '1'."""
    s = f'{abs(v):.1f}'.rstrip('0').rstrip('.')
    return r'$0$' if s == '0' else (r'$-' if v < 0 else r'$') + s + r'$'


def make_figure(e1, e2, L, show=True):
    _needed = [f_odd(e1, e2, L), f_even('E', e1, e2, L), f_even('Cov', e1, e2, L)]
    _missing = [f for f in _needed if not os.path.exists(f)]
    if _missing:
        print('  Faltan datos (corre primero cov_theta_delta.py), se omite:\n    '
              + '\n    '.join(os.path.basename(f) for f in _missing))
        return

    _odd = np.loadtxt(f_odd(e1, e2, L))                    # columnas: θ, E_o, Cov_o
    th, E_o, C_o = _odd[:, 0], _odd[:, 1], _odd[:, 2]
    E_e = np.loadtxt(f_even('E', e1, e2, L))               # filas = Δ, columnas = θ
    C_e = np.loadtxt(f_even('Cov', e1, e2, L))

    dE  = E_e - E_o[None, :]
    X12 = np.where(dE < 0.0, C_e, C_o[None, :])            # covarianza del estado base

    print(f'  Cov(X1, X2) del estado base ∈ [{X12.min():+.4e}, {X12.max():+.4e}]')
    if os.path.exists(f_val(e1, e2, L)):
        R = np.atleast_2d(np.loadtxt(f_val(e1, e2, L)))
        print(f'  Validación: max |col − lab| = {np.max(np.abs(R[:, 7] - R[:, 8])):.2e} (fórmula), '
              f'max |var − lab| = {np.max(np.abs(R[:, 6] - R[:, 8])):.2e} (ansatz), '
              f'max |lab − lab+| = {np.max(np.abs(R[:, 8] - R[:, 9])):.2e} (truncación)')

    vmax = np.ceil(np.round(10.0*np.max(np.abs(X12)), 6))/10.0   # extremo con una cifra decimal (hacia arriba)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    im = ax.imshow(X12, origin='lower', aspect='auto', cmap=CMAP, norm=norm,
                   extent=[th[0], th[-1], DELTA[0], DELTA[-1]])
    if dE.min() < 0.0 < dE.max():
        ax.contour(th, DELTA, dE, levels=[0.0], colors='0.15', linewidths=1.75, linestyles='--')

    ax.set_xticks([0.0, np.pi/4, np.pi/2])
    ax.set_xticklabels([r'$0$', r'$\pi/4$', r'$\pi/2$'])
    ax.set_yticks([1.0, 2.0])
    ax.set_yticklabels([r'$1$', r'$2$'])
    ax.set_xlim(th[0], th[-1])
    ax.set_ylim(DELTA[0], DELTA[-1])
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\Delta/t$')
    ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True,
                   color='black', size=9, width=2.2)

    cb = fig.colorbar(im, ax=ax, pad=0.03, ticks=[-vmax, 0.0, vmax])
    cb.ax.set_yticklabels([_fmt(-vmax), _fmt(0.0), _fmt(vmax)])
    cb.set_label(r'$\mathrm{Cov}(\hat X_1,\hat X_2)$')
    cb.ax.tick_params(direction='in')

    NAME = f'cov_theta_delta_{_tag(e1, e2, L)}_PAPS'
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, f'{NAME}.png'), bbox_inches='tight', transparent=False, dpi=300)
    fig.savefig(os.path.join(OUT_DIR, f'{NAME}.svg'), bbox_inches='tight', transparent=False)
    print(f'  Figura guardada en: {os.path.join(OUT_DIR, NAME)}.png / .svg')
    if show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sets, show = SWEEP, False
    elif len(sys.argv) < 4:
        sys.exit('uso: python plot_cov_theta_delta.py [eps1 eps2 L]    (L = Λ/√2)')
    else:
        sets, show = [tuple(float(v) for v in sys.argv[1:4])], True
    for e1, e2, L in sets:
        print(f'ε1 = {e1:g}, ε2 = {e2:g}, Λ/√2 = {L:g}')
        make_figure(e1, e2, L, show)
