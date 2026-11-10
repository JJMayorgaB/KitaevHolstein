# -*- coding: utf-8 -*-
"""
Diagrama de paridad fermiónica δE = E_e − E_o en el plano (g/t, Δ/t), acoplamientos isótropos
(λ1 = λ2 = λ, Λ = √2 λ, g = λ ω0), con el ansatz (P+AP)S de variational_P_AP.py
(funcional cerrado, mismo minimize_sector). Figura 1x3, mismos parámetros salvo:
    (a) ε1 = ε2 = 0      (b) ε1 = ε2 = 0.5      (c) ε1 = ε2 = 1.0
Uso:  python parity_maps_eps_1x3.py
Los mapas δE se guardan en Figures/parity/data/ y no se recalculan si ya existen
(bórralos para recalcular); la figura va a Figures/parity/.
"""

import os
import sys
import shutil
import warnings

import numpy as np
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from variational_P_AP import minimize_sector, sector_params, omega0   # (P+AP)S, convención de las notas

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════
#  PARÁMETROS (los de Parity_Splitting_aniso_PAPS salvo ε1 = ε2)
# ══════════════════════════════════════════════════════════════════
N_JOBS    = -1
T_VAL     = 1.0                              # t (= ω0)
LAM_ARR   = np.linspace(0.0, 1.5, 250)       # λ = g/ω0  (eje g/t, ω0 = t)
DELTA_ARR = np.linspace(0.0, 1.5, 250)       # Δ/t
EPS_LIST  = [0.0, 0.5, 1.0]                  # ε1 = ε2 en cada panel (en unidades de t)
EPS_LAB   = [r'0', r'0.5\,t', r't']          # cómo se escribe en el título de cada panel

OUT_DIR  = os.path.join(HERE, 'Figures', 'parity')
DATA_DIR = os.path.join(OUT_DIR, 'data')


# ══════════════════════════════════════════════════════════════════
#  CÁLCULO: una fila en λ por tarea, con continuación (warm start) en λ
# ══════════════════════════════════════════════════════════════════
def _row(sector, eps_val, Delta):
    E = np.empty(len(LAM_ARR))
    w = None
    for k, lam in enumerate(LAM_ARR):
        Lam = np.sqrt(2.0)*lam
        eps, epsp, J = sector_params(sector, lam, lam, eps_val, eps_val, T_VAL, Delta)
        E[k], w = minimize_sector(sector, eps, epsp, J, Lam, w)
    return E


def _fname(eps_val):
    return os.path.join(DATA_DIR, f'parity_dE_iso_eps{eps_val}_t{T_VAL}'
                                  f'_nL{len(LAM_ARR)}_nD{len(DELTA_ARR)}_PAPS.txt')


def compute_dE(eps_val):
    """δE(Δ, λ) = E_e − E_o; filas = Δ, columnas = λ."""
    jobs = ([delayed(_row)('odd', eps_val, 0.0)]                     # el impar no depende de Δ
            + [delayed(_row)('even', eps_val, D) for D in DELTA_ARR])
    res = Parallel(n_jobs=N_JOBS, verbose=5)(jobs)
    E_odd  = res[0]
    E_even = np.array(res[1:])
    return E_even - E_odd[None, :]


def load_or_compute(eps_val):
    f = _fname(eps_val)
    if os.path.exists(f):
        print(f'ε = {eps_val}: leído {os.path.basename(f)}')
        return np.loadtxt(f)
    print(f'ε = {eps_val}: calculando {len(DELTA_ARR) + 1} filas × {len(LAM_ARR)} λ ...', flush=True)
    dE = compute_dE(eps_val)
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savetxt(f, dE, fmt='%.10e',
               header=(f'deltaE = E_even - E_odd, (P+AP)S | filas = Delta/t ({len(DELTA_ARR)}, '
                       f'{DELTA_ARR[0]}..{DELTA_ARR[-1]}), columnas = g/t ({len(LAM_ARR)}, '
                       f'{LAM_ARR[0]}..{LAM_ARR[-1]}) | eps1 = eps2 = {eps_val}, t = {T_VAL}, '
                       f'omega0 = {omega0}, lambda1 = lambda2'))
    return dE


# ══════════════════════════════════════════════════════════════════
#  FIGURA 1x3
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    DE = [load_or_compute(e) for e in EPS_LIST]
    for e, dE in zip(EPS_LIST, DE):
        print(f'  ε = {e}:  δE ∈ [{np.nanmin(dE):.4f}, {np.nanmax(dE):.4f}]')

    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    plt.rcParams.update({
        'text.usetex': shutil.which('latex') is not None,
        'text.latex.preamble': r'\usepackage{amsmath}',
        'mathtext.fontset': 'cm',
        'font.family': 'serif',
        'font.serif': ['Computer Modern'],
        'font.size': 26,
        'axes.labelsize': 26,
        'axes.titlesize': 24,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 20,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
    })

    extent = [LAM_ARR[0], LAM_ARR[-1], DELTA_ARR[0], DELTA_ARR[-1]]
    norm_p = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    ticks  = [0.0, 0.5, 1.0, 1.5]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), sharey=True,
                             gridspec_kw=dict(wspace=0.08))
    for i, (ax, lab, dE) in enumerate(zip(axes, EPS_LAB, DE)):
        im = ax.imshow(np.tanh(dE), origin='lower', aspect='auto', extent=extent,
                       cmap='seismic', norm=norm_p)
        ax.contour(LAM_ARR, DELTA_ARR, dE, levels=[0.0], colors='k',
                   linewidths=2.0, linestyles='--')
        ax.set_box_aspect(1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        if i > 0:
            ax.set_xticklabels([''] + [f'{v}' for v in ticks[1:]])   # sin '0.0' junto al '1.5' anterior
        ax.set_xlabel(r'$g/t$')
        ax.set_title(rf'$\varepsilon_1=\varepsilon_2={lab}$', pad=10)
    axes[0].set_ylabel(r'$\Delta/t$')

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label(r'$\tanh(\delta E)$')
    cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    name = f'parity_maps_eps_1x3_t{T_VAL}_PAPS'
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(OUT_DIR, f'{name}.{ext}'), bbox_inches='tight',
                    transparent=False, dpi=300)
    plt.show()
    print(f'  guardado: {name}.png / .svg')
