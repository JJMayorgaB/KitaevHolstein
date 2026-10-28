import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import sys
import subprocess

# Rama del gato y J_α/ω0:  python script.py +1 1.0   (|Cat_+>, J_α/ω0 = 1.0)
# Sin argumentos genera todas las ramas y todas las J_α/ω0 de CV_LIST, cada una con sus figuras
# (se omiten las combinaciones que todavía no tienen datos).
CV_LIST = (0.5, 1.0, 1.5)
if len(sys.argv) == 1:
    for _sg in ('+1', '-1'):
        for _cv in CV_LIST:
            subprocess.run([sys.executable, os.path.abspath(__file__), _sg, str(_cv)], check=True)
    sys.exit(0)
SIGN = int(sys.argv[1])
CV   = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}\usepackage[utf8]{inputenc}',
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'font.size': 22,
    'axes.labelsize': 22,
    'axes.titlesize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'figure.titlesize': 20,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'grid.alpha': 0.3,
    'grid.color': 'gray',
    'axes.axisbelow': True,
})

# ══════════════════════════════════════════════════════════════════
#  PARÁMETROS
# ══════════════════════════════════════════════════════════════════
omega0 = 1.0
HERE   = os.path.dirname(os.path.abspath(__file__))
OUT    = os.path.join(HERE, 'Figures', 'cat', 'info_resources', 'relations')
CMAP   = 'inferno'

# malla de cada conjunto de datos
nLam_ex,  nth_ex  = 300, 300     # exacto (carpeta KitaevHolstein, subcarpeta cv{...})
nLam_var, nth_var = 300, 300     # variacional (P+AP)S, rama SIGN, subcarpeta cv{...}

LAM_LIST = [2*np.sqrt(2), 3*np.sqrt(2), 4*np.sqrt(2)]
LAM_LAB  = [r'2\sqrt{2}', r'3\sqrt{2}', r'4\sqrt{2}']

SRC_EX  = os.path.join(HERE, '..',
                       f'cat_state_information_resources_exact_sign{SIGN:+d}_omega0{omega0:.1f}'
                       f'_nL_{nLam_ex}_nth_{nth_ex}', f'cv{CV}')
SRC_VAR = os.path.join(HERE, 'Figures', 'cat', 'data',
                       f'cat_state_information_resources_PAPS_sign{SIGN:+d}_omega0{omega0:.1f}'
                       f'_nL_{nLam_var}_nth_{nth_var}', f'cv{CV}')
TAG = f'nLex{nLam_ex}_nLvar{nLam_var}_PAPS_sign{SIGN:+d}_cv{CV}'
os.makedirs(OUT, exist_ok=True)
for _src in (SRC_EX, SRC_VAR):
    if not os.path.isdir(_src):
        print(f'sign={SIGN:+d}, J/ω0={CV}: no está {_src}, se omite')
        sys.exit(0)


def _load(src):
    return dict(
        Lam   = np.loadtxt(os.path.join(src, 'coord_Lam.txt')),
        theta = np.loadtxt(os.path.join(src, 'coord_theta.txt')),
        d1    = np.loadtxt(os.path.join(src, 'delta1.txt')),
        d2    = np.loadtxt(os.path.join(src, 'delta2.txt')),
        EN    = np.loadtxt(os.path.join(src, 'EN.txt')),
        S     = np.loadtxt(os.path.join(src, 'S.txt')),
    )


EX  = _load(SRC_EX)
VAR = _load(SRC_VAR)

print(f'Exacto      {SRC_EX}:  {EX["d1"].shape[0]} Λ × {EX["d1"].shape[1]} θ')
print(f'Variacional {SRC_VAR}: {VAR["d1"].shape[0]} Λ × {VAR["d1"].shape[1]} θ')

# factor geométrico: 0 en θ = nπ/2 (estado producto),
#                    1 en θ = (2n+1)π/4 (acoplamientos simétricos)
EX['c']  = np.abs(np.sin(2.0*EX['theta']))
VAR['c'] = np.abs(np.sin(2.0*VAR['theta']))


def _save(fig, name):
    for ext in ('jpg', 'svg'):
        fig.savefig(os.path.join(OUT, f'{name}_{TAG}.{ext}'),
                    bbox_inches='tight', transparent=False, dpi=300)
    plt.show()
    print(f'  guardado: {name}_{TAG}.jpg / .svg')


def _curve(xkey, ykey, xlabel, ylabel, name, Lam_target, Lam_label):
    """y vs x en un corte de Λ fijo, coloreado por |sin 2θ|.
       Círculos = exacto,  cuadrados con borde negro = variacional."""

    def _pick(D):
        iL = int(np.argmin(np.abs(D['Lam'] - Lam_target)))
        x  = xkey(D, iL)
        y  = D[ykey][iL]
        return iL, x, y

    iL_ex,  x_ex,  y_ex  = _pick(EX)
    iL_var, x_var, y_var = _pick(VAR)

    fig, ax = plt.subplots(figsize=(6.0, 5.5))

    sc = ax.scatter(x_ex, y_ex, c=EX['c'], cmap=CMAP, s=26,
                    marker='o', edgecolors='none', vmin=0.0, vmax=1.0,
                    zorder=3)
    ax.scatter(x_var, y_var, c=VAR['c'], cmap=CMAP, s=34,
               marker='s', edgecolors='none',
               vmin=0.0, vmax=1.0, zorder=2)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.025)
    cbar.ax.set_title(r'$|\sin 2\theta|$', pad=16)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.ax.tick_params(direction='in')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='both', direction='in', bottom=True, top=True,
                   left=True, right=True)
    ax.text(0.96, 0.95, rf'$\Lambda = {Lam_label}$',
            transform=ax.transAxes, ha='right', va='top', fontsize=22)

    leg = [mlines.Line2D([], [], color='gray', marker='o', ls='none',
                         markersize=7, markeredgecolor='none', label='exact'),
           mlines.Line2D([], [], color='gray', marker='s', ls='none',
                         markersize=7, markeredgecolor='none',
                         markeredgewidth=0.6, label='variational')]
    ax.legend(handles=leg, frameon=True, edgecolor='black', framealpha=0.0,
              fancybox=False, loc='best', borderpad=0.5,
              handlelength=1.2, labelspacing=0.3)

    plt.tight_layout()
    _save(fig, name)
    print(f'    Λ pedido = {Lam_target:.4f}  |  exacto {EX["Lam"][iL_ex]:.4f} '
          f'(i={iL_ex})  variacional {VAR["Lam"][iL_var]:.4f} (i={iL_var})')


_dsum = lambda D, iL: D['d1'][iL] + D['d2'][iL]
_d1   = lambda D, iL: D['d1'][iL]

# ── 1. E_N vs δ1 + δ2 ─────────────────────────────────────────────
for L, lab in zip(LAM_LIST, LAM_LAB):
    _curve(_dsum, 'EN', r'$\delta_1 + \delta_2$', r'$E_N$',
           f'EN_vs_deltasum_Lam{L:.3f}', L, lab)

# ── 2. E_N vs δ1 ──────────────────────────────────────────────────
for L, lab in zip(LAM_LIST, LAM_LAB):
    _curve(_d1, 'EN', r'$\delta_1$', r'$E_N$',
           f'EN_vs_delta1_Lam{L:.3f}', L, lab)

# ── 3. S vs δ1 + δ2 ───────────────────────────────────────────────
for L, lab in zip(LAM_LIST, LAM_LAB):
    _curve(_dsum, 'S', r'$\delta_1 + \delta_2$', r'$S$',
           f'S_vs_deltasum_Lam{L:.3f}', L, lab)

# ── 4. S vs δ1 ────────────────────────────────────────────────────
for L, lab in zip(LAM_LIST, LAM_LAB):
    _curve(_d1, 'S', r'$\delta_1$', r'$S$',
           f'S_vs_delta1_Lam{L:.3f}', L, lab)

print('Listo.')