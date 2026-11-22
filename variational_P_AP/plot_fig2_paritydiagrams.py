# -*- coding: utf-8 -*-
"""
FIGURA 2 del paper (mapas de paridad, 2x2) a partir de los datos (P+AP)S que genera
fig2_paritydiagrams.py en Figures/paper_fig2/data/. Mismas especificaciones que la celda 28
de anisotropic_kitaev_holstein.ipynb (estilo de la celda 1):
    (a) mapa isótropo (g, Δ)  | (b) Δ/t = 0.5
    (c) Δ/t = 1               | (d) Δ/t = 1.5
    colorbar vertical a la derecha, centrada, de la altura de un panel
Además imprime una verificación de (c): con Δ = t, la frontera exacta son las rectas
g1 = sqrt(ε1 ω0) y g2 = sqrt(ε2 ω0).
Los datos de (b)–(d) son del primer cuadrante; con MIRROR = True se reconstruye el plano
completo por reflexión (el espectro es par en cada g_i); con MIRROR = False se grafica
solo el primer cuadrante.
Uso:  python plot_fig2_paritydiagrams.py
"""

import os
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fig2_paritydiagrams import (DATA_DIR, DELTA_2D, EPS1_2D, EPS2_2D, f_coord, f_dE_a,
                                 f_dE_2d)
from variational_P_AP import omega0

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

OUT_DIR = os.path.join(HERE, 'Figures', 'paper_fig2')
NAME    = 'parity_maps_2x2_PAPS'
MIRROR  = False                   # True: plano completo por reflexión; False: solo g1, g2 ≥ 0

# ══════════════════════════════════════════════════════════════════
#  CARGA
# ══════════════════════════════════════════════════════════════════
_needed = [f_coord('g_a'), f_coord('Delta_a'), f_coord('g_2d_Q1'), f_dE_a()] + [f_dE_2d(D) for D in DELTA_2D]
_missing = [f for f in _needed if not os.path.exists(f)]
if _missing:
    sys.exit('Faltan datos (corre primero fig2_paritydiagrams.py):\n  '
             + '\n  '.join(os.path.basename(f) for f in _missing))

g_iso     = np.loadtxt(f_coord('g_a'))
Delta_iso = np.loadtxt(f_coord('Delta_a'))
g_2d      = np.loadtxt(f_coord('g_2d_Q1'))                    # primer cuadrante, g_2d[0] = 0
dE_iso    = np.loadtxt(f_dE_a())                              # filas = Δ, columnas = g
dE_2d     = {D: np.loadtxt(f_dE_2d(D)) for D in DELTA_2D}     # filas = g2, columnas = g1

if MIRROR:
    # reflexión g_i -> -g_i (sin duplicar g = 0): índices del cuadrante para cada punto del plano
    k0     = 1 if np.isclose(g_2d[0], 0.0) else 0
    idx    = np.concatenate([np.arange(len(g_2d) - 1, k0 - 1, -1), np.arange(len(g_2d))])
    g_2d   = np.concatenate([-g_2d[k0:][::-1], g_2d])
    dE_2d  = {D: M[np.ix_(idx, idx)] for D, M in dE_2d.items()}


# ══════════════════════════════════════════════════════════════════
#  VERIFICACIÓN DE (c): Δ = t  ->  fronteras exactas g_i = sqrt(ε_i ω0)
# ══════════════════════════════════════════════════════════════════
def _zeros(x, y):
    """Ceros de y(x) por interpolación lineal entre cambios de signo."""
    s = np.sign(y)
    idx = np.where(s[:-1]*s[1:] < 0)[0]
    return x[idx] - y[idx]*(x[idx+1] - x[idx])/(y[idx+1] - y[idx])


if 1.0 in dE_2d:
    dE_c = dE_2d[1.0]
    i0 = np.argmin(np.abs(g_2d))                              # fila/columna más cercana a 0
    z1 = _zeros(g_2d, dE_c[i0, :])                            # cruces en g1 con g2 ≈ 0
    z2 = _zeros(g_2d, dE_c[:, i0])                            # cruces en g2 con g1 ≈ 0
    print('Verificación (c), Δ = t:')
    print(f'  cruces en g1 (g2 = {g_2d[i0]:+.4f}): {np.round(z1, 4)}   exacto: ±{np.sqrt(EPS1_2D*omega0):.4f}')
    print(f'  cruces en g2 (g1 = {g_2d[i0]:+.4f}): {np.round(z2, 4)}   exacto: ±{np.sqrt(EPS2_2D*omega0):.4f}')


# ══════════════════════════════════════════════════════════════════
#  FIGURA 2x2 — mapas de paridad
# ══════════════════════════════════════════════════════════════════
norm_p = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

fig = plt.figure(figsize=(10.0, 8.0))
outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.0, 0.03], wspace=0.08)
gs    = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0],
                                         hspace=0.35, wspace=0.38)
gs_cb = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[1],
                                         height_ratios=[0.5, 1.0, 0.5], hspace=0.0)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0], sharex=ax_b, sharey=ax_b)
ax_d = fig.add_subplot(gs[1, 1], sharex=ax_b, sharey=ax_b)
cax  = fig.add_subplot(gs_cb[1, 0])


def _ticks(ax, col='white'):
    ax.tick_params(axis='both', direction='in', bottom=True, top=True,
                   left=True, right=True, color=col, size=5, width=1.5)


# ---------- (a) mapa isotrópico (g, Δ) ----------
ext_iso = [g_iso[0], g_iso[-1], Delta_iso[0], Delta_iso[-1]]
im = ax_a.imshow(np.tanh(dE_iso), origin='lower', aspect='auto',
                 extent=ext_iso, cmap='seismic', norm=norm_p)
ax_a.contour(g_iso, Delta_iso, dE_iso, levels=[0.0], colors='0.15',
             linewidths=1.75, linestyles='--')
ax_a.set_xlabel(r'$g/t$')
ax_a.set_ylabel(r'$\Delta/t$')
ax_a.set_ylim(0.1, 1.5)
ax_a.text(0.05, 0.95, r'(a)', transform=ax_a.transAxes, ha='left', va='top', color='white')
_ticks(ax_a)

# ---------- (b),(c),(d) mapas (g1, g2) ----------
ext_2d = [g_2d[0], g_2d[-1], g_2d[0], g_2d[-1]]
L1, L2 = np.meshgrid(g_2d, g_2d)

for ax, D, lab in zip([ax_b, ax_c, ax_d], DELTA_2D, [r'(b)', r'(c)', r'(d)']):
    ax.imshow(np.tanh(dE_2d[D]), origin='lower', aspect='auto',
              extent=ext_2d, cmap='seismic', norm=norm_p)
    ax.contour(L1, L2, dE_2d[D], levels=[0.0], colors='0.15',
               linewidths=1.5, linestyles='--')
    ax.set_xticks([-2, -1, 0, 1, 2] if MIRROR else [0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2] if MIRROR else [0, 1, 2])
    ax.set_xlabel(r'$g_1/t$')
    ax.set_ylabel(r'$g_2/t$')
    ax.text(0.05, 0.95, lab, transform=ax.transAxes,
            ha='left', va='top', color='white')
    _ticks(ax)

# ---------- colorbar vertical común ----------
cb = fig.colorbar(im, cax=cax, orientation='vertical', ticks=[-1, 0, 1])
cb.set_label(r'$\tanh(\delta E/t)$')
cb.ax.tick_params(direction='in')

os.makedirs(OUT_DIR, exist_ok=True)
fig.savefig(os.path.join(OUT_DIR, f'{NAME}.png'),
            bbox_inches='tight', transparent=False, dpi=300)
fig.savefig(os.path.join(OUT_DIR, f'{NAME}.svg'),
            bbox_inches='tight', transparent=False)
plt.show()
print(f'Figura guardada en: {os.path.join(OUT_DIR, NAME)}.png / .svg')
