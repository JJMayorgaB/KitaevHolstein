"""
Heatmaps de los recursos de información del gato bimodal en el plano (Λ, θ).

Lee los .txt de   cat_state_information_resources_omega0{W}_nL_{nL}_nth_{nth}/
y genera 4 figuras en Resultados_aniso/ (.jpg y .svg):
    1. δ1 y δ2   — dos paneles, colorbar compartida
    2. E_N
    3. S
    4. ∫W1 y ∫W2 — control de malla, dos paneles, colorbar compartida
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
#  PARÁMETROS — etiquetas del nombre de la carpeta de datos
# ══════════════════════════════════════════════════════════════════
omega0 = 1.0
nLam   = 32
nth    = 32
OUT    = 'Resultados_aniso'
CMAP   = 'inferno'

SRC = (f'cat_state_information_resources_omega0{omega0:.1f}'  f'_nL_{nLam}_nth_{nth}')
TAG = f'nL{nLam}_nth{nth}'
os.makedirs(OUT, exist_ok=True)

# ── carga ─────────────────────────────────────────────────────────
Lam_arr   = np.loadtxt(os.path.join(SRC, 'coord_Lam.txt'))
theta_arr = np.loadtxt(os.path.join(SRC, 'coord_theta.txt'))
d1 = np.loadtxt(os.path.join(SRC, 'delta1.txt'))
d2 = np.loadtxt(os.path.join(SRC, 'delta2.txt'))
EN = np.loadtxt(os.path.join(SRC, 'EN.txt'))
S  = np.loadtxt(os.path.join(SRC, 'S.txt'))
n1 = np.loadtxt(os.path.join(SRC, 'norm1.txt'))
n2 = np.loadtxt(os.path.join(SRC, 'norm2.txt'))

print(f'Leído {SRC}:  {d1.shape[0]} Λ × {d1.shape[1]} θ')
print(f'  δ1 ∈ [{np.nanmin(d1):.4f}, {np.nanmax(d1):.4f}]')
print(f'  δ2 ∈ [{np.nanmin(d2):.4f}, {np.nanmax(d2):.4f}]')
print(f'  E_N ∈ [{np.nanmin(EN):.4f}, {np.nanmax(EN):.4f}]')
print(f'  S   ∈ [{np.nanmin(S):.4f}, {np.nanmax(S):.4f}]')
print(f'  min ∫W = {min(np.nanmin(n1), np.nanmin(n2)):.4f}')

# filas = Λ, columnas = θ  →  extent = [θmin, θmax, Λmin, Λmax]
extent = [theta_arr[0], theta_arr[-1], Lam_arr[0], Lam_arr[-1]]

# ── ticks ─────────────────────────────────────────────────────────
xtick_pos = np.arange(0, 5) * np.pi/2
xtick_lab = [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']

# eje Λ en múltiplos de √2 (en cada marca, λ1 = λ2 = k)
_sq2   = np.sqrt(2.0)
_n_max = int(np.floor(Lam_arr[-1]/_sq2))
ytick_pos = np.array([k*_sq2 for k in range(1, _n_max+1)])
ytick_lab = ([r'$\sqrt{2}$'] + [rf'${k}\sqrt{{2}}$' for k in range(2, _n_max+1)])


def _fmt_axes(ax, show_ylabel=True, col_ticks='white'):
    ax.set_xlabel(r'$\theta$')
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lab)
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_lab)
    ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True, color=col_ticks, size=7.5, width=2.0)
    if show_ylabel:
        ax.set_ylabel(r'$\Lambda$', labelpad=5)
    else:
        ax.tick_params(axis='y', labelleft=False)


def _save(fig, name):
    for ext in ('jpg', 'svg'):
        fig.savefig(os.path.join(OUT, f'{name}_{TAG}.{ext}'), bbox_inches='tight', transparent=False, dpi=300)
    plt.show()
    print(f'  guardado: {name}_{TAG}.jpg / .svg')


def _single(data, label, name, col_ticks='white'):
    """Heatmap de un solo recurso. Escala de 0 al máximo del propio mapa."""
    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    im = ax.imshow(data, origin='lower', aspect='auto', extent=extent,  cmap=CMAP, vmin=0.0, vmax=np.nanmax(data))
    _fmt_axes(ax, True, col_ticks)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cbar.ax.set_title(label, pad=16)
    cbar.ax.tick_params(direction='in')
    plt.tight_layout()
    _save(fig, name)


def _pair(dA, dB, labA, labB, cbar_label, name, col_ticks='white', vmin=0.0):
    """Dos paneles con una única colorbar.
       vmin=0.0 por defecto; pásalo como None para usar el mínimo de los datos."""
    if vmin is None:
        vmin = min(np.nanmin(dA), np.nanmin(dB))
    vmax = max(np.nanmax(dA), np.nanmax(dB))

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5))
    for idx, (ax, data, lab) in enumerate(zip(axes, [dA, dB], [labA, labB])):
        im = ax.imshow(data, origin='lower', aspect='auto', extent=extent,cmap=CMAP, vmin=vmin, vmax=vmax)
        _fmt_axes(ax, show_ylabel=(idx == 0), col_ticks=col_ticks)
        ax.text(0.05, 0.95, lab, transform=ax.transAxes, ha='left', va='top', fontsize=22, color=col_ticks)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.025, shrink=0.85)
    cbar.ax.set_title(cbar_label, pad=16)
    cbar.ax.tick_params(direction='in')
    _save(fig, name)

# ── 1. negatividades marginales: escala desde 0 ──
_pair(d1, d2, r'(a)', r'(b)', r'$\delta_i$', 'heatmap_marginal_negativity')

# ── 2. negatividad logarítmica ─────────────────────────────────────
_single(EN, r'$E_N$', 'heatmap_log_negativity')

# ── 3. entropía de von Neumann ─────────────────────────────────────
_single(S, r'$S$', 'heatmap_vn_entropy')

# ── 4. control de malla: escala desde el mínimo de los datos ──
_pair(n1, n2, r'(a)', r'(b)', r'$\int W_i$', 'heatmap_wigner_norm', vmin=None)

print('Listo.')