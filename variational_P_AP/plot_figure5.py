import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

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

# ══════════════════════════════════════════════════════════════════
#  PARÁMETROS
# ══════════════════════════════════════════════════════════════════
omega0 = 1.0
HERE   = os.path.dirname(os.path.abspath(__file__))
OUT    = os.path.join(HERE, 'Figures', 'cat', 'info_resources', 'heatmaps')
CMAP   = 'inferno'

nLam_ex,  nth_ex  = 300, 300     # exacto (carpeta KitaevHolstein, subcarpeta cv{...})
nLam_var, nth_var = 300, 300     # variacional (P+AP)S, rama SIGN, subcarpeta cv{...}

SRC_EX  = os.path.join(HERE, '..',
                       f'cat_state_information_resources_exact_sign{SIGN:+d}_omega0{omega0:.1f}'
                       f'_nL_{nLam_ex}_nth_{nth_ex}', f'cv{CV}')
SRC_VAR = os.path.join(HERE, 'Figures', 'cat', 'data',
                       f'cat_state_information_resources_PAPS_sign{SIGN:+d}_omega0{omega0:.1f}'
                       f'_nL_{nLam_var}_nth_{nth_var}', f'cv{CV}')
TAG = f'compare_nLex{nLam_ex}_nLvar{nLam_var}_PAPS_sign{SIGN:+d}_cv{CV}'
os.makedirs(OUT, exist_ok=True)
for _src in (SRC_EX, SRC_VAR):
    if not os.path.isdir(_src):
        print(f'sign={SIGN:+d}, J/ω0={CV}: no está {_src}, se omite')
        sys.exit(0)


def _load(src):
    D = dict(Lam   = np.loadtxt(os.path.join(src, 'coord_Lam.txt')),
             theta = np.loadtxt(os.path.join(src, 'coord_theta.txt')),
             S     = np.loadtxt(os.path.join(src, 'S.txt')),
             d1    = np.loadtxt(os.path.join(src, 'delta1.txt')),
             d2    = np.loadtxt(os.path.join(src, 'delta2.txt')))
    D['extent'] = [D['theta'][0], D['theta'][-1], D['Lam'][0], D['Lam'][-1]]
    return D


EX  = _load(SRC_EX)
VAR = _load(SRC_VAR)
print(f'Exacto      {SRC_EX}:  {EX["S"].shape}')
print(f'Variacional {SRC_VAR}: {VAR["S"].shape}')

# ── ticks ─────────────────────────────────────────────────────────
xtick_pos = np.arange(0, 5) * np.pi/2
xtick_lab = [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']

_sq2   = np.sqrt(2.0)
_n_max = int(np.floor(EX['Lam'][-1]/_sq2))
ytick_pos = np.array([k*_sq2 for k in range(1, _n_max+1)])
ytick_lab = ([r'$\sqrt{2}$'] + [rf'${k}\sqrt{{2}}$' for k in range(2, _n_max+1)])

ROWS = [('S',  r'$S$'),
        ('d1', r'$\delta_1$'),
        ('d2', r'$\delta_2$')]
PANEL_LABELS = [[r'(a)', r'(b)'],
                [r'(c)', r'(d)'],
                [r'(e)', r'(f)']]
LABEL_COLOR  = ['black', 'white', 'white']    # color de las etiquetas por fila

# altura de cada colorbar dentro de su fila: [hueco arriba, barra, hueco abajo]
CB_RATIOS = [0.20, 0.70, 0.10]

# ══════════════════════════════════════════════════════════════════
#  FIGURA 3x2
# ══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(10.0, 12.0))
gs  = gridspec.GridSpec(3, 3, figure=fig, width_ratios=[1, 1, 0.05],
                        hspace=0.10, wspace=0.10)

axes = np.empty((3, 2), dtype=object)
for r in range(3):
    for c in range(2):
        kw = {}
        if (r, c) != (0, 0):
            kw = dict(sharex=axes[0, 0], sharey=axes[0, 0])
        axes[r, c] = fig.add_subplot(gs[r, c], **kw)

for r, (key, cb_label) in enumerate(ROWS):
    vmax = max(np.nanmax(EX[key]), np.nanmax(VAR[key]))      # escala común por fila
    for c, D in enumerate([EX, VAR]):
        ax = axes[r, c]
        im = ax.imshow(D[key], origin='lower', aspect='auto', extent=D['extent'],
                       cmap=CMAP, vmin=0.0, vmax=vmax)
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_lab)
        ax.set_yticks(ytick_pos)
        ax.set_yticklabels(ytick_lab)
        ax.tick_params(axis='both', direction='in', bottom=True, top=True,
                       left=True, right=True, color='white', size=7.5, width=2.0)
        ax.text(0.05, 0.95, PANEL_LABELS[r][c], transform=ax.transAxes,
                ha='left', va='top', color=LABEL_COLOR[r])

        if r < 2:
            ax.tick_params(axis='x', labelbottom=False)
        else:
            ax.set_xlabel(r'$\theta$')
        if c > 0:
            ax.tick_params(axis='y', labelleft=False)
        else:
            ax.set_ylabel(r'$\Lambda$', labelpad=5)

    gs_cb = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[r, 2],
                                             height_ratios=CB_RATIOS, hspace=0.0)
    cax  = fig.add_subplot(gs_cb[1, 0])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_title(cb_label, pad=16)
    cbar.ax.tick_params(direction='in')

# ══════════════════════════════════════════════════════════════════
#  LLAVES DE COLUMNA  ┌───── Exact ─────┐
# ══════════════════════════════════════════════════════════════════
fig.canvas.draw()                             # fija las posiciones finales de los ejes

BR_GAP = 0.018     # separación entre el borde superior del panel y la llave
BR_ARM = 0.010     # longitud de los brazos verticales
BR_LW  = 1.5

for c, title in enumerate([r'Exact', r'Variational']):
    pos = axes[0, c].get_position()
    x0, x1 = pos.x0, pos.x1
    yb = pos.y1 + BR_GAP                      # altura de la línea horizontal
    for xs, ys in [([x0, x1], [yb, yb]),                  # horizontal
                   ([x0, x0], [yb, yb - BR_ARM]),         # brazo izquierdo
                   ([x1, x1], [yb, yb - BR_ARM])]:        # brazo derecho
        fig.add_artist(Line2D(xs, ys, transform=fig.transFigure,
                              color='black', lw=BR_LW, solid_capstyle='butt'))
    fig.text(0.5*(x0 + x1), yb + 0.006, title,
             ha='center', va='bottom', fontsize=26)

for ext in ('jpg', 'svg'):
    fig.savefig(os.path.join(OUT, f'heatmap_S_delta_{TAG}.{ext}'),
                bbox_inches='tight', transparent=False, dpi=300)
plt.show()
print(f'  guardado: heatmap_S_delta_{TAG}.jpg / .svg')