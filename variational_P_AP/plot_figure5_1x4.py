import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker

import sys
import subprocess

# Nueva figura 5 (1x4), solo (P+AP)S en los mapas:
#   (a) S,  (b) δ1,  (c) δ2   en el plano (θ, Λ/√2)
#   (d) corte en Λ = 3√2:  S (eje izquierdo, negro) y δ1, δ2 (eje derecho, rojo/azul);
#       líneas = (P+AP)S, círculos abiertos = exacto
# Rama del gato y J_α/ω0:  python plot_figure5_1x4.py +1 1.0   (|Cat_+>, J_α/ω0 = 1.0)
# Sin argumentos genera todas las ramas y todas las J_α/ω0 de CV_LIST
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

nLam_var, nth_var = 300, 300     # mapa variacional (P+AP)S, rama SIGN, subcarpeta cv{...}
LAM_CUT = 3*np.sqrt(2)           # corte del panel (d), el mismo Λ de marginal_negativities*.py
EX_STEP  = 2                     # en (d) se dibuja uno de cada EX_STEP puntos exactos

SRC_VAR = os.path.join(HERE, 'Figures', 'cat', 'data',
                       f'cat_state_information_resources_PAPS_sign{SIGN:+d}_omega0{omega0:.1f}'
                       f'_nL_{nLam_var}_nth_{nth_var}', f'cv{CV}')
SRC_CUT = os.path.join(HERE, 'Figures', 'cat', 'data')
TAG = f'nLvar{nLam_var}_PAPS_sign{SIGN:+d}_cv{CV}'
os.makedirs(OUT, exist_ok=True)

f_cut = {k: os.path.join(SRC_CUT, f'marginal_negativity_vs_theta_{k}_sign{SIGN:+d}_cv{CV}_{LAM_CUT}.txt')
         for k in ('exact', 'PAPS')}
for _f in [SRC_VAR] + list(f_cut.values()):
    if not os.path.exists(_f):
        print(f'sign={SIGN:+d}, J/ω0={CV}: no está {_f}, se omite')
        sys.exit(0)

# ── carga ─────────────────────────────────────────────────────────
Lam_arr   = np.loadtxt(os.path.join(SRC_VAR, 'coord_Lam.txt')) / np.sqrt(2.0)   # eje en Λ/√2
theta_arr = np.loadtxt(os.path.join(SRC_VAR, 'coord_theta.txt'))
MAPS = dict(S  = np.loadtxt(os.path.join(SRC_VAR, 'S.txt')),
            d1 = np.loadtxt(os.path.join(SRC_VAR, 'delta1.txt')),
            d2 = np.loadtxt(os.path.join(SRC_VAR, 'delta2.txt')))
extent = [theta_arr[0], theta_arr[-1], Lam_arr[0], Lam_arr[-1]]

# columnas de los cortes: theta delta1 delta2 norm1 norm2 E_N S rank_eff
COL = dict(theta=0, d1=1, d2=2, S=6)
EX  = np.atleast_2d(np.loadtxt(f_cut['exact']))
VAR = np.atleast_2d(np.loadtxt(f_cut['PAPS']))

print(f'Mapa (P+AP)S {SRC_VAR}: {MAPS["S"].shape}')
print(f'Corte Λ = {LAM_CUT:.6f}: exacto {EX.shape[0]} θ, (P+AP)S {VAR.shape[0]} θ')

# ── ticks ─────────────────────────────────────────────────────────
xtick_pos = np.arange(0, 5) * np.pi/2
xtick_lab = [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
_n_max    = int(np.floor(Lam_arr[-1]))
ytick_pos = np.arange(1, _n_max+1)                 # en cada marca, λ1 = λ2 = k en θ = π/4
ytick_lab = [rf'${k}$' for k in range(1, _n_max+1)]

cut_xpos = np.arange(0, 5) * np.pi/4
cut_xlab = [r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']


C1 = plt.cm.seismic(0.88)          # δ1 (rojo)
C2 = plt.cm.seismic(0.12)          # δ2 (azul)
CS = 'black'                       # S

PANEL_LABELS = [r'(a)', r'(b)', r'(c)', r'(d)']
LABEL_COLOR  = ['black', 'white', 'white']     # color de las etiquetas de los mapas

# ══════════════════════════════════════════════════════════════════
#  FIGURA 1x4
#  fila 0: vacía (bbox_inches='tight' la recorta); las colorbars van dentro
#          de los mapas, en la franja oscura inferior (Λ/√2 < 1), y la
#          leyenda del panel (d) dentro de (d), sobre S = 1
#  fila 1: paneles
# ══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20.0, 5.3))     # ancho 20 (texto en el PDF ~ infid_neg_1x2); alto para que los
                                          # paneles cuadrados llenen sus celdas (sin holgura horizontal)
gs  = gridspec.GridSpec(2, 5, figure=fig,
                        width_ratios=[1, 1, 1, 0.06, 1],
                        height_ratios=[0.055, 1], hspace=0.07, wspace=0.08)

ax_a = fig.add_subplot(gs[1, 0])
ax_b = fig.add_subplot(gs[1, 1], sharey=ax_a)     # sin sharex: cada mapa lleva sus propias etiquetas de θ
ax_c = fig.add_subplot(gs[1, 2], sharey=ax_a)
ax_d = fig.add_subplot(gs[1, 4])
for _ax in (ax_a, ax_b, ax_c, ax_d):
    _ax.set_box_aspect(1)                     # los cuatro paneles cuadrados y del mismo tamaño

IM = {}
for i, (ax, key) in enumerate([(ax_a, 'S'), (ax_b, 'd1'), (ax_c, 'd2')]):
    IM[key] = ax.imshow(MAPS[key], origin='lower', aspect='auto', extent=extent,
                        cmap=CMAP, vmin=0.0,
                        vmax=(max(np.nanmax(MAPS[key]), 1.0) if key == 'S'      # S: escala hasta 1 como mínimo,
                              else np.nanmax(MAPS[key])))                     # para que exista la marca 1.0
    # línea del corte del panel (d): Λ/√2 = 3, a lo ancho de todo el mapa
    ax.axhline(LAM_CUT/np.sqrt(2.0), color='white', ls='--', lw=1.6)
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lab if i == 0 else [''] + xtick_lab[1:])   # sin '0' junto al '2π' del panel anterior
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_lab)
    ax.tick_params(axis='both', direction='in', bottom=True, top=True,
                   left=True, right=True, color='white', size=7.5, width=2.0)
    ax.set_xlabel(r'$\theta$')
    ax.text(0.05, 0.95, PANEL_LABELS[i], transform=ax.transAxes,
            ha='left', va='top', color=LABEL_COLOR[i])
    if i == 0:
        ax.set_ylabel(r'$\Lambda/\sqrt{2}$', labelpad=5)
    else:
        ax.tick_params(axis='y', labelleft=False)


def _hcbar(ax, im, label, locator):
    """Colorbar horizontal dentro del mapa, en la franja oscura inferior,
       con marcas, contorno y etiqueta en blanco."""
    cax  = ax.inset_axes([0.07, 0.07, 0.55, 0.045])
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.locator = locator
    cbar.update_ticks()
    cax.xaxis.set_ticks_position('top')
    cax.tick_params(direction='in', labelsize=19, colors='white', length=3)
    cbar.outline.set_edgecolor('white')
    cbar.outline.set_linewidth(0.8)
    ax.text(0.66, 0.0925, label, transform=ax.transAxes, ha='left', va='center',
            color='white', fontsize=23)
    return cbar


_hcbar(ax_a, IM['S'],  r'$S$',        ticker.FixedLocator([0.0, 0.5, 1.0]))
_hcbar(ax_b, IM['d1'], r'$\delta_1$', ticker.MultipleLocator(0.1))
_hcbar(ax_c, IM['d2'], r'$\delta_2$', ticker.MultipleLocator(0.1))

# ── (d) corte en Λ = 3√2 ──────────────────────────────────────────
ax_r = ax_d.twinx()
ax_d.plot(VAR[:, COL['theta']], VAR[:, COL['S']], '-', color=CS, lw=1.8, zorder=2)
ax_d.plot(EX[::EX_STEP, COL['theta']],  EX[::EX_STEP, COL['S']],  'o', color=CS, mfc='none', mew=1.2, ms=5, zorder=3)
for key, col in (('d1', C1), ('d2', C2)):
    ax_r.plot(VAR[:, COL['theta']], VAR[:, COL[key]], '-', color=col, lw=1.8, zorder=2)
    ax_r.plot(EX[::EX_STEP, COL['theta']],  EX[::EX_STEP, COL[key]],  'o', color=col, mfc='none', mew=1.2, ms=5, zorder=3)
ax_d.axhline(1.0, color='k', ls=':', lw=1.2)

S_max = max(np.nanmax(VAR[:, COL['S']]), np.nanmax(EX[:, COL['S']]))
d_max = max(np.nanmax(VAR[:, [COL['d1'], COL['d2']]]), np.nanmax(EX[:, [COL['d1'], COL['d2']]]))
# escalas alineadas: S = 1 y el máximo de δ_i quedan a la misma altura
ax_d.set_ylim(0.0, 1.45*max(S_max, 1.0))           # franja sobre S = 1 para la leyenda
ax_r.set_ylim(0.0, 1.45*d_max*max(S_max, 1.0))
ax_d.set_xlim(0.0, np.pi)
ax_d.set_xticks(cut_xpos)
ax_d.set_xticklabels(cut_xlab)
ax_d.set_xlabel(r'$\theta$')
ax_d.set_yticks([0.5, 1.0])                      # sin 0.0 en el eje izquierdo
ax_r.set_yticks(np.arange(1, int(np.round(10*d_max*max(S_max, 1.0))) + 1)/10)   # 0.1, 0.2, ...: sin 0.0 y sin marcas sobre los datos
# etiquetas horizontales en las esquinas superiores, sobre la columna de números
# (que termina en S = 1): no ocupan ancho extra entre (c) y (d)
ax_d.set_ylabel(r'$S$', color=CS, rotation=0, ha='right', va='top')
ax_d.yaxis.set_label_coords(-0.03, 1.0)
ax_r.set_ylabel(r'$\delta_i$', rotation=0, ha='left', va='top')
ax_r.yaxis.set_label_coords(1.03, 1.0)
ax_d.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=False)
ax_r.tick_params(axis='y', direction='in', right=True)
ax_d.text(0.05, 0.95, PANEL_LABELS[3], transform=ax_d.transAxes, ha='left', va='top')

# leyenda del panel (d), dentro del panel sobre S = 1:
#   fila superior = magnitudes (color), fila inferior = método (estilo)
h = [Line2D([], [], color=CS, lw=1.8, label=r'$S$'),
     Line2D([], [], color=C1, lw=1.8, label=r'$\delta_1$'),
     Line2D([], [], color=C2, lw=1.8, label=r'$\delta_2$')]
leg_q = ax_d.legend(handles=h, loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False,
                    handlelength=1.3, columnspacing=0.9, handletextpad=0.35, borderaxespad=0.25,
                    fontsize=20)
ax_d.add_artist(leg_q)
h = [Line2D([], [], color='gray', lw=1.8, label=r'Variational'),
     Line2D([], [], color='gray', marker='o', mfc='none', mew=1.2, ls='none', ms=6, label=r'Exact')]
ax_d.legend(handles=h, loc='upper right', bbox_to_anchor=(1.0, 0.885), ncol=2, frameon=False,
            handlelength=1.0, columnspacing=0.8, handletextpad=0.3, borderaxespad=0.25, fontsize=18)

for ext in ('jpg', 'svg'):
    fig.savefig(os.path.join(OUT, f'figure5_1x4_{TAG}.{ext}'),
                bbox_inches='tight', transparent=False, dpi=300)
plt.show()
print(f'  guardado: figure5_1x4_{TAG}.jpg / .svg')
