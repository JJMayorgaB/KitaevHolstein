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
#  Cortes en θ a Λ fijo: exacto (marginal_negativities.py, marcadores) contra
#  variacional (P+AP)S (marginal_negativities_variational.py, líneas). Rama SIGN.
# ══════════════════════════════════════════════════════════════════
HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(HERE, 'Figures', 'cat', 'data')
OUT  = os.path.join(HERE, 'Figures', 'cat', 'info_resources', 'cuts')

LAM_LIST = [2*np.sqrt(2), 3*np.sqrt(2)]
LAM_LAB  = [r'2\sqrt{2}', r'3\sqrt{2}']
os.makedirs(OUT, exist_ok=True)
if not any(os.path.exists(os.path.join(SRC, f'marginal_negativity_vs_theta_{k}_sign{SIGN:+d}_cv{CV}_{L}.txt'))
           for k in ('exact', 'PAPS') for L in LAM_LIST):
    print(f'sign={SIGN:+d}, J/ω0={CV}: sin cortes en θ, se omite')
    sys.exit(0)

C1 = plt.cm.seismic(0.88)          # δ1, E_N
C2 = plt.cm.seismic(0.12)          # δ2, S

# columnas: theta delta1 delta2 norm1 norm2 E_N S rank_eff
COL = dict(theta=0, d1=1, d2=2, n1=3, n2=4, EN=5, S=6, rk=7)


def _load(kind, Lam):
    f = os.path.join(SRC, f'marginal_negativity_vs_theta_{kind}_sign{SIGN:+d}_cv{CV}_{Lam}.txt')
    if not os.path.exists(f):
        print(f'  (no está {os.path.basename(f)}: se grafica sin él)')
        return None
    A = np.atleast_2d(np.loadtxt(f))
    print(f'  {os.path.basename(f)}: {A.shape[0]} θ,  min ∫W = '
          f'{np.nanmin(A[:, [COL["n1"], COL["n2"]]]):.6f}')
    return A


xtick_pos = np.arange(0, 5) * np.pi/4
xtick_lab = [r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']
PANEL_LABELS = [[r'(a)', r'(b)'],
                [r'(c)', r'(d)']]
ROWS = [(('d1', r'$\delta_1$', C1), ('d2', r'$\delta_2$', C2), r'$\delta_i$'),
        (('EN', r'$E_N$',      C1), ('S',  r'$S$',        C2), r'$E_N,\ S$')]

fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), sharex=True)

for c, (Lam, lab) in enumerate(zip(LAM_LIST, LAM_LAB)):
    EX  = _load('exact', Lam)
    VAR = _load('PAPS',  Lam)
    for r, (q1, q2, ylab) in enumerate(ROWS):
        ax = axes[r, c]
        for key, qlab, col in (q1, q2):
            if VAR is not None:
                ax.plot(VAR[:, 0], VAR[:, COL[key]], '-', color=col, lw=1.8, zorder=2)
            if EX is not None:
                ax.plot(EX[:, 0], EX[:, COL[key]], 'o', color=col, mfc='none',
                        mew=1.2, ms=5, zorder=3)
        if r == 1:
            ax.axhline(1.0, color='k', ls=':', lw=1.2)
            ax.set_xlabel(r'$\theta$')
        if c == 0:
            ax.set_ylabel(ylab)
        ax.set_xlim(0.0, np.pi)
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_lab)
        ax.tick_params(axis='both', direction='in', bottom=True, top=True,
                       left=True, right=True)
        ax.text(0.04, 0.95, PANEL_LABELS[r][c], transform=ax.transAxes,
                ha='left', va='top')
        if r == 0:
            ax.set_title(rf'$\Lambda = {lab}$', pad=10)

# escala común por fila, con espacio arriba para etiquetas y leyendas
for r in range(2):
    lo = min(axes[r, c].get_ylim()[0] for c in range(2))
    hi = max(max(l.get_ydata().max() for l in axes[r, c].get_lines()
                 if len(l.get_ydata()) > 2) for c in range(2))
    for c in range(2):
        axes[r, c].set_ylim(lo, 1.25*hi)

# leyendas: magnitud (color) en la columna izquierda, método (estilo) en la derecha
LEG = dict(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
           loc='upper center', ncol=2, borderpad=0.4, handlelength=1.6,
           columnspacing=1.0)
for r, (q1, q2, _) in enumerate(ROWS):
    h = [mlines.Line2D([], [], color=col, lw=1.8, label=qlab) for _, qlab, col in (q1, q2)]
    axes[r, 0].legend(handles=h, **LEG)
h = [mlines.Line2D([], [], color='gray', marker='o', mfc='none', mew=1.2, ls='none',
                   ms=6, label='exact'),
     mlines.Line2D([], [], color='gray', lw=1.8, label=r'$(\mathrm{P}+\mathrm{AP})\mathrm{S}$')]
axes[1, 1].legend(handles=h, **LEG)

plt.tight_layout()
name = f'cut_theta_marginal_entanglement_PAPS_sign{SIGN:+d}_cv{CV}'
for ext in ('jpg', 'svg'):
    fig.savefig(os.path.join(OUT, f'{name}.{ext}'), bbox_inches='tight',
                transparent=False, dpi=300)
plt.show()
print(f'  guardado: {name}.jpg / .svg')
