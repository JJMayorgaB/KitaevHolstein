import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

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
KIND   = 'var'            # 'var' (variacional (P+AP)S, rama Cat+) o 'exact' (mapas viejos)
HERE   = os.path.dirname(os.path.abspath(__file__))
OUT    = os.path.join(HERE, 'Figures', 'cat')
CMAP   = 'inferno'

if KIND == 'var':
    nLam, nth = 300, 300
    SRC = os.path.join(HERE, 'Figures', 'cat', 'data',
                       f'cat_state_information_resources_PAPS_sign+1_omega0{omega0:.1f}'
                       f'_nL_{nLam}_nth_{nth}')
    TAG = f'{KIND}_nL{nLam}_nth{nth}_PAPS'
else:
    nLam, nth = 200, 200
    SRC = os.path.join(HERE, '..',
                       f'cat_state_information_resources_exact_omega0{omega0:.1f}'
                       f'_nL_{nLam}_nth_{nth}')
    TAG = f'{KIND}_nL{nLam}_nth{nth}'
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


def _fmt_cut_axes(ax):
    ax.set_xlabel(r'$\Lambda$')
    ax.set_xlim(Lam_arr[0], Lam_arr[-1])
    ax.set_xticks(ytick_pos)
    ax.set_xticklabels(ytick_lab)
    ax.tick_params(axis='both', direction='in', bottom=True, top=True,
                   left=True, right=True)


def _cut_entanglement(theta_cut, name):
    """E_N y S vs Λ en un corte de θ fijo, superpuestos."""
    it = int(np.argmin(np.abs(theta_arr - theta_cut)))
    print(f'  corte E_N/S en θ = {theta_arr[it]:.4f}  '
          f'({theta_arr[it]/np.pi:.4f} π, índice {it})')

    C_EN = plt.cm.seismic(0.88)
    C_S  = plt.cm.seismic(0.12)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(Lam_arr, EN[:, it], '-',  color=C_EN, lw=1.8, label=r'$E_N$')
    ax.plot(Lam_arr, S[:, it],  '-', color=C_S,  lw=1.8, label=r'$S$')
    ax.axhline(1.0, color='k', ls=':', lw=1.2)
    ax.set_xticks([0, np.sqrt(2), 2*np.sqrt(2), 3*np.sqrt(2), 4*np.sqrt(2)])
    ax.set_xticklabels([r'$0$', r'$\sqrt{2}$', r'$2\sqrt{2}$', r'$3\sqrt{2}$', r'$4\sqrt{2}$'])
    ax.set_ylabel(r'$E_N,\ S$')
    _fmt_cut_axes(ax)
    ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
              loc='lower right', borderpad=0.5, handlelength=2.0, labelspacing=0.3)
    plt.tight_layout()
    _save(fig, name)


def _cut_marginal(theta_cut, name):
    """δ1 vs Λ en un corte de θ fijo."""
    it = int(np.argmin(np.abs(theta_arr - theta_cut)))
    print(f'  corte δ1 en θ = {theta_arr[it]:.4f}  '
          f'({theta_arr[it]/np.pi:.4f} π, índice {it})')

    C1 = plt.cm.seismic(0.88)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(Lam_arr, d1[:, it], '-', color=C1, lw=1.8, label=r'$\delta_1$')
    ax.set_yticklabels(['0', r'$1/\pi$'])
    ax.set_yticks([0, 1/np.pi])
    ax.set_ylabel(r'$\delta_1$')
    _fmt_cut_axes(ax)
    ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
              loc='lower right', borderpad=0.5, handlelength=2.0, labelspacing=0.3)
    plt.tight_layout()
    _save(fig, name)


# ── 1. negatividades marginales: escala desde 0 ──
_pair(d1, d2, r'(a)', r'(b)', r'$\delta_i$', 'heatmap_marginal_negativity')

# ── 2. negatividad logarítmica ─────────────────────────────────────
_single(EN, r'$E_N$', 'heatmap_log_negativity')

# ── 3. entropía de von Neumann ─────────────────────────────────────
_single(S, r'$S$', 'heatmap_vn_entropy')

# ── 4. control de malla: escala desde el mínimo de los datos ──
_pair(n1, n2, r'(a)', r'(b)', r'$\int W_i$', 'heatmap_wigner_norm', vmin=None)

# ── 5. corte E_N y S en θ = π/4 ───────────────────────────────────
_cut_entanglement(np.pi/4, 'cut_entanglement_theta_pi4')

# ── 6. corte δ1 en θ = π ──────────────────────────────────────────
_cut_marginal(np.pi, 'cut_marginal_theta_pi')

print('Listo.')