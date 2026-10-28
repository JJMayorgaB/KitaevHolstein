# -*- coding: utf-8 -*-
"""
Figuras de los estados gato con (P+AP)S, a partir de los datos que guarda variational_P_AP_cat.py
en Figures/cat/data/cat_PAPS/ (no recalcula nada). Las figuras (sufijo _PAPS) se guardan en
Figures/cat/ con los mismos nombres de antes.
Uso:  python variational_P_AP_cat.py        (datos, una vez)
      python plot_variational_P_AP_cat.py   (figuras, cada vez que se cambie el estilo)
"""

import io
import os
import shutil
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline, RegularGridInterpolator

warnings.filterwarnings('ignore')

# parámetros de la corrida (los mismos con los que se generaron los datos)
from variational_P_AP_cat import (OUT_DIR, DATA_DIR, GRIDS, cv_list, RHO, CV_2X2, WIG_POINTS,
                                  LAM_JOINT, THETA_JOINT, SECTOR_JOINT, SIGN_JOINT, joint_tag)


# ══════════════════════════════════════════════════════════════════
#  LECTURA DE LOS DATOS
# ══════════════════════════════════════════════════════════════════
def _load_named(path):
    """Columnas de un .txt cuyo encabezado termina con la línea de nombres."""
    with open(path, encoding='utf-8') as f:
        names = [ln for ln in f if ln.startswith('#')][-1][1:].split()
    A = np.loadtxt(path, ndmin=2)
    return {k: A[:, i] for i, k in enumerate(names)}


def load_sweeps():
    return {(g, cv): _load_named(os.path.join(DATA_DIR, f'sweep_{g}_cv{cv}.txt'))
            for g in GRIDS for cv in cv_list}


def load_negativity(name):
    d = _load_named(os.path.join(DATA_DIR, f'negativity_{name}.txt'))
    return dict(Lam=d['Lam'], dp=d['delta_plus'], dm=d['delta_minus'],
                np_=d['norm_plus'], nm=d['norm_minus'])


def _cuts(W, xvec, pvec):
    """Cortes p = 0 y x = 0 (los mismos de wigner_and_cuts_*)."""
    ip0 = np.argmin(np.abs(pvec))
    ix0 = np.argmin(np.abs(xvec))
    return W[ip0, :], W[:, ix0]


# ══════════════════════════════════════════════════════════════════
#  ESTILO (el original; si no hay LaTeX, mathtext con fuente cm)
# ══════════════════════════════════════════════════════════════════
_BASE_RC = {
    'font.size': 24,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 18,
    'figure.titlesize': 22,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'grid.alpha': 0.3,
    'grid.color': 'gray',
    'axes.axisbelow': True,
}


def _latex_available():
    if shutil.which('latex') is None:
        return False
    try:
        with plt.rc_context({'text.usetex': True}):
            fig = plt.figure(figsize=(1, 1))
            fig.text(0.5, 0.5, r'$\alpha$')
            fig.savefig(io.BytesIO(), format='png')
        plt.close('all')
        return True
    except Exception:
        plt.close('all')
        return False


def setup_style():
    HAS_LATEX = _latex_available()
    rc = dict(_BASE_RC)
    rc.update({
        'text.usetex': HAS_LATEX,
        'text.latex.preamble': r'\usepackage{amsmath}\usepackage[utf8]{inputenc}',
        'mathtext.fontset': 'cm',
        'font.family': 'serif',
        'font.serif': ['Computer Modern'] if HAS_LATEX else ['DejaVu Serif'],
    })
    plt.rcParams.update(rc)
    print('[estilo] ' + ('LaTeX disponible: usetex' if HAS_LATEX
                         else 'LaTeX no disponible: mathtext (cm)'))
    return HAS_LATEX


# ══════════════════════════════════════════════════════════════════
#  UTILIDADES DE GRÁFICAS (originales)
# ══════════════════════════════════════════════════════════════════
def _nice_tick(m):
    for step in [0.5, 0.25, 0.2, 0.1, 0.05]:
        if step <= m:
            return step
    return np.round(m, 2)


def _fmt_ticks(vals):
    return ['0' if abs(v) < 1e-9 else f'{v:.2f}' for v in vals]


def _lo_hi(cut, m, tol):
    hp = np.max(cut) >  tol
    hn = np.min(cut) < -tol
    if hp and hn: return (-m, m)
    if hp:        return (0.0, m)
    if hn:        return (-m, 0.0)
    return (-m, m)


def _smooth(y, window=7, poly=3):
    y = np.asarray(y, float)
    m = np.isfinite(y)
    if m.sum() < window:
        return y
    ys = y.copy()
    if (~m).any():
        idx = np.arange(len(y))
        ys[~m] = np.interp(idx[~m], idx[m], y[m])
    win = min(window, len(ys) - (1 - len(ys) % 2))
    if win < poly + 2:
        return y
    if win % 2 == 0:
        win -= 1
    out = savgol_filter(ys, win, poly)
    out[~m] = np.nan
    return out


def _spline_smooth(x, y, s_factor=1e-4):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return y
    xv, yv = x[m], y[m]
    s = s_factor * len(yv) * np.nanvar(yv)
    spl = UnivariateSpline(xv, yv, k=2, s=s)
    out = np.full_like(y, np.nan)
    out[m] = spl(xv)
    return out


class SciFormatter(ticker.ScalarFormatter):
    """Notación científica con multiplicador común y cifras fijas en los ticks."""
    def __init__(self, fformat='%1.1f'):
        self.fformat = fformat
        super().__init__(useOffset=False, useMathText=True)

    def _set_format(self):
        self.format = r'$%s$' % self.fformat


def plot_wigner_stack_num(Ws, cuts_p0, cuts_x0, xvec, pvec, cmap='RdBu',
                          axis_lim=(-4.2, 4.2), ylim=(-2.1, 2.1),
                          data_aspect=0.5, cut_margin=1.15, panel_frac=0.30,
                          tol=1e-3, labeleven=True, figsize=(6.0, 5.5), suptitle=None,
                          savepath=None, fig=None, label_offset=0):
    re = xvec; im = pvec; n = len(Ws)
    X = axis_lim[1]-axis_lim[0]; Y = ylim[1]-ylim[0]; Fw, Fh = figsize
    vmax = max(np.max(np.abs(W)) for W in Ws)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    if fig is None:
        fig = plt.figure(figsize=figsize)
    wm = 0.62; hm = data_aspect*wm*(Fw/Fh)*(Y/X)
    wr = panel_frac*wm - 0.01; hb = panel_frac*hm + 0.01
    padr, padb = 0.01, 0.016
    Lm, Tm     = 0.10, 0.12
    gap_v      = 0.06
    gap_cb, hcb = 0.1, 0.035
    Hg = hm + padb + hb
    axes = []
    for i in range(n):
        y_main = (1.0 - Tm - hm) - i*(Hg + gap_v)
        ax_main  = fig.add_axes([Lm,         y_main,         wm, hm])
        ax_below = fig.add_axes([Lm,         y_main-padb-hb, wm, hb])
        ax_right = fig.add_axes([Lm+wm+padr, y_main,         wr, hm])
        axes.append((ax_main, ax_below, ax_right))
        ax_main.pcolormesh(re, im, Ws[i], cmap=cmap, norm=norm, shading='auto', zorder=0)
        ax_main.axhline(0, color='k', ls='--', lw=1.3, zorder=1)
        ax_main.axvline(0, color='k', ls='--', lw=1.3, zorder=1)
        ax_main.set_xlim(axis_lim[0], axis_lim[1])
        ax_main.set_ylim(*ylim)
        ax_main.yaxis.set_major_locator(ticker.MultipleLocator(2 if labeleven else 1))
        ax_main.tick_params(axis='both', direction='in', length=3.5, top=True, bottom=True, left=True, right=True)
        ax_main.tick_params(axis='x', labelbottom=False, labeltop=(i == 0))
        for spine in ax_main.spines.values():
            spine.set_zorder(5)
        ax_main.xaxis.set_label_position('top')
        if i == 0:
            ax_main.set_xlabel(r'$\mathrm{Re}(\alpha)$', labelpad=7.5)
        ax_main.set_ylabel(r'$\mathrm{Im}(\alpha)$', labelpad=5)
        ax_main.text(0.015, 0.94, f'({chr(97+label_offset+i)})', transform=ax_main.transAxes,
                     ha='left', va='top', bbox=dict(boxstyle='round,pad=0.12', fc='none', ec='none'))
        m = cut_margin * max(np.max(np.abs(cuts_p0[i])), np.max(np.abs(cuts_x0[i])))
        ax_below.plot(re, cuts_p0[i], 'k', lw=1.2)
        ax_below.set_xlim(axis_lim[1], axis_lim[0])
        lo, hi = _lo_hi(cuts_p0[i], m, tol); ax_below.set_ylim(lo, hi)
        ax_below.set_ylabel(r'$W$', rotation=0, labelpad=14, va='center')
        ax_below.tick_params(axis='x', labelbottom=False, direction='in', top=True, bottom=True, size=3.5)
        ax_below.tick_params(axis='y', direction='in', left=True, right=True, size=3.5)
        ax_right.plot(cuts_x0[i], im, 'k', lw=1.2)
        ax_right.set_ylim(*ylim)
        lo, hi = _lo_hi(cuts_x0[i], m, tol); ax_right.set_xlim(hi, lo)
        ax_right.set_xlabel(r'$W$')
        ax_right.tick_params(axis='y', labelleft=False, direction='in', left=True, right=True, size=3.5, rotation=45)
        ax_right.tick_params(axis='x', direction='in', top=True, bottom=True, size=3.5)
    y_last_below = (1.0 - Tm - hm) - (n-1)*(Hg + gap_v) - padb - hb
    y_cb = y_last_below - gap_cb - hcb
    cax = fig.add_axes([Lm, y_cb, wm, hcb])
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    fig.colorbar(sm, cax=cax, orientation='horizontal')
    cax.xaxis.set_ticks_position('bottom'); cax.xaxis.set_label_position('bottom')
    cax.text(1.035, 0.5, r'$W$', transform=cax.transAxes, ha='left', va='center')
    if suptitle:
        fig.suptitle(suptitle, y=1.0)
    if savepath:
        fig.savefig(savepath, bbox_inches='tight', transparent=False, dpi=300)
    return fig, axes


def plot_joint_wigner_2x2(W_re, W_im, uvec, vvec, theta, axis_lim=(-3.5, 3.5), cmap='RdBu',
                          figsize=(9.0, 5.2), n_pts=401, share_scale=True, cut_margin=1.15,
                          show_cut_line=True, cut_size="45%"):
    u_cut = np.linspace(axis_lim[0], axis_lim[1], n_pts)
    v_cut = u_cut*np.tan(theta)

    def _cut(W):
        itp = RegularGridInterpolator((vvec, uvec), W, bounds_error=False, fill_value=np.nan)
        return itp(np.column_stack([v_cut, u_cut]))

    cut_re = _cut(W_re)
    cut_im = _cut(W_im)
    if share_scale:
        vmax = max(np.nanmax(np.abs(W_re)), np.nanmax(np.abs(W_im)))
        vmax_re = vmax_im = vmax
    else:
        vmax_re = np.nanmax(np.abs(W_re)); vmax_im = np.nanmax(np.abs(W_im))

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.5)

    def _block(cell, W, cut, vmax, xlabel, ylabel, show_ylabel_W):
        ax = fig.add_subplot(cell)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        h = ax.pcolormesh(uvec, vvec, W, cmap=cmap, norm=norm, shading='auto')
        if show_cut_line:
            ax.plot(u_cut, v_cut, color='k', ls=':', lw=1.4)
        ax.set_xlim(*axis_lim); ax.set_ylim(*axis_lim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_axisbelow(False)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', labelbottom=False, direction='in', bottom=True, top=True, size=5, width=1.5)
        ax.tick_params(axis='y', direction='in', left=True, right=True, size=5, width=1.5)
        div    = make_axes_locatable(ax)
        ax_cut = div.append_axes("bottom", size=cut_size, pad=0.06, sharex=ax)
        ax_cut.plot(u_cut, cut, color='k', lw=1.4)
        m = cut_margin*np.nanmax(np.abs(cut))
        ax_cut.set_ylim(-m, m)
        tv = _nice_tick(m)
        ax_cut.set_yticks([-tv, 0.0, tv])
        ax_cut.set_yticklabels(_fmt_ticks([-tv, 0.0, tv]))
        ax_cut.set_xlabel(xlabel, labelpad=5)
        if show_ylabel_W:
            ax_cut.set_ylabel(r'$W$', rotation=0, labelpad=10, va='center')
        else:
            ax_cut.tick_params(axis='y', labelleft=False)
        ax_cut.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True, size=5, width=1.5)
        return ax, ax_cut, h

    ax_a, ax_c, h_re = _block(gs[0, 0], W_re, cut_re, vmax_re, r'$\mathrm{Re}(\alpha_1)$', r'$\mathrm{Re}(\alpha_2)$', True)
    ax_b, ax_d, h_im = _block(gs[0, 1], W_im, cut_im, vmax_im, r'$\mathrm{Im}(\alpha_1)$', r'$\mathrm{Im}(\alpha_2)$', False)
    if share_scale:
        cax = ax_b.inset_axes([1.10, 0.0, 0.075, 0.8])
        fig.colorbar(h_im, cax=cax)
        cax.set_title(r'$W$', pad=10)
        cax.tick_params(direction='in')
    else:
        for axm, h in [(ax_a, h_re), (ax_b, h_im)]:
            cax = axm.inset_axes([1.10, 0.0, 0.075, 0.8])
            fig.colorbar(h, cax=cax)
            cax.set_title(r'$W$', pad=10)
            cax.tick_params(direction='in')
    return fig, (ax_a, ax_b, ax_c, ax_d)


def _save(fig, name, svg=False):
    fig.savefig(os.path.join(OUT_DIR, name + '.png'), bbox_inches='tight', transparent=False, dpi=300)
    if svg:
        fig.savefig(os.path.join(OUT_DIR, name + '.svg'), bbox_inches='tight', transparent=False)
    plt.close(fig)


def _plot_signed(ax, x, y, sym, ls, color, lw=1.0):
    yy = y[np.isfinite(y)]
    s  = -1.0 if (yy.size and np.mean(yy) < 0) else 1.0
    ax.plot(x, s*y, ls, color=color, lw=lw, label=(rf'$-{sym}$' if s < 0 else rf'${sym}$'))


# ══════════════════════════════════════════════════════════════════
#  FIGURAS
# ══════════════════════════════════════════════════════════════════
C_A = plt.cm.seismic(0.12)    # azul
C_B = plt.cm.seismic(0.88)    # rojo
LS  = {0.5: '-', 1.0: '--', 1.5: ':'}


def fig_params(S, suffix):
    """Parámetros del gato vs Λ (corte θ = π/4)."""
    ylim_by_cv = {0.5: 0.3, 1.0: 0.55, 1.5: 0.7}
    with plt.rc_context({'xtick.direction': 'in', 'ytick.direction': 'in'}):
        for cv in cv_list:
            d = S[('params', cv)]
            Lc = d['Lam']
            fig, ax = plt.subplots(figsize=(6, 5.5))
            ax.plot(Lc,  d['eta']/2, '-',  color=C_B, lw=1.0, label=r'$\eta/2$')
            ax.plot(Lc, -d['r'],     '--', color=C_A, lw=1.0, label=r'$-r$')
            ax.plot(Lc,  d['kap'],   '-.', color='k', lw=1.0, label=r'$\kappa$')
            ax.set_xlim(Lc[0], 5.0)                   # mismo rango en Λ que la infidelidad
            ax.set_ylim(0.0, ylim_by_cv[cv])
            ax.set_xlabel(r'$\Lambda$'); ax.set_ylabel(r'Variational parameters')
            ax.tick_params(axis='both', direction='in', bottom=True, top=True, right=True, left=True)
            ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
                      borderpad=0.6, handlelength=2.2, labelspacing=0.4)
            plt.tight_layout()
            _save(fig, f'cut_cat_params_rho{RHO}_cv{cv}{suffix}')


def fig_infid(S, suffix):
    C_PLUS, C_MINUS = 'black', plt.cm.seismic(0.95)
    Lam_max = 5.0
    with plt.rc_context({'xtick.direction': 'in', 'ytick.direction': 'in'}):
        # ── versión con eje doble ──
        fig, ax_o1 = plt.subplots(figsize=(6.5, 5.5))
        ax_e1 = ax_o1.twinx()
        for cv in cv_list:
            d = S[('twin', cv)]
            ax_o1.plot(d['Lam'], d['infid_p'], LS[cv], color=C_PLUS,  lw=1.4)
            ax_e1.plot(d['Lam'], d['infid_m'], LS[cv], color=C_MINUS, lw=1.4)
        Lc = d['Lam']
        ax_o1.set_xlabel(r'$\Lambda$')
        ax_o1.set_ylabel(r'$1-|\langle\mathrm{Cat}_+^{\mathrm{exact}}|\mathrm{Cat}_+^{\mathrm{var}}\rangle|^2$', color=C_PLUS)
        ax_o1.tick_params(axis='y', colors=C_PLUS, direction='in')
        ax_o1.tick_params(axis='x', direction='in')
        ax_o1.set_xlim(Lc[0], Lam_max)
        ax_e1.set_ylabel(r'$1-|\langle\mathrm{Cat}_-^{\mathrm{exact}}|\mathrm{Cat}_-^{\mathrm{var}}\rangle|^2$', color=C_MINUS)
        ax_e1.tick_params(axis='y', colors=C_MINUS, direction='in')
        ax_e1.spines['right'].set_color(C_MINUS)
        ax_e1.spines['left'].set_color(C_PLUS)
        ax_o1.set_xticks(np.arange(0, np.floor(Lc[-1]) + 1, 1))
        handles  = [mlines.Line2D([], [], color='gray', ls=LS[cv], lw=1.4,
                                  label=rf'$J_\alpha/\omega_0={cv}$') for cv in cv_list]
        handles += [mlines.Line2D([], [], color=C_PLUS,  lw=1.4, label=r'$|\mathrm{Cat}_+\rangle$'),
                    mlines.Line2D([], [], color=C_MINUS, lw=1.4, label=r'$|\mathrm{Cat}_-\rangle$')]
        ax_o1.legend(handles=handles, loc='upper left', ncol=1, frameon=True,
                     edgecolor='black', framealpha=0.0, fancybox=False,
                     borderpad=0.6, handlelength=2.2, labelspacing=0.4)
        plt.tight_layout()
        _save(fig, f'cat_infid_1D_twin_rho{RHO}{suffix}')

        # ── versión con un solo eje ──
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        for cv in cv_list:
            d = S[('single', cv)]
            ax.plot(d['Lam'], d['infid_p'], LS[cv], color=C_PLUS,  lw=1.4)
            ax.plot(d['Lam'], d['infid_m'], LS[cv], color=C_MINUS, lw=1.4)
        Lc = d['Lam']
        ax.set_xlabel(r'$\Lambda$')
        ax.set_ylabel(r'$1-\mathcal{F}$')
        ax.tick_params(axis='both', direction='in', bottom=True, top=True, right=True, left=True)
        ax.set_xlim(Lc[0], Lam_max)
        ax.set_xticks(np.arange(0, np.floor(Lc[-1]) + 1, 1))
        ax.legend(handles=handles, loc='upper left', ncol=1, frameon=True,
                  edgecolor='black', framealpha=0.0, fancybox=False,
                  borderpad=0.6, handlelength=2.2, labelspacing=0.4)
        plt.tight_layout()
        _save(fig, f'cat_infid_1D_rho{RHO}{suffix}')


def _cut_quantities(d):
    """p±, β̃ (exacto y variacional) a partir de los solapamientos."""
    o_ex, o_var = np.abs(d['o_ex']), np.abs(d['o_var'])
    q = dict(Lc=d['Lam'],
             pp_e=0.5*(1.0 + o_ex), pm_e=0.5*(1.0 - o_ex),
             pp_v=0.5*(1.0 + o_var), pm_v=0.5*(1.0 - o_var),
             th_e=np.sqrt(np.clip(-0.5*np.log(np.clip(o_ex,  1e-300, None)), 0.0, None)),
             th_v=np.sqrt(np.clip(-0.5*np.log(np.clip(o_var, 1e-300, None)), 0.0, None)),
             ov_e=o_ex, ov_v=o_var)
    return q


def fig_prob_theta_overlap(S, suffix):
    with plt.rc_context({'xtick.direction': 'in', 'ytick.direction': 'in'}):
        for cv in cv_list:
            q  = _cut_quantities(S[('cut', cv)])
            Lc = q['Lc']

            # (a) probabilidades
            C_PLUS, C_MINUS = plt.cm.seismic(0.88), plt.cm.seismic(0.12)
            pp_e = _spline_smooth(Lc, _smooth(q['pp_e'], window=15, poly=3), s_factor=1e-4)
            pm_e = _spline_smooth(Lc, _smooth(q['pm_e'], window=15, poly=3), s_factor=1e-4)
            fig, ax = plt.subplots(figsize=(6.0, 4.5))
            ax.plot(Lc, _spline_smooth(Lc, _smooth(q['pp_v'], window=15, poly=3), s_factor=1e-4), '-',  color=C_PLUS,  lw=1.5)
            ax.plot(Lc, pp_e,      '--', color=C_PLUS,  lw=1.5)
            ax.plot(Lc, _spline_smooth(Lc, _smooth(q['pm_v'], window=15, poly=3), s_factor=1e-4), '-',  color=C_MINUS, lw=1.5)
            ax.plot(Lc, pm_e,      '--', color=C_MINUS, lw=1.5)
            ax.set_xlabel(r'$\Lambda$')
            ax.set_ylabel(r'Measurement probability', labelpad=5)
            ax.set_ylim(-0.02, 1.02)
            ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
            ax.set_xlim(Lc[0], 5)
            ax.yaxis.set_major_locator(ticker.FixedLocator([0.0, 0.5, 1.0]))
            ax.text(0.975, 0.85, r'$\dfrac{1}{4\,|\mathcal{N}_+|^{2}}$', transform=ax.transAxes, color=C_PLUS,  ha='right', va='center', fontsize=20)
            ax.text(0.975, 0.15, r'$\dfrac{1}{4\,|\mathcal{N}_-|^{2}}$', transform=ax.transAxes, color=C_MINUS, ha='right', va='center', fontsize=20)
            leg = [mlines.Line2D([], [], color='k', ls='-',  lw=1.5, label='variational'),
                   mlines.Line2D([], [], color='k', ls='--', lw=1.5, label='exact')]
            ax.legend(handles=leg, frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
                      loc='center left', borderpad=0.6, handlelength=2.2, labelspacing=0.4)
            plt.tight_layout()
            _save(fig, f'prob_cat_rho{RHO}_cv{cv}{suffix}')

            # (a') probabilidades + fórmula cerrada de (P+AP)S (sin corte en Fock), figura de control
            d_cf = S[('cut', cv)]
            fig, ax = plt.subplots(figsize=(6.0, 4.5))
            ax.plot(Lc, _spline_smooth(Lc, _smooth(q['pp_v'], window=15, poly=3), s_factor=1e-4), '-',  color=C_PLUS,  lw=1.5)
            ax.plot(Lc, pp_e,      '--', color=C_PLUS,  lw=1.5)
            ax.plot(Lc, d_cf['pp_cf'], ':', color=C_PLUS,  lw=1.8)
            ax.plot(Lc, _spline_smooth(Lc, _smooth(q['pm_v'], window=15, poly=3), s_factor=1e-4), '-',  color=C_MINUS, lw=1.5)
            ax.plot(Lc, pm_e,      '--', color=C_MINUS, lw=1.5)
            ax.plot(Lc, d_cf['pm_cf'], ':', color=C_MINUS, lw=1.8)
            ax.set_xlabel(r'$\Lambda$')
            ax.set_ylabel(r'Measurement probability', labelpad=5)
            ax.set_ylim(-0.02, 1.02)
            ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
            ax.set_xlim(Lc[0], 5)
            ax.yaxis.set_major_locator(ticker.FixedLocator([0.0, 0.5, 1.0]))
            leg = [mlines.Line2D([], [], color='k', ls='-',  lw=1.5, label='variational'),
                   mlines.Line2D([], [], color='k', ls='--', lw=1.5, label='exact'),
                   mlines.Line2D([], [], color='k', ls=':',  lw=1.8, label='closed formula')]
            ax.legend(handles=leg, frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
                      loc='center left', borderpad=0.6, handlelength=2.2, labelspacing=0.4, fontsize=16)
            plt.tight_layout()
            _save(fig, f'prob_cat_rho{RHO}_cv{cv}{suffix}_CHECKFORMULA')

            # (b) β̃
            C_VAR, C_EX = plt.cm.seismic(0.88), plt.cm.seismic(0.12)
            th_e = _spline_smooth(Lc, _smooth(q['th_e'], window=13, poly=3), s_factor=1e-4)
            fig, ax = plt.subplots(figsize=(6, 5.5))
            ax.plot(Lc, _spline_smooth(Lc, _smooth(q['th_v'], window=13, poly=3), s_factor=1e-4), '-',  color=C_VAR, lw=1.5, label='variational')
            ax.plot(Lc, th_e,      '--', color=C_EX,  lw=1.5, label='exact')
            ax.plot(Lc, Lc/2.0, 'k:', lw=1.5, label=r'$\Lambda/2$')
            ax.set_xlabel(r'$\Lambda$'); ax.set_ylabel(r'$\tilde{\beta}$')
            ax.set_xlim(0.0, Lc[-1])
            ax.tick_params(axis='both', direction='in', bottom=True, top=True, right=True, left=True)
            ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
                      borderpad=0.6, handlelength=2.2, labelspacing=0.4)
            plt.tight_layout()
            _save(fig, f'theta_tilde_rho{RHO}_cv{cv}{suffix}')

            # (b') β̃ + fórmula cerrada de (P+AP)S (sin corte en Fock), figura de control
            fig, ax = plt.subplots(figsize=(6, 5.5))
            ax.plot(Lc, _spline_smooth(Lc, _smooth(q['th_v'], window=13, poly=3), s_factor=1e-4), '-',  color=C_VAR, lw=1.5, label='variational')
            ax.plot(Lc, th_e,      '--', color=C_EX,  lw=1.5, label='exact')
            ax.plot(Lc, S[('cut', cv)]['th_cf'], '-.', color=C_VAR, lw=1.5, label='closed formula')
            ax.plot(Lc, Lc/2.0, 'k:', lw=1.5, label=r'$\Lambda/2$')
            ax.set_xlabel(r'$\Lambda$'); ax.set_ylabel(r'$\tilde{\beta}$')
            ax.set_xlim(0.0, Lc[-1])
            ax.tick_params(axis='both', direction='in', bottom=True, top=True, right=True, left=True)
            ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
                      borderpad=0.6, handlelength=2.2, labelspacing=0.4, fontsize=16)
            plt.tight_layout()
            _save(fig, f'theta_tilde_rho{RHO}_cv{cv}{suffix}_CHECKFORMULA')

            # (c) solapamientos
            ov_e = _smooth(q['ov_e'], window=39, poly=3)
            fig, ax = plt.subplots(figsize=(6, 5.5))
            ax.plot(Lc, _smooth(q['ov_v'], window=39, poly=3), '-',  color=C_VAR, lw=1.5, label='variational')
            ax.plot(Lc, ov_e,      '--', color=C_EX,  lw=1.5, label='exact')
            ax.plot(Lc, np.exp(-Lc**2/2), 'k:', lw=1.5, label=r'$e^{-\Lambda^2/2}$')
            ax.set_xlabel(r'$\Lambda$')
            ax.set_ylabel(r'$|\langle\varphi_{b,1}|\hat D(\Lambda)|\varphi_{b,2}\rangle|$')
            ax.set_xlim(Lc[0], Lc[-1]); ax.set_ylim(-0.02, 1.02)
            ax.tick_params(axis='both', direction='in', bottom=True, top=True, right=True, left=True)
            ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
                      borderpad=0.6, handlelength=2.2, labelspacing=0.4)
            plt.tight_layout()
            _save(fig, f'overlap_cat_rho{RHO}_cv{cv}{suffix}')


def fig_negativity(NE, NV, suffix):
    C_PLUS, C_MINUS = plt.cm.seismic(0.88), plt.cm.seismic(0.12)
    with plt.rc_context({'xtick.direction': 'in', 'ytick.direction': 'in'}):
        # exacto
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(NE['Lam'], NE['dp'], '--', color=C_PLUS,  lw=1.5, label=r'$(+)$')
        ax.plot(NE['Lam'], NE['dm'], '--', color=C_MINUS, lw=1.5, label=r'$(-)$')
        ax.set_xlabel(r'$\Lambda$'); ax.set_ylabel(r'$\delta$')
        ax.set_xlim(NE['Lam'][0], NE['Lam'][-1])
        ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        ax.set_xticks(np.arange(0, 7, 1))
        ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False, loc='lower right',
                  borderpad=0.6, handlelength=2.2, labelspacing=0.4)
        plt.tight_layout()
        _save(fig, f'negativity_vs_Lambda{suffix}')

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(NE['Lam'], NE['np_'], '--', color=C_PLUS,  lw=1.5, label=r'$(+)$')
        ax.plot(NE['Lam'], NE['nm'],  '--', color=C_MINUS, lw=1.5, label=r'$(-)$')
        ax.axvline(2.45, color='k', ls=':', lw=1.5)
        ax.set_xlabel(r'$\Lambda$'); ax.set_ylabel(r'$\int W\,\mathrm{d}^2\alpha$')
        ax.set_xlim(NE['Lam'][0], NE['Lam'][-1])
        ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False, loc='best',
                  borderpad=0.6, handlelength=2.2, labelspacing=0.4)
        plt.tight_layout()
        _save(fig, f'diag_norm_check{suffix}')

        # variacional
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(NV['Lam'], NV['dp'], '-', color=C_PLUS,  lw=1.5, label=r'$(+)$')
        ax.plot(NV['Lam'], NV['dm'], '-', color=C_MINUS, lw=1.5, label=r'$(-)$')
        ax.set_xlabel(r'$\Lambda$'); ax.set_ylabel(r'$\delta$')
        ax.set_xlim(NV['Lam'][0], 7.0)
        ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        ax.set_xticks(np.arange(0, 7, 1))
        ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False, loc='lower right',
                  borderpad=0.6, handlelength=2.2, labelspacing=0.4)
        plt.tight_layout()
        _save(fig, f'negativity_variational_vs_Lambda{suffix}')

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(NV['Lam'], NV['np_'], '-', color=C_PLUS,  lw=1.5, label=r'$(+)$')
        ax.plot(NV['Lam'], NV['nm'],  '-', color=C_MINUS, lw=1.5, label=r'$(-)$')
        ax.axhline(1.0, color='k', ls='--', lw=1.0)
        ax.set_xlabel(r'$\Lambda$'); ax.set_ylabel(r'$\int W\,\mathrm{d}^2\alpha$')
        ax.set_xlim(NV['Lam'][0], 7.0)
        ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False, loc='best',
                  borderpad=0.6, handlelength=2.2, labelspacing=0.4)
        plt.tight_layout()
        _save(fig, f'diag_norm_check_variational{suffix}')

        # comparación
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(NE['Lam'], NE['dp'], '--', color=C_PLUS,  lw=1.25)
        ax.plot(NE['Lam'], NE['dm'], '--', color=C_MINUS, lw=1.25)
        ax.plot(NV['Lam'], NV['dp'], '-',  color=C_PLUS,  lw=1.5)
        ax.plot(NV['Lam'], NV['dm'], '-',  color=C_MINUS, lw=1.5)
        ax.axvline(2*np.sqrt(2), color='k', ls=':', lw=1.5)
        ax.set_xlabel(r'$\Lambda$'); ax.set_ylabel(r'$\delta$')
        ax.set_xlim(0.01, 7.0)
        ax.set_xticks(np.arange(0, 7, 1))
        ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        leg = [mlines.Line2D([], [], color=C_PLUS,  lw=1.5, label=r'$(+)$'),
               mlines.Line2D([], [], color=C_MINUS, lw=1.5, label=r'$(-)$')]
        ax.legend(handles=leg, frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
                  loc='lower right', borderpad=0.6, handlelength=2.2, labelspacing=0.4)
        plt.tight_layout()
        _save(fig, f'negativity_exact_vs_variational{suffix}')


def fig_2x2(S, NE, NV, suffix, checkformula=False):
    cv = CV_2X2
    LAM_XMAX = 5.0
    XTICKS   = np.arange(0, np.floor(LAM_XMAX) + 1, 1)
    C_PLUS, C_MINUS = plt.cm.seismic(0.88), plt.cm.seismic(0.12)

    q  = _cut_quantities(S[('cut', cv)])
    Lc = q['Lc']
    th_e = _spline_smooth(Lc, _smooth(q['th_e'], window=13, poly=3), s_factor=1e-4)
    pp_e = _spline_smooth(Lc, _smooth(q['pp_e'], window=15, poly=3), s_factor=1e-4)
    pm_e = _spline_smooth(Lc, _smooth(q['pm_e'], window=15, poly=3), s_factor=1e-4)

    fig = plt.figure(figsize=(10.0, 8.0))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.08, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharex=ax_a)
    ax_c = fig.add_subplot(gs[1, 0], sharex=ax_a)
    ax_d = fig.add_subplot(gs[1, 1], sharex=ax_a)

    for cv_i in cv_list:
        ray = S[('twin', cv_i)]
        ax_a.plot(ray['Lam'], ray['infid_p'], LS[cv_i], color=C_PLUS,  lw=1.4)
        ax_a.plot(ray['Lam'], ray['infid_m'], LS[cv_i], color=C_MINUS, lw=1.4)
    ax_a.set_ylabel(r'$1-F$')
    fmt_a = SciFormatter('%1.1f')
    fmt_a.set_powerlimits((0, 0))
    ax_a.yaxis.set_major_formatter(fmt_a)
    ax_a.yaxis.get_offset_text().set_fontsize(22)
    handles  = [mlines.Line2D([], [], color='gray', ls=LS[cv_i], lw=1.4,
                              label=rf'$J_\alpha/\omega_0={cv_i}$') for cv_i in cv_list]
    handles += [mlines.Line2D([], [], color=C_PLUS,  lw=1.4, label=r'$|\mathrm{Cat}_+\rangle$'),
                mlines.Line2D([], [], color=C_MINUS, lw=1.4, label=r'$|\mathrm{Cat}_-\rangle$')]
    ax_a.legend(handles=handles, loc='upper left', ncol=1, frameon=True, edgecolor='black',
                framealpha=0.0, fancybox=False, borderpad=0.25, handlelength=1.2, labelspacing=0.4, fontsize=16)
    ax_a.text(0.85, 0.95, r'(a)', transform=ax_a.transAxes, ha='left', va='top', fontsize=20)

    ax_b.plot(Lc, _spline_smooth(Lc, _smooth(q['th_v'], window=13, poly=3), s_factor=1e-4), '-',  color='k', lw=1.5)
    ax_b.plot(Lc, th_e,      '--', color='k', lw=1.5)
    ax_b.plot(Lc, Lc/2.0, 'k:', lw=1.5, label=r'$\Lambda/2$')
    if checkformula:                # fórmula cerrada de β̃ de (P+AP)S (sin corte en Fock)
        ax_b.plot(Lc, S[('cut', cv)]['th_cf'], '-.', color='k', lw=1.5, label='closed formula')
    ax_b.set_ylabel(r'$\tilde{\beta}$')
    if checkformula:
        ax_b.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False, fontsize=14,
                    borderpad=0.25, handlelength=1.6, labelspacing=0.3, loc='upper left',
                    bbox_to_anchor=(0.0, 0.86))
    else:
        ax_b.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
                    borderpad=0.25, handlelength=1.2, labelspacing=0.4, loc='lower right')
    ax_b.text(0.05, 0.95, r'(b)', transform=ax_b.transAxes, ha='left', va='top', fontsize=22)

    ax_c.plot(Lc, _spline_smooth(Lc, _smooth(q['pp_v'], window=15, poly=3), s_factor=1e-4), '-',  color=C_PLUS,  lw=1.5)
    ax_c.plot(Lc, pp_e,      '--', color=C_PLUS,  lw=1.5)
    ax_c.plot(Lc, _spline_smooth(Lc, _smooth(q['pm_v'], window=15, poly=3), s_factor=1e-4), '-',  color=C_MINUS, lw=1.5)
    ax_c.plot(Lc, pm_e,      '--', color=C_MINUS, lw=1.5)
    ax_c.set_xlabel(r'$\Lambda$')
    ax_c.set_ylabel(r'$p_{\pm}$', labelpad=5)
    ax_c.set_ylim(-0.02, 1.02)
    ax_c.yaxis.set_major_locator(ticker.FixedLocator([0.0, 0.5, 1.0]))
    ax_c.text(0.975, 0.8, r'$\dfrac{1}{4\,|\mathcal{N}_+|^{2}}$', transform=ax_c.transAxes,
              color=C_PLUS,  ha='right', va='center', fontsize=22)
    ax_c.text(0.975, 0.2, r'$\dfrac{1}{4\,|\mathcal{N}_-|^{2}}$', transform=ax_c.transAxes,
              color=C_MINUS, ha='right', va='center', fontsize=22)
    ax_c.text(0.05, 0.925, r'(c)', transform=ax_c.transAxes, ha='left', va='top', fontsize=22)
    if checkformula:                # fórmula cerrada de (P+AP)S (sin corte en Fock)
        d_cf = S[('cut', cv)]
        ax_c.plot(Lc, d_cf['pp_cf'], ':', color=C_PLUS,  lw=1.8)
        ax_c.plot(Lc, d_cf['pm_cf'], ':', color=C_MINUS, lw=1.8)
        leg_c = [mlines.Line2D([], [], color='k', ls='-',  lw=1.5, label='variational'),
                 mlines.Line2D([], [], color='k', ls='--', lw=1.5, label='exact'),
                 mlines.Line2D([], [], color='k', ls=':',  lw=1.8, label='closed formula')]
        ax_c.legend(handles=leg_c, frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
                    loc='center left', borderpad=0.25, handlelength=1.6, labelspacing=0.3, fontsize=14)

    ax_d.plot(NE['Lam'], _smooth(NE['dp'], window=13, poly=3), '--', color=C_PLUS,  lw=1.25)
    ax_d.plot(NE['Lam'], _smooth(NE['dm'], window=13, poly=3), '--', color=C_MINUS, lw=1.25)
    ax_d.plot(NV['Lam'], NV['dp'], '-', color=C_PLUS,  lw=1.5)
    ax_d.plot(NV['Lam'], NV['dm'], '-', color=C_MINUS, lw=1.5)
    ax_d.set_xlabel(r'$\Lambda$')
    ax_d.set_ylabel(r'$\delta_W$')
    leg = [mlines.Line2D([], [], color=C_PLUS,  lw=1.5, label=r'$|\mathrm{Cat}_{+}\rangle$'),
           mlines.Line2D([], [], color=C_MINUS, lw=1.5, label=r'$|\mathrm{Cat}_{-}\rangle$')]
    ax_d.legend(handles=leg, frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
                loc='lower right', borderpad=0.25, handlelength=1.2, labelspacing=0.4, fontsize=16)
    ax_d.text(0.05, 0.95, r'(d)', transform=ax_d.transAxes, ha='left', va='top', fontsize=22)

    ax_a.set_xlim(0.0, LAM_XMAX)
    ax_a.set_xticks(XTICKS)
    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.tick_params(axis='both', direction='in', bottom=True, top=True,
                       left=True, right=True, size=5, width=1.5)
    for ax in (ax_a, ax_b):
        ax.tick_params(axis='x', labelbottom=False)
    _save(fig, f'infid_beta_prob_neg_2x2_rho{RHO}_cv{cv}{suffix}' + ('_CHECKFORMULA' if checkformula else ''), svg=True)

def fig_wigner_stacks(suffix):
    g = _load_named(os.path.join(DATA_DIR, 'wigner_grid.txt'))
    xvec, pvec = g['x'], g['p']
    for lam_val, figsize in WIG_POINTS:
        W_ex = np.loadtxt(os.path.join(DATA_DIR, f'wigner_exact_lam{lam_val}.txt'))
        W_v  = np.loadtxt(os.path.join(DATA_DIR, f'wigner_var_lam{lam_val}.txt'))
        cp_ex, cx_ex = _cuts(W_ex, xvec, pvec)
        cp_v,  cx_v  = _cuts(W_v,  xvec, pvec)
        fig, _ = plot_wigner_stack_num([W_ex, W_v], [cp_ex, cp_v], [cx_ex, cx_v],
                                       xvec=xvec, pvec=pvec, axis_lim=(-4.2, 4.2),
                                       labeleven=True, figsize=figsize)
        _save(fig, f'wigner_cat_exact_vs_var_lam{lam_val}{suffix}')


def fig_joint_wigner(suffix):
    uv   = np.loadtxt(os.path.join(DATA_DIR, 'wigner_joint_grid.txt'))
    W_re = np.loadtxt(os.path.join(DATA_DIR, f'wigner_joint_{joint_tag()}_ReRe.txt'))
    W_im = np.loadtxt(os.path.join(DATA_DIR, f'wigner_joint_{joint_tag()}_ImIm.txt'))
    fig, _ = plot_joint_wigner_2x2(W_re, W_im, uv, uv, THETA_JOINT,
                                   axis_lim=(-2.5, 2.5), figsize=(7.0, 7.0), share_scale=True)
    _save(fig, f'wigner_joint_{joint_tag()}{suffix}', svg=True)


# ══════════════════════════════════════════════════════════════════
#  EJECUCIÓN
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    setup_style()
    suffix = '_PAPS'

    S  = load_sweeps()
    NE = load_negativity('exact')
    NV = load_negativity('variational')

    fig_params(S, suffix)
    fig_infid(S, suffix)
    fig_prob_theta_overlap(S, suffix)
    fig_negativity(NE, NV, suffix)
    fig_2x2(S, NE, NV, suffix)
    fig_2x2(S, NE, NV, suffix, checkformula=True)
    fig_wigner_stacks(suffix)
    fig_joint_wigner(suffix)
    print(f'\nFiguras guardadas en: {OUT_DIR}')
