# -*- coding: utf-8 -*-
"""
DATOS de la nueva figura 4 del paper: funciones de Wigner de los gatos |Cat_∓> del modo colectivo
en resonancia (δ_α = 0, J_α/ω0 = 1, θ = π/4), con el ansatz (P+AP)S y con el exacto.
La figura la hace plot_wigner_cats.py.

    (a) |Cat_->  en Λ/√2 = 2      (b) |Cat_+>  en Λ/√2 = 3
        mapa variacional W(α), α = Re α + i Im α, y cortes W(Re α, 0) y W(0, Im α):
        variacional (malla fina) y exacto (malla gruesa, círculos en la figura)
    (c) δ_W(Λ/√2) de ambos gatos: se reutilizan los datos de variational_P_AP_cat.py
        (Figures/cat/data/cat_PAPS/negativity_{exact,variational}_cv1.0.txt); aquí no se recalcula.

Mismo esquema que variational_P_AP_cat.py (se importan sus funciones):
    variacional: minimize_cat + variational_cat_vector (marco centrado,
                 D(Λ/2)|φ↓> ± D(−Λ/2)|φ↑>)
    exacto:      E_cat_exact + exact_cat_state (misma convención de signo)
    Wigner:      wigner_pure_padded, convención W(α) = (2/π) Σ_n (−1)^n |<n|D(−α)|ψ>|²
                 (W ∈ [−2/π, 2/π], la misma de δ_W)
Con sign = −1 el gato es impar (W(0) = −2/π, δ_W -> 0.213 = δ_W(|1>) cuando Λ -> 0);
con sign = +1, par (W(0) = +2/π).

Uso:  python wigner_cats.py
Los .txt se guardan en Figures/paper_fig4/data/ y no se recalculan si ya existen.
"""

import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from variational_P_AP_cat import (minimize_cat, variational_cat_vector, E_cat_exact,
                                  exact_cat_state, wigner_pure_padded, _Mof_Lam)

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════
#  PARÁMETROS
# ══════════════════════════════════════════════════════════════════
CV     = 1.0                                     # J_α/ω0
PANELS = [(2.0, -1), (3.0, +1)]                  # (Λ/√2, sign): (a) |Cat_->, (b) |Cat_+>

X_MAP  = np.linspace(-4.5, 4.5, 451)             # Re α (mapa y corte variacional; incluye 0); x = √2 Re α ∈ [−6.36, 6.36]
P_MAP  = np.linspace(-2.0, 2.0, 201)             # Im α (incluye 0)
X_EX   = np.linspace(-4.5, 4.5, 41)              # cortes exactos (círculos; incluyen 0)
P_EX   = np.linspace(-2.0, 2.0, 25)

DATA_DIR = os.path.join(HERE, 'Figures', 'paper_fig4', 'data')
NEG_DIR  = os.path.join(HERE, 'Figures', 'cat', 'data', 'cat_PAPS')   # δ_W ya calculado


def _tag(L, sign):
    return f'L{L:g}_{"plus" if sign > 0 else "minus"}_cv{CV:g}'


def f_map(L, sign):
    return os.path.join(DATA_DIR, f'wigner_var_map_{_tag(L, sign)}_x{len(X_MAP)}_p{len(P_MAP)}.txt')


def f_cuts(kind, L, sign):
    """kind = 'var' o 'exact'; columnas: coordenada, W."""
    n = (len(X_MAP), len(P_MAP)) if kind == 'var' else (len(X_EX), len(P_EX))
    return (os.path.join(DATA_DIR, f'wigner_{kind}_cutRe_{_tag(L, sign)}_n{n[0]}.txt'),
            os.path.join(DATA_DIR, f'wigner_{kind}_cutIm_{_tag(L, sign)}_n{n[1]}.txt'))


def f_neg(kind):
    return os.path.join(NEG_DIR, f'negativity_{kind}_cv{CV}.txt')


def compute_panel(L, sign):
    files = [f_map(L, sign), *f_cuts('var', L, sign), *f_cuts('exact', L, sign)]
    if all(os.path.exists(f) for f in files):
        print(f'Λ/√2 = {L:g}, sign = {sign:+d}: ya existen, se omite')
        return
    Lam = np.sqrt(2.0)*L
    M   = _Mof_Lam(Lam)
    # variacional
    _, x  = minimize_cat(Lam, CV)
    psi_v = variational_cat_vector(Lam, x, sign, M)
    W_v   = wigner_pure_padded(psi_v, X_MAP, P_MAP)              # filas = Im α, columnas = Re α
    cRe_v = W_v[np.argmin(np.abs(P_MAP)), :]
    cIm_v = W_v[:, np.argmin(np.abs(X_MAP))]
    # exacto (solo los cortes)
    _, vec, _ = E_cat_exact(L, L, CV, M=M)
    psi_e = exact_cat_state(L, L, vec, M, sign=sign)
    cRe_e = wigner_pure_padded(psi_e, X_EX, np.array([0.0]))[0, :]
    cIm_e = wigner_pure_padded(psi_e, np.array([0.0]), P_EX)[:, 0]

    os.makedirs(DATA_DIR, exist_ok=True)
    hdr = (f'Lambda/sqrt2 = {L}, sign = {sign:+d}, J/omega0 = {CV}, theta = pi/4; '
           f'W(alpha) = (2/pi) sum_n (-1)^n |<n|D(-alpha)|psi>|^2')
    np.savetxt(f_map(L, sign), W_v, fmt='%.10e',
               header=(f'W variacional (P+AP)S, filas = Im alpha ({len(P_MAP)}, {P_MAP[0]}..{P_MAP[-1]}), '
                       f'columnas = Re alpha ({len(X_MAP)}, {X_MAP[0]}..{X_MAP[-1]}) | {hdr} | '
                       f'eta = {x[0]:.10e}, r = {x[1]:.10e}, kappa = {x[2]:.10e}'))
    for (fRe, fIm), (cRe, cIm), xs, ps, kind in (
            (f_cuts('var', L, sign), (cRe_v, cIm_v), X_MAP, P_MAP, 'variacional'),
            (f_cuts('exact', L, sign), (cRe_e, cIm_e), X_EX, P_EX, 'exacto')):
        np.savetxt(fRe, np.column_stack([xs, cRe]), fmt='%.10e',
                   header=f'corte W(Re alpha, 0), {kind} | {hdr}\nRe_alpha  W')
        np.savetxt(fIm, np.column_stack([ps, cIm]), fmt='%.10e',
                   header=f'corte W(0, Im alpha), {kind} | {hdr}\nIm_alpha  W')
    # control: el corte variacional interpolado en los puntos exactos
    dRe = np.max(np.abs(np.interp(X_EX, X_MAP, cRe_v) - cRe_e))
    dIm = np.max(np.abs(np.interp(P_EX, P_MAP, cIm_v) - cIm_e))
    fid = abs(np.vdot(psi_v, psi_e))**2
    print(f'Λ/√2 = {L:g}, sign = {sign:+d}: (η, r, κ) = ({x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f}), '
          f'F = {fid:.6f}, W(0) = {W_v[np.argmin(np.abs(P_MAP)), np.argmin(np.abs(X_MAP))]:+.4f}, '
          f'max|ΔW| cortes: Re {dRe:.2e}, Im {dIm:.2e}')


if __name__ == '__main__':
    for L, sign in PANELS:
        compute_panel(L, sign)
    for kind in ('exact', 'variational'):
        print(f'δ_W ({kind}): {"ok" if os.path.exists(f_neg(kind)) else "FALTA"} {f_neg(kind)}')
    print(f'\nDatos en: {DATA_DIR}')
