# -*- coding: utf-8 -*-
"""
Ansatz variacional polarón + antipolarón (P+AP)S para la cadena de Kitaev de dos sitios
con cavidades locales (base de Lang-Firsov, modos colectivos).

Convención de pseudoespín (notas):
    sector impar:  |↑_o> = |10>,  |↓_o> = |01>
    sector par:    |↑_e> = |11>,  |↓_e> = |00>
En ambos sectores, en la base {|↑>, |↓>}:
    H_α = [[ ε  + ω0 b†b ,  J D(Λ)        ],
           [ J D(Λ)†     ,  ε' + ω0 b†b   ]]
    impar: ε = ε̃1,        ε' = ε̃2,  J = t
    par:   ε = ε̃1 + ε̃2,   ε' = 0,   J = Δ

Ansatz por rama (|r, η> = D(η) S(r)|0>):
    |φ↑> = N↑ [ |r↑, η↑> + κ↑ |r↓, η↓ + Λ> ]
    |φ↓> = N↓ [ |r↓, η↓> + κ↓ |r↑, η↑ − Λ> ]
    D(η) = exp[η (b† − b)],  S(r) = exp[r (b² − b†²)/2]
Vector de parámetros: x = (η↑, η↓, r↑, r↓, κ↑, κ↓).

La minimización usa únicamente el funcional de energía cerrado (todos los elementos de Fock):
    E = (A↑ + A↓)/2 − sqrt[ ((A↑ − A↓)/2)² + T² ],
    A↑ = ε  + ω0 [η↑² + sinh²r↑ + κ↑²((η↓+Λ)² + sinh²r↓) + 2κ↑ G↑] / (1 + κ↑² + 2κ↑ F),
    A↓ = ε' + ω0 [η↓² + sinh²r↓ + κ↓²((η↑−Λ)² + sinh²r↑) + 2κ↓ G↓] / (1 + κ↓² + 2κ↓ F),
    T  = J [κ↑ + κ↓ + (1 + κ↑κ↓) F] / sqrt[(1 + κ↑² + 2κ↑F)(1 + κ↓² + 2κ↓F)],
con F y G↑, G↓ cerrados (ver _F_G). En δ_α = 0 (ε = ε') se usa la reducción por paridad de
Rabi η↑ = −η↓ = η/2, r↑ = r↓ = r, κ↑ = κ↓ = κ y el funcional reducido E(η, r, κ).
El vector en Fock solo se construye con los parámetros óptimos (fidelidad, pesos, tests).

Después de L-BFGS-B (si JAX está instalado; si no, se omiten):
  * pulido de Newton amortiguado con ∇E y H exactos (JAX) -> mínimo a precisión de máquina;
  * continuación de variación mínima: entre los x con E(x) ≤ E* + MV_DELTA_REL·|E*|, el más
    cercano al punto anterior del barrido en Λ. Solo actúa en los valles casi planos
    (p. ej. κ↓, η↓ en el sector par), donde la energía no fija los parámetros.

Este script calcula y guarda los datos (.txt) en Figures/anstaz/data/; las figuras (parámetros
óptimos y fidelidad / error relativo por sector, sufijo _PAPS, en Figures/anstaz/) las hace
plot_variational_P_AP.py.
"""

import os
import warnings

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.special import factorial, eval_genlaguerre
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

# Derivadas exactas (JAX) para el pulido de Newton y la continuación de variación mínima.
# Si JAX no está instalado, el script corre igual que antes (sin pulido ni variación mínima).
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
try:
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    HAS_JAX = True
except Exception:
    HAS_JAX = False

#  PARÁMETROS

N_JOBS   = -1                  # núcleos para joblib (-1 = todos)

POLISH        = True           # pulido de Newton amortiguado (∇E y H exactos) tras L-BFGS-B
MIN_VARIATION = True           # continuación de variación mínima en Λ (valles planos)
MV_DELTA_REL  = 1e-12          # holgura en energía de la variación mínima: E ≤ E* + MV_DELTA_REL·|E*|

omega0   = 1.0
eps1_val = 0.25
eps2_val = 0.25
cv_list  = [0.5, 1.0, 1.5]     # J_α/ω0  (t = Δ = cv en las figuras)
KAP_MAX  = 1.0                 # κ ∈ [0, KAP_MAX] (redundancia κ -> 1/κ: basta κ <= 1)

# Figuras: eje Λ = √(λ₁² + λ₂²), δλ = 0
LAM_F2   = np.linspace(0.025, 4.0*np.sqrt(2), 400)

OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Figures', 'anstaz')
DATA_DIR = os.path.join(OUT_DIR, 'data')           # datos que lee plot_variational_P_AP.py

#  FOCK: desplazamiento exacto (para el hamiltoniano exacto)
def disp_matrix(M_ph, alpha):
    """Elementos de matriz <m'|D(alpha)|m> en base de Fock, alpha real."""
    N      = M_ph + 1
    alpha2 = alpha * alpha
    gauss  = np.exp(-0.5 * alpha2)
    facts  = np.array([float(factorial(k, exact=True)) for k in range(N)])
    D = np.zeros((N, N))
    for mp in range(N):
        for n in range(N):
            if mp >= n:
                k    = mp - n
                pref = gauss * (alpha**k) * np.sqrt(facts[n] / facts[mp])
                D[mp, n] = pref * eval_genlaguerre(n, k, alpha2)
            else:
                k    = n - mp
                pref = gauss * ((-alpha)**k) * np.sqrt(facts[mp] / facts[n])
                D[mp, n] = pref * eval_genlaguerre(mp, k, alpha2)
    return D


def _Mof_Lam(Lam):
    return int(min(170, max(50, np.ceil(4.0*Lam*Lam) + 25)))


def Lam_to_lam12(Lam, dlam):
    """λ₁, λ₂ con λ₁² + λ₂² = Λ² y λ₁ − λ₂ = δλ. Devuelve (nan, nan) si no existe."""
    disc = 2.0*Lam**2 - dlam**2
    if disc < 0:
        return np.nan, np.nan
    lam1 = 0.5*(dlam + np.sqrt(disc))
    return lam1, lam1 - dlam


#  FOCK: estados del ansatz (solo con los parámetros óptimos)
class FockOps:
    """Generadores reales antisimétricos en Fock truncado (dim = M+1):
        G1 = b† − b             ->  D(η) = exp(η G1)
        G2 = (b² − b†²)/2       ->  S(r) = exp(r G2)
    exp(x G) se evalúa exactamente con la diagonalización de H = iG (hermítica):
        exp(x G) = V exp(−i x w) V†.
    """

    def __init__(self, M):
        self.M    = M
        self.dim  = M + 1
        self.nvec = np.arange(self.dim, dtype=float)
        b  = np.diag(np.sqrt(np.arange(1, self.dim, dtype=float)), 1)
        bd = b.T
        self.b = b
        gens = {1: bd - b,
                2: 0.5*(b @ b - bd @ bd)}
        self.eig = {}
        for k, G in gens.items():
            w, V = eigh(1j*G)
            self.eig[k] = (w, V, V.conj().T)

    def apply(self, k, x, v):
        """exp(x G_k) v."""
        if x == 0.0:
            return v
        w, V, Vh = self.eig[k]
        return (V @ (np.exp(-1j*x*w) * (Vh @ v))).real

    def state(self, eta, r):
        """|r, η> = D(η) S(r) |0>."""
        v = np.zeros(self.dim)
        v[0] = 1.0
        v = self.apply(2, r, v)
        v = self.apply(1, eta, v)
        return v


_OPS_CACHE = {}


def get_ops(M):
    if M not in _OPS_CACHE:
        _OPS_CACHE[M] = FockOps(M)
    return _OPS_CACHE[M]


#  SECTORES 
def sector_params(sector, lam1, lam2, eps1, eps2, t, Delta):
    """(ε, ε', J) del bloque en la base {|↑>, |↓>}."""
    e1t = eps1 - lam1**2 * omega0
    e2t = eps2 - lam2**2 * omega0
    if sector == 'odd':        # |↑_o> = |10>, |↓_o> = |01>
        return e1t, e2t, t
    else:                      # |↑_e> = |11>, |↓_e> = |00>
        return e1t + e2t, 0.0, Delta


#  FUNCIONAL DE ENERGÍA 
def _F_G(x, Lam):
    """F = <r↑,η↑|D(Λ)|r↓,η↓>, G↑ = <r↑,η↑|b†b|r↓,η↓+Λ>, G↓ = <r↓,η↓|b†b|r↑,η↑−Λ>."""
    eu, ed, ru, rd = x[0], x[1], x[2], x[3]
    Dl = Lam + ed - eu                                   # Λ + η↓ − η↑
    ch = np.cosh(ru - rd)
    F  = np.exp(-Dl**2/(np.exp(-2.0*ru) + np.exp(-2.0*rd)))/np.sqrt(ch)
    su, sd = np.sinh(ru), np.sinh(rd)
    au = Dl*np.exp(rd)*su/ch                             # Δ e^{r↓} sinh r↑ / cosh(r↑ − r↓)
    ad = Dl*np.exp(ru)*sd/ch                             # Δ e^{r↑} sinh r↓ / cosh(r↑ − r↓)
    Gu = F*((eu - au)*(ed + Lam + ad) + su*sd/ch)
    Gd = F*((ed + ad)*(eu - Lam - au) + su*sd/ch)
    return F, Gu, Gd


def closed_ABC(x, eps, epsp, J, Lam):
    """A↑, A↓, T del funcional  (x = (η↑, η↓, r↑, r↓, κ↑, κ↓))."""
    eu, ed, ru, rd, ku, kd = x[:6]
    F, Gu, Gd = _F_G(x, Lam)
    nu = 1.0 + ku*ku + 2.0*ku*F                          # N↑^{-2}
    nd = 1.0 + kd*kd + 2.0*kd*F                          # N↓^{-2}
    if nu < 1e-12 or nd < 1e-12:                         # |φ_s> = 0: no es un estado
        return np.nan, np.nan, np.nan
    A = eps  + omega0*(eu*eu + np.sinh(ru)**2 + ku*ku*((ed + Lam)**2 + np.sinh(rd)**2)
                       + 2.0*ku*Gu)/nu
    B = epsp + omega0*(ed*ed + np.sinh(rd)**2 + kd*kd*((eu - Lam)**2 + np.sinh(ru)**2)
                       + 2.0*kd*Gd)/nd
    T = J*(ku + kd + (1.0 + ku*kd)*F)/np.sqrt(nu*nd)
    return A, B, T


def energy_closed(x, eps, epsp, J, Lam):
    """Funcional general."""
    A, B, T = closed_ABC(x, eps, epsp, J, Lam)
    if not np.isfinite(A):
        return 1e6
    return 0.5*(A + B) - np.sqrt(0.25*(A - B)**2 + T*T)


def energy_closed_sym(xs, eps, epsp, J, Lam):
    """Funcional reducido en δ_α = 0, E(η, r, κ),
    E = ε + {ω0[η_n²/4 + (1+κ²) sinh²r + κ²(Λ − η_n/2)² + 2κF(Λ²/4 − (Λ−η_n)² e^{4r}/4 + sinh²r)]
             − J |2κ + (1+κ²)F|} / (1 + κ² + 2κF),   F = exp[−(Λ−η_n)² e^{2r}/2]."""
    eta_n, r, kap = 2.0*xs[0], xs[1], xs[2]
    F   = np.exp(-0.5*(Lam - eta_n)**2*np.exp(2.0*r))
    sh2 = np.sinh(r)**2
    den = 1.0 + kap**2 + 2.0*kap*F
    if den < 1e-12:                                      # κ = −1 con F = 1: |φ↑> = 0
        return 1e6
    num = (omega0*(0.25*eta_n**2 + (1.0 + kap**2)*sh2 + kap**2*(Lam - 0.5*eta_n)**2
                   + 2.0*kap*F*(0.25*Lam**2 - 0.25*(Lam - eta_n)**2*np.exp(4.0*r) + sh2))
           - abs(J)*abs(2.0*kap + (1.0 + kap**2)*F))
    return eps + num/den


def var_states(x, Lam, ops):
    """|φ↑>, |φ↓> normalizados en Fock (solo con los parámetros óptimos)."""
    eu, ed, ru, rd, ku, kd = x[:6]
    pu = ops.state(eu, ru) + ku*ops.state(ed + Lam, rd)
    pd = ops.state(ed, rd) + kd*ops.state(eu - Lam, ru)
    return pu/np.linalg.norm(pu), pd/np.linalg.norm(pd)


def energy_fock(x, eps, epsp, J, Lam, ops):
    """El mismo funcional evaluado en Fock (solo para el test T2)."""
    pu, pd = var_states(x, Lam, ops)
    A = eps  + omega0*(pu @ (ops.nvec*pu))
    B = epsp + omega0*(pd @ (ops.nvec*pd))
    T = J*(pu @ ops.apply(1, Lam, pd))
    return 0.5*(A + B) - np.sqrt(0.25*(A - B)**2 + T*T)


def weights_2x2(A, B, C):
    """(c↑, c↓) del menor autovalor de [[A, C], [C, B]].
    Usa (C, E−A) o (E−B, C), la de mayor norma (evita 0/0)."""
    E  = 0.5*(A + B) - np.sqrt(0.25*(A - B)**2 + C*C)
    v1 = np.array([C, E - A])
    v2 = np.array([E - B, C])
    v  = v1 if np.hypot(*v1) >= np.hypot(*v2) else v2
    return v / np.hypot(*v)


#  MINIMIZACIÓN
KAP_SEEDS = (0.15, 0.5, 0.0)


def _seeds(Lam, n_par, sector, warm=False):

    h = 0.5*Lam
    old = [(0, 0, 0, 0), (-h, h, -0.1, -0.1), (h, -h, -0.1, -0.1), (-h, h, 0, 0),
           (0, 0, -0.1, -0.1), (-Lam, 0, -0.1, -0.1), (0, Lam, -0.1, -0.1),
           (h, -h, 0, 0), (h, -h, -0.2, -0.2)]
    if sector == 'even':
        old = [(b, a, rb, ra) for (a, b, ra, rb) in old]
    kaps = KAP_SEEDS[:2] if warm else KAP_SEEDS
    s4 = [tuple(float(v) for v in sd) for sd in old]
    if n_par == 6:                                   # general
        return [sd + (k, k) for sd in s4 for k in kaps]
    s2 = list(dict.fromkeys((0.5*(a - b), 0.5*(ra + rb)) for (a, b, ra, rb) in s4))
    return [sd + (k,) for sd in s2 for k in kaps]   # simétrico


def _bounds(Lam, n_par):
    be, br = (-Lam - 1.0, Lam + 1.0), (-1.5, 1.5)
    bk = (0.0, KAP_MAX)
    return {3: [be, br, bk], 6: [be, be, br, br, bk, bk]}[n_par]


def _run_seeds(fun, args, Lam, n_par, x_warm, sector):
    seeds = _seeds(Lam, n_par, sector, warm=x_warm is not None)
    if x_warm is not None:
        seeds = [tuple(x_warm)] + seeds
    bnd = _bounds(Lam, n_par)
    lo = np.array([b[0] for b in bnd])
    hi = np.array([b[1] for b in bnd])
    bestE, bestx = np.inf, None
    for sd in seeds:
        x0  = np.clip(np.asarray(sd, float), lo, hi)
        res = minimize(fun, x0, args=args, method='L-BFGS-B', bounds=bnd,
                       options={'ftol': 1e-15, 'gtol': 1e-11, 'maxiter': 1000})
        if res.fun < bestE:
            bestE, bestx = res.fun, res.x
    return bestE, bestx


#  DERIVADAS EXACTAS (JAX): el MISMO funcional cerrado que energy_closed / energy_closed_sym
def _energy_closed_jax(x, eps, epsp, J, Lam):
    eu, ed, ru, rd, ku, kd = x[0], x[1], x[2], x[3], x[4], x[5]
    Dl = Lam + ed - eu
    ch = jnp.cosh(ru - rd)
    F  = jnp.exp(-Dl**2/(jnp.exp(-2.0*ru) + jnp.exp(-2.0*rd)))/jnp.sqrt(ch)
    su, sd = jnp.sinh(ru), jnp.sinh(rd)
    au = Dl*jnp.exp(rd)*su/ch
    ad = Dl*jnp.exp(ru)*sd/ch
    Gu = F*((eu - au)*(ed + Lam + ad) + su*sd/ch)
    Gd = F*((ed + ad)*(eu - Lam - au) + su*sd/ch)
    nu = 1.0 + ku*ku + 2.0*ku*F
    nd = 1.0 + kd*kd + 2.0*kd*F
    A = eps  + omega0*(eu*eu + su**2 + ku*ku*((ed + Lam)**2 + sd**2) + 2.0*ku*Gu)/nu
    B = epsp + omega0*(ed*ed + sd**2 + kd*kd*((eu - Lam)**2 + su**2) + 2.0*kd*Gd)/nd
    T = J*(ku + kd + (1.0 + ku*kd)*F)/jnp.sqrt(nu*nd)
    return 0.5*(A + B) - jnp.sqrt(0.25*(A - B)**2 + T*T)


def _energy_closed_sym_jax(xs, eps, epsp, J, Lam):
    eta_n, r, kap = 2.0*xs[0], xs[1], xs[2]
    F   = jnp.exp(-0.5*(Lam - eta_n)**2*jnp.exp(2.0*r))
    sh2 = jnp.sinh(r)**2
    den = 1.0 + kap**2 + 2.0*kap*F
    num = (omega0*(0.25*eta_n**2 + (1.0 + kap**2)*sh2 + kap**2*(Lam - 0.5*eta_n)**2
                   + 2.0*kap*F*(0.25*Lam**2 - 0.25*(Lam - eta_n)**2*jnp.exp(4.0*r) + sh2))
           - jnp.abs(J)*jnp.abs(2.0*kap + (1.0 + kap**2)*F))
    return eps + num/den


_AD_CACHE = {}


def _ad(n_par):
    """(E, ∇E, H) exactos y compilados para el funcional de n_par parámetros."""
    if n_par not in _AD_CACHE:
        f = _energy_closed_sym_jax if n_par == 3 else _energy_closed_jax
        _AD_CACHE[n_par] = (jax.jit(f), jax.jit(jax.grad(f)), jax.jit(jax.hessian(f)))
    return _AD_CACHE[n_par]


def _polish_newton(fun, n_par, x, args, bnd, itmax=40):
    """Newton amortiguado (Levenberg–Marquardt) con ∇E y H exactos, en las variables libres.
    Solo acepta pasos que bajan la energía (evaluada con el funcional numpy de producción).
    En las direcciones casi planas (λ_min ~ 1e-12) el amortiguamiento impide pasos enormes."""
    _, g_f, H_f = _ad(n_par)
    lo = np.array([b[0] for b in bnd]); hi = np.array([b[1] for b in bnd])
    x = np.clip(np.asarray(x, float), lo, hi)
    E = fun(x, *args)
    mu = None
    for _ in range(itmax):
        g = np.asarray(g_f(x, *args), float)
        H = np.asarray(H_f(x, *args), float); H = 0.5*(H + H.T)
        free = ~(((x <= lo + 1e-12) & (g > 0)) | ((x >= hi - 1e-12) & (g < 0)))
        if not free.any() or np.max(np.abs(g[free])) < 1e-14:
            break
        Hf, gf = H[np.ix_(free, free)], g[free]
        scale = max(float(np.max(np.abs(np.diag(Hf)))), 1e-300)
        mu = 1e-8*scale if mu is None else mu
        accepted = False
        for _ in range(40):
            try:
                d = -np.linalg.solve(Hf + mu*np.eye(len(gf)), gf)
            except np.linalg.LinAlgError:
                mu *= 10.0
                continue
            xn = x.copy(); xn[free] += d; xn = np.clip(xn, lo, hi)
            En = fun(xn, *args)
            if np.isfinite(En) and En < E:
                x, E, accepted = xn, En, True
                mu = max(0.1*mu, 1e-16*scale)
                break
            mu *= 10.0
        if not accepted:
            break
    return E, x


def _min_variation(fun, n_par, x_star, E_star, x_prev, args, bnd):
    """Entre los x con E(x) ≤ E* + δ (δ = MV_DELTA_REL·|E*|), el más cercano al punto anterior
    de la continuación. Donde el mínimo está bien definido la restricción no deja moverse;
    en los valles planos fija la dirección que la energía no determina.
    SLSQP arranca desde x* y desde el punto anterior y se queda con la solución factible más
    cercana a este: desde x* solo, si L-BFGS-B cayó lejos dentro del valle, quedaban escalones."""
    _, g_f, _ = _ad(n_par)
    lo = np.array([b[0] for b in bnd]); hi = np.array([b[1] for b in bnd])
    s = np.array([1.0, 0.1, 0.1]) if n_par == 3 else np.array([1.0, 1.0, 0.1, 0.1, 0.1, 0.1])
    xp = np.asarray(x_prev, float)
    delta = max(MV_DELTA_REL*abs(E_star), 1e-14)
    cons = {'type': 'ineq',
            'fun': lambda x: (E_star + delta - fun(x, *args))/delta,
            'jac': lambda x: -np.asarray(g_f(x, *args), float)/delta}
    best = (np.inf, E_star, np.asarray(x_star, float))
    for x0 in (np.asarray(x_star, float), np.clip(xp, lo, hi)):     # desde el óptimo y desde el punto anterior
        try:
            res = minimize(lambda x: np.sum(((x - xp)/s)**2), x0,
                           jac=lambda x: 2.0*(x - xp)/s**2, method='SLSQP', bounds=bnd,
                           constraints=[cons], options={'ftol': 1e-14, 'maxiter': 300})
            x = np.clip(res.x, lo, hi)
            E = fun(x, *args)
        except Exception:
            continue
        dist = np.sum(((x - xp)/s)**2)
        if np.isfinite(E) and E <= E_star + 1.01*delta and dist < best[0]:
            best = (dist, E, x)
    return best[1], best[2]


def _sym_to_full(xs):
    """(η, r, κ) -> (η, −η, r, r, κ, κ): paridad de Rabi para δ_α = 0."""
    eta, r, kap = xs[0], xs[1], xs[2]
    return np.array([eta, -eta, r, r, kap, kap])


def minimize_sector(sector, eps, epsp, J, Lam, x_warm=None, info=None):
    """Devuelve (E, x) con x = (η↑, η↓, r↑, r↓, κ↑, κ↓).
    Si δ_α = (ε − ε')/2 = 0, se impone la paridad de Rabi Ŝ_α de las notas:
    η↓ = −η↑, r↓ = r↑, κ↓ = κ↑, y se minimiza el funcional reducido E(η, r, κ).
    Después de L-BFGS-B (con JAX): pulido de Newton y, si hay punto anterior (x_warm),
    continuación de variación mínima. 'info' (opcional) recibe E y x antes de ambos pasos."""
    sym = abs(eps - epsp) < 1e-12
    if sym:
        xw = None if x_warm is None else (x_warm[0], x_warm[2], x_warm[4])
        fun, n_par = energy_closed_sym, 3
    else:
        xw = None if x_warm is None else tuple(x_warm[:6])
        fun, n_par = energy_closed, 6
    args = (eps, epsp, J, Lam)
    E, x = _run_seeds(fun, args, Lam, n_par, xw, sector)
    E_raw, x_raw = E, np.array(x, float)
    if HAS_JAX and POLISH:
        E, x = _polish_newton(fun, n_par, x, args, _bounds(Lam, n_par))
    E_pol = E
    if HAS_JAX and MIN_VARIATION and xw is not None:
        E, x = _min_variation(fun, n_par, x, E, xw, args, _bounds(Lam, n_par))
    to_full = _sym_to_full if sym else (lambda v: np.array(v, float))
    if info is not None:
        info.update(E_raw=E_raw, x_raw=to_full(x_raw), E_pol=E_pol)
    return E, to_full(x)


#  EXACTO (un modo acoplado, convención de las notas)
def exact_ground(eps, epsp, J, Lam, M):
    n1 = M + 1
    DL = disp_matrix(M, Lam)
    nd = np.arange(n1)*omega0
    H  = np.zeros((2*n1, 2*n1))
    H[:n1, :n1] = np.diag(eps + nd)
    H[n1:, n1:] = np.diag(epsp + nd)
    H[:n1, n1:] = J*DL
    H[n1:, :n1] = J*DL.T
    w, V = eigh(H)
    return w[0], V[:, 0]


#  BARRIDOS
def sweep_Lambda(sector, cv, Lam_arr):
    """Barrido en Λ con continuación (warm start). t = Δ = cv."""
    n    = len(Lam_arr)
    X    = np.full((6, n), np.nan)
    Ev   = np.full(n, np.nan)
    Eex  = np.full(n, np.nan)
    Efk  = np.full(n, np.nan)
    fid  = np.full(n, np.nan)
    rel  = np.full(n, np.nan)
    leak = np.full(n, np.nan)
    cu   = np.full(n, np.nan)
    cd   = np.full(n, np.nan)
    Eraw = np.full(n, np.nan)          # L-BFGS-B, antes del pulido y de la variación mínima
    Epol = np.full(n, np.nan)          # después del pulido de Newton
    Xraw = np.full((6, n), np.nan)
    fraw = np.full(n, np.nan)
    warm = None
    for k, Lam in enumerate(Lam_arr):
        lam1, lam2 = Lam_to_lam12(Lam, 0.0)
        if np.isnan(lam1):
            continue
        M = _Mof_Lam(Lam)
        eps, epsp, J = sector_params(sector, lam1, lam2, eps1_val, eps2_val, cv, cv)
        info = {}
        E, x = minimize_sector(sector, eps, epsp, J, Lam, warm, info)
        warm = x
        Eraw[k], Epol[k], Xraw[:, k] = info['E_raw'], info['E_pol'], info['x_raw']
        X[:, k] = x
        Ev[k]   = E
        ops = get_ops(M)
        A, B, C = closed_ABC(x, eps, epsp, J, Lam)
        c1, c2  = weights_2x2(A, B, C)
        pu, pd  = var_states(x, Lam, ops)
        E0, psi_ex = exact_ground(eps, epsp, J, Lam, M)
        psi_var = np.concatenate([c1*pu, c2*pd])
        Eex[k]  = E0
        Efk[k]  = energy_fock(x, eps, epsp, J, Lam, ops)
        fid[k]  = abs(psi_ex @ psi_var)**2
        rel[k]  = abs(E - E0)/abs(E0) if abs(E0) > 1e-12 else np.nan
        leak[k] = max(np.sum(pu[-5:]**2), np.sum(pd[-5:]**2))
        cu[k], cd[k] = c1, c2
        Ar, Br, Cr = closed_ABC(info['x_raw'], eps, epsp, J, Lam)
        r1, r2 = weights_2x2(Ar, Br, Cr)
        pur, pdr = var_states(info['x_raw'], Lam, ops)
        fraw[k] = abs(psi_ex @ np.concatenate([r1*pur, r2*pdr]))**2
    return dict(X=X, Evar=Ev, Eex=Eex, Efock=Efk, fid=fid, rel=rel, leak=leak, cu=cu, cd=cd,
                Eraw=Eraw, Epol=Epol, Xraw=Xraw, fid_raw=fraw)


def compute_fig2():
    _OPS_CACHE.clear()
    keys = [(s, cv) for cv in cv_list for s in ('odd', 'even')]
    res  = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(sweep_Lambda)(s, cv, LAM_F2) for s, cv in keys)
    return dict(zip(keys, res))


#  GUARDADO (.txt en DATA_DIR; los lee plot_variational_P_AP.py)
X_NAMES = ('eta_up', 'eta_dn', 'r_up', 'r_dn', 'kap_up', 'kap_dn')        # x = (η↑, η↓, r↑, r↓, κ↑, κ↓)
SCALARS = ('Evar', 'Eex', 'Efock', 'fid', 'rel', 'leak', 'cu', 'cd', 'Eraw', 'Epol', 'fid_raw')


def save_fig2(R):
    names = ('Lam',) + X_NAMES + SCALARS + tuple(n + '_raw' for n in X_NAMES)
    for (sector, cv), d in R.items():
        A = np.column_stack([LAM_F2] + list(d['X']) + [d[k] for k in SCALARS] + list(d['Xraw']))
        np.savetxt(os.path.join(DATA_DIR, f'sweep_{sector}_cv{cv}.txt'), A, fmt='%.16e',
                   header=(f'barrido en Lambda (delta lambda = 0), sector = {sector}, J/omega0 = {cv}, '
                           f'eps1 = {eps1_val}, eps2 = {eps2_val}, omega0 = {omega0}\n' + ' '.join(names)))


#  TESTS IMPRESOS
def _hr(title):
    print('\n' + '='*72 + f'\n {title}\n' + '='*72)


def tests_pre():
    _hr('TESTS PREVIOS')
    rng = np.random.default_rng(1234)
    ops = get_ops(170)

    # T2: funcional cerrado general vs el mismo funcional evaluado en Fock, puntos aleatorios
    for sector in ('odd', 'even'):
        for Lam in (1.0, 2.5, 4.0):
            lam1, lam2 = Lam_to_lam12(Lam, 0.0)
            eps, epsp, J = sector_params(sector, lam1, lam2, eps1_val, eps2_val, 1.0, 1.0)
            d = []
            for _ in range(100):
                x6 = np.array([rng.uniform(-Lam, Lam), rng.uniform(-Lam, Lam),
                               rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8),
                               rng.uniform(-0.9, 0.9), rng.uniform(-0.9, 0.9)])
                d.append(abs(energy_fock(x6, eps, epsp, J, Lam, ops)
                             - energy_closed(x6, eps, epsp, J, Lam)))
            print(f'[T2] {sector:4s}  Λ={Lam:3.1f}  max|E_Fock - E_cerrada| = {max(d):.2e}')

    # T2b: funcional reducido E(η, r, κ) vs funcional general en la reducción simétrica
    d = []
    for _ in range(200):
        Lam = rng.uniform(0.2, 5.0)
        xs  = (rng.uniform(-Lam, Lam), rng.uniform(-0.8, 0.3), rng.uniform(-0.9, 0.9))
        eps = rng.uniform(-3.0, 1.0)
        d.append(abs(energy_closed_sym(xs, eps, eps, 1.0, Lam)
                     - energy_closed(_sym_to_full(xs), eps, eps, 1.0, Lam)))
    print(f'[T2b] max|E_reducido(η,r,κ) - E_general(simétrico)| = {max(d):.2e}')

    # T2c: paridad de Rabi: con la reducción simétrica, |φ↓> = Π|φ↑>
    Pi = (-1.0)**np.arange(ops.dim)
    for xs in ((0.3, -0.1, 0.2), (0.6, -0.2, 0.1)):
        pu, pd = var_states(_sym_to_full(xs), 2.5, ops)
        print(f'[T2c] (η,r,κ)={xs}:  ||φ↓ - Π φ↑|| = {np.linalg.norm(pd - Pi*pu):.2e}')

    # T9: funcional JAX (derivadas del pulido y de la variación mínima) = funcional numpy
    if not HAS_JAX:
        print('[T9] JAX no está instalado: se omiten el pulido de Newton y la variación mínima')
        return
    dE, dg = 0.0, 0.0
    for sector in ('odd', 'even'):
        for Lam in (0.5, 2.0, 3.5, 5.0):
            lam1, lam2 = Lam_to_lam12(Lam, 0.0)
            eps, epsp, J = sector_params(sector, lam1, lam2, eps1_val, eps2_val, 1.0, 1.0)
            for n_par, fun in ((6, energy_closed), (3, energy_closed_sym)):
                E_j, g_j, _ = _ad(n_par)
                lo = np.array([b[0] for b in _bounds(Lam, n_par)])
                hi = np.array([b[1] for b in _bounds(Lam, n_par)])
                for _ in range(20):
                    x = lo + (hi - lo)*rng.random(n_par)
                    a = (eps, eps, J, Lam) if n_par == 3 else (eps, epsp, J, Lam)
                    dE = max(dE, abs(float(E_j(x, *a)) - fun(x, *a)))
                    h = 1e-6
                    gfd = np.array([(fun(x + h*e, *a) - fun(x - h*e, *a))/(2*h) for e in np.eye(n_par)])
                    dg = max(dg, float(np.max(np.abs(np.asarray(g_j(x, *a)) - gfd))))
    print(f'[T9] máx|E_JAX - E_numpy| = {dE:.2e}   máx|∇E_JAX - ∇E_FD| = {dg:.2e}   '
          f'(pulido = {POLISH}, variación mínima = {MIN_VARIATION}, δ = {MV_DELTA_REL:.0e}·|E*|)')


def tests_post(R):
    _hr('TESTS POSTERIORES')
    for (sector, cv), d in R.items():
        gap = d['Evar'] - d['Eex']
        print(f'[T5] {sector:4s} J/ω0={cv}:  min(E_var - E_ex) = {np.nanmin(gap):+.2e}'
              f'   violaciones de la cota variacional = {int(np.sum(gap < -1e-9))}')
        print(f'[T6] {sector:4s} J/ω0={cv}:  máx|E_cerrada - E_Fock| en el óptimo = '
              f'{np.nanmax(np.abs(d["Evar"] - d["Efock"])):.2e}')
        print(f'[T7] {sector:4s} J/ω0={cv}:  máx. peso en los 5 últimos niveles de Fock = {np.nanmax(d["leak"]):.2e}')
        nk = int(np.sum(np.abs(d['X'][4:6]) > KAP_MAX - 1e-6))
        neg = np.any(d['X'][4:6] < 1e-6, axis=0)
        print(f'[T7] {sector:4s} J/ω0={cv}:  κ ∈ [{np.nanmin(d["X"][4:6]):+.3f}, {np.nanmax(d["X"][4:6]):+.3f}]'
              f'   puntos con |κ| en la cota ({KAP_MAX}) = {nk}   puntos con κ en la cota inferior (0) = {int(np.sum(neg))}'
              + (f'   Λ = {np.array2string(LAM_F2[neg], precision=3, separator=", ")}' if neg.any() else ''))
        if sector == 'odd':
            print(f'[T8] odd  J/ω0={cv}:  máx|c↑ + c↓| (δ_o = 0) = {np.nanmax(np.abs(d["cu"] + d["cd"])):.2e}')
        # T10: pulido de Newton y variación mínima: la energía no sube más que δ, la física no cambia,
        #      y la variación total de cada parámetro a lo largo de Λ (antes -> después)
        g_pol = d['Epol'] - d['Eraw']
        up = (d['Evar'] - d['Epol'])/np.abs(d['Epol'])
        tv = lambda Y: np.nansum(np.abs(np.diff(Y, axis=1)), axis=1)
        tv0, tv1 = tv(d['Xraw']), tv(d['X'])
        names = ('η↑', 'η↓', 'r↑', 'r↓', 'κ↑', 'κ↓')
        print(f'[T10] {sector:4s} J/ω0={cv}:  pulido: E_pol - E_LBFGSB ∈ [{np.nanmin(g_pol):+.1e}, {np.nanmax(g_pol):+.1e}]'
              f'   variación mínima: máx (E - E_pol)/|E_pol| = {np.nanmax(up):.1e} (δ = {MV_DELTA_REL:.0e})'
              f'   máx|Δ fidelidad| = {np.nanmax(np.abs(d["fid"] - d["fid_raw"])):.1e}')
        print(f'[T10] {sector:4s} J/ω0={cv}:  variación total en Λ (antes -> después): '
              + '  '.join(f'{nm}: {a:.3g} -> {b:.3g}' for nm, a, b in zip(names, tv0, tv1)))

#Ejecucion
if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    tests_pre()

    _hr('CORRIDA: (P+AP)S, funcional cerrado')
    print('Figuras 2 y 3: barridos en Λ ...')
    R = compute_fig2()
    save_fig2(R)

    tests_post(R)
    print(f'\nDatos guardados en: {DATA_DIR}')
    print('Figuras: python plot_variational_P_AP.py')
