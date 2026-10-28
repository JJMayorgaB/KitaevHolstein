# -*- coding: utf-8 -*-
"""
Estados gato en detuning cero con el ansatz variacional polarón + antipolarón (P+AP)S.

En detuning cero (δ_α = 0) el bloque es
    H = [[ ω0 b†b ,  s D(Λ) ],
         [ s D(Λ)† , ω0 b†b ]],      s = t (impar) o Δ (par),
y la paridad de Rabi Ŝ = s_x e^{iπ b†b} fija (notas):
    |φ↑> = N[ |r, η/2> + κ |r, Λ − η/2> ],   |φ↓> = Π|φ↑>,   c↑ = −c↓ = 1/√2.
La minimización usa únicamente el funcional de energía cerrado (todos los elementos de Fock):
    E(η, r, κ) = { ω0[ η²/4 + (1+κ²) sinh²r + κ²(Λ − η/2)²
                       + 2κ F (Λ²/4 − (Λ−η)² e^{4r}/4 + sinh²r) ]
                   − s |2κ + (1+κ²) F| } / (1 + κ² + 2κF),
    F = exp[−(Λ−η)² e^{2r}/2],   κ ∈ [0, 1].
Con κ = 0 se recupera el funcional original del gato D S.
El vector en Fock solo se construye con los parámetros óptimos (gato, Wigner, solapamientos).

En el marco de laboratorio centrado el gato es
    |Cat_±> ∝ |φ↓>_lab ± |φ↑>_lab ∝ (1 ± κ) [ |r, β> ± |r, −β> ],   β = (Λ − η)/2,
es decir, la forma normalizada del gato es la misma que con D S; κ entra en las
probabilidades p± = (1 ± <φ↑|D(Λ)|φ↓>)/2 y en los parámetros óptimos.

Todo depende solo de Λ: se usa el corte θ = π/4 (λ₁ = λ₂ = Λ/√2).
Este script calcula y guarda los datos (.txt) en Figures/cat/data/cat_PAPS/;
las figuras (sufijo _PAPS, en Figures/cat/) las hace plot_variational_P_AP_cat.py.
La parte EXACTA (diagonalización y gato exacto) se
mantiene igual que en el código original, incluido el corte escalonado en Fock del
solapamiento o_ex; el mismo corte se aplica a o_var.
"""

import os
import warnings
from math import comb as _C

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.special import factorial, eval_genlaguerre
import scipy.sparse as sps
from scipy.sparse.linalg import expm_multiply
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

# ══════════════════════════════════════════════════════════════════
#  PARÁMETROS
# ══════════════════════════════════════════════════════════════════
N_JOBS  = -1

POLISH        = True           # pulido de Newton amortiguado (∇E y H exactos) tras L-BFGS-B
MIN_VARIATION = True           # continuación de variación mínima en Λ (valles planos)
MV_DELTA_REL  = 1e-12          # holgura en energía de la variación mínima: E ≤ E* + MV_DELTA_REL·|E*|

omega0  = 1.0
cv_list = [0.5, 1.0, 1.5]       # J_α/ω0 = s/ω0
KAP_MAX = 1.0                   # κ ∈ [0, KAP_MAX] (redundancia κ -> 1/κ con η -> 2Λ − η)
RHO     = 1.0                   # corte θ = π/4

# mallas en Λ (las mismas de las figuras originales)
LAM_PARAMS     = np.linspace(0.01, 4.0*np.sqrt(2), 200)        # parámetros del gato
LAM_INFID_TWIN = np.linspace(0.01, 5.0, 200)                   # infidelidad 1D, eje doble (+ 2x2)
LAM_INFID      = np.linspace(0.01, 5.0, 300)                   # infidelidad 1D, un eje
LAM_CUT        = np.sqrt(2.0)*np.linspace(0.01, 4.0, 300)      # p±, β̃, solapamientos

# negatividad δ_W, Wigner y Wigner conjunta: se calculan para cada J_α/ω0 de cv_list
LAM_NEG_EX  = np.linspace(0.01, 7.0, 200)
# malla común (exacto y variacional) en coordenadas de α, Δx = Δp = 0.02.
# Convergencia verificada contra x∈[−10,10], p∈[−7,7] (1501×1051): |Δδ| ≤ 6e-5, ∫W = 1.
X_NEG_EX    = np.linspace(-8.0, 8.0, 801)
P_NEG_EX    = np.linspace(-5.5, 5.5, 551)
LAM_NEG_VAR = np.linspace(0.01, 7.0, 300)
X_NEG_VAR   = X_NEG_EX
P_NEG_VAR   = P_NEG_EX

# Wigner exacto vs variacional (λ₁ = λ₂ = λ_val)
WIG_POINTS = [(3.0, (7.0, 7.0)), (2.0, (9.0, 9.0))]
WIG_GRID   = np.linspace(-4.2, 4.2, 301)

# Wigner conjunta de dos modos
LAM_JOINT    = 3.0*np.sqrt(2)
THETA_JOINT  = np.pi/4
N_FOCK_JOINT = 50
SECTOR_JOINT = 'even'
SIGN_JOINT   = -1
UV_JOINT     = np.linspace(-3.5, 3.5, 201)

OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Figures', 'cat')
DATA_DIR = os.path.join(OUT_DIR, 'data', 'cat_PAPS')        # datos que lee plot_variational_P_AP_cat.py


def _trapz(y, x, axis=-1):
    f = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')
    return f(y, x, axis=axis)


# ══════════════════════════════════════════════════════════════════
#  FOCK (funciones originales)
# ══════════════════════════════════════════════════════════════════
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


def disp_matrix_complex(M_ph, alpha):
    """Elementos de matriz <m'|D(alpha)|n>, alpha complejo."""
    N       = M_ph + 1
    alpha   = complex(alpha)
    alpha_c = np.conj(alpha)
    abs2    = (alpha * alpha_c).real
    gauss   = np.exp(-0.5 * abs2)
    facts   = np.array([float(factorial(k, exact=True)) for k in range(N)])
    D = np.zeros((N, N), dtype=complex)
    for mp in range(N):
        for n in range(N):
            if mp >= n:
                k    = mp - n
                pref = gauss * (alpha**k) * np.sqrt(facts[n] / facts[mp])
                D[mp, n] = pref * eval_genlaguerre(n, k, abs2)
            else:
                k    = n - mp
                pref = gauss * ((-alpha_c)**k) * np.sqrt(facts[mp] / facts[n])
                D[mp, n] = pref * eval_genlaguerre(mp, k, abs2)
    return D


def squeezed_vacuum(M, r):
    """S(r)|0>, convención S(r)=exp[r/2(â²−â†²)]."""
    vec  = np.zeros(M + 1)
    pref = 1.0 / np.sqrt(np.cosh(r))
    th   = np.tanh(r)
    for k in range(M // 2 + 1):
        if 2*k <= M:
            vec[2*k] = pref * ((-th)**k) * np.sqrt(float(_C(2*k, k))) / (2.0**k)
    return vec


def _Lambda(lam1, lam2):
    return np.sqrt(lam1**2 + lam2**2)


def _Mof_Lam(Lam):
    return int(min(170, max(50, np.ceil(4.0*Lam*Lam) + 25)))


def _lam12_from_Lam_rho(Lam, rho):
    lam2 = Lam / np.sqrt(1.0 + rho**2)
    lam1 = rho * lam2
    return lam1, lam2


# ══════════════════════════════════════════════════════════════════
#  OPERADORES DEL ANSATZ (construidos a mano, como en variational_trisqueezing.py)
# ══════════════════════════════════════════════════════════════════
class FockOps:
    """G1 = b†−b (D), G2 = (b²−b†²)/2 (S);
    exp(xG) = V exp(−i x w) V†, con H = iG = V w V†."""

    def __init__(self, M):
        self.M    = M
        self.dim  = M + 1
        self.nvec = np.arange(self.dim, dtype=float)
        b  = np.diag(np.sqrt(np.arange(1, self.dim, dtype=float)), 1)
        bd = b.T
        self.b = b
        gens = {1: bd - b, 2: 0.5*(b @ b - bd @ bd)}
        self.eig = {}
        for k, G in gens.items():
            w, V = eigh(1j*G)
            self.eig[k] = (w, V, V.conj().T)

    def apply(self, k, x, v):
        if x == 0.0:
            return v
        w, V, Vh = self.eig[k]
        return (V @ (np.exp(-1j*x*w) * (Vh @ v))).real

    def state(self, eta, r):
        """|r, η> = D(η) S(r)|0>."""
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


# ══════════════════════════════════════════════════════════════════
#  FUNCIONAL DEL GATO
# ══════════════════════════════════════════════════════════════════
def energy_grad_cat(x, Lam, s):
    """Funcional analítico original E(a, r) (τ = 0). Solo para los tests."""
    a, r = x
    d = (Lam - a)
    A = d*d
    B = np.exp(2.0*r)
    F = np.exp(-0.5*A*B)
    E = omega0*(0.25*a*a + np.sinh(r)**2) - s*F
    dE_da = 0.5*omega0*a       - s*F*B*d
    dE_dr = omega0*np.sinh(2.0*r) + s*A*B*F
    return E, np.array([dE_da, dE_dr])


def _minimize_cat(lam1, lam2, s, x_warm=None):
    """Minimizador analítico original (τ = 0). Solo para los tests."""
    Lam = np.sqrt(lam1**2 + lam2**2)
    seeds = [(0.0, 0.0), (0.0, -0.1), (Lam, 0.0), (Lam, -0.25), (0.5*Lam, -0.1)]
    if x_warm is not None:
        seeds = [tuple(x_warm)] + seeds
    bnd = [(-Lam-1, Lam+1), (-1.0, 0.0)]
    lo = [b[0] for b in bnd]; hi = [b[1] for b in bnd]
    bestE, bestx = np.inf, None
    for sd in seeds:
        x0 = np.clip(np.array(sd, float), lo, hi)
        res = minimize(energy_grad_cat, x0, args=(Lam, s), jac=True, method='L-BFGS-B',
                       bounds=bnd, options={'ftol': 1e-14, 'gtol': 1e-11, 'maxiter': 500})
        if res.fun < bestE:
            bestE, bestx = res.fun, res.x
    return bestE, bestx


def energy_paps_closed(x, Lam, s):
    """Funcional cerrado de (P+AP)S en δ = 0, x = (η, r, κ) (el que se minimiza):
    E = {ω0[η²/4 + (1+κ²) sinh²r + κ²(Λ − η/2)² + 2κF(Λ²/4 − (Λ−η)² e^{4r}/4 + sinh²r)]
         − s|2κ + (1+κ²)F|} / (1 + κ² + 2κF),   F = exp[−(Λ−η)² e^{2r}/2]."""
    eta, r, kap = x
    F   = np.exp(-0.5*(Lam - eta)**2*np.exp(2.0*r))
    sh2 = np.sinh(r)**2
    den = 1.0 + kap*kap + 2.0*kap*F
    if den < 1e-12:                 # κ = −1 con F = 1: |φ↑> = 0, no es un estado
        return 1e6
    num = (omega0*(0.25*eta**2 + (1.0 + kap*kap)*sh2 + kap*kap*(Lam - 0.5*eta)**2
                   + 2.0*kap*F*(0.25*Lam**2 - 0.25*(Lam - eta)**2*np.exp(4.0*r) + sh2))
           - abs(s)*abs(2.0*kap + (1.0 + kap*kap)*F))
    return num/den


def phis_paps(x, Lam, ops):
    """|φ↑>, |φ↓> = Π|φ↑> (marco de Lang-Firsov, normalizados) con x = (η, r, κ)."""
    eta, r, kap = x[0], x[1], x[2]
    pu  = ops.state(+0.5*eta, r) + kap*ops.apply(1, Lam, ops.state(-0.5*eta, r))
    pu  = pu/np.linalg.norm(pu)
    pd  = ((-1.0)**ops.nvec)*pu
    return pu, pd


def cat_quantities(x, Lam, s, ops):
    """E (evaluada en Fock), o = <φ↑|D(Λ)|φ↓>, |φ↑>, |φ↓> con x = (η, r, κ)."""
    pu, pd = phis_paps(x, Lam, ops)
    m  = pu @ (ops.nvec*pu)
    F  = pu @ ops.apply(1, Lam, pd)
    E  = omega0*m - abs(s)*abs(F)
    return E, F, pu, pd


def energy_cat_fock(x, Lam, s, ops):
    """El mismo funcional evaluado en Fock (solo para los tests)."""
    return cat_quantities(x, Lam, s, ops)[0]


#  DERIVADAS EXACTAS (JAX): el MISMO funcional cerrado que energy_paps_closed
def energy_paps_closed_jax(x, Lam, s):
    eta, r, kap = x[0], x[1], x[2]
    F   = jnp.exp(-0.5*(Lam - eta)**2*jnp.exp(2.0*r))
    sh2 = jnp.sinh(r)**2
    den = 1.0 + kap*kap + 2.0*kap*F
    num = (omega0*(0.25*eta**2 + (1.0 + kap*kap)*sh2 + kap*kap*(Lam - 0.5*eta)**2
                   + 2.0*kap*F*(0.25*Lam**2 - 0.25*(Lam - eta)**2*jnp.exp(4.0*r) + sh2))
           - jnp.abs(s)*jnp.abs(2.0*kap + (1.0 + kap*kap)*F))
    return num/den


_AD_CACHE = {}


def _ad():
    """(E, ∇E, H) exactos y compilados del funcional cerrado."""
    if 'f' not in _AD_CACHE:
        f = energy_paps_closed_jax
        _AD_CACHE['f'] = (jax.jit(f), jax.jit(jax.grad(f)), jax.jit(jax.hessian(f)))
    return _AD_CACHE['f']


def _polish_newton(fun, x, args, bnd, itmax=40):
    """Newton amortiguado (Levenberg–Marquardt) con ∇E y H exactos, en las variables libres.
    Solo acepta pasos que bajan la energía (evaluada con el funcional numpy)."""
    _, g_f, H_f = _ad()
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


def _min_variation(fun, x_star, E_star, x_prev, args, bnd):
    """Entre los x con E(x) ≤ E* + δ (δ = MV_DELTA_REL·|E*|), el más cercano al punto anterior
    de la continuación. SLSQP arranca desde x* y desde el punto anterior y se queda con la
    solución factible más cercana a este."""
    _, g_f, _ = _ad()
    lo = np.array([b[0] for b in bnd]); hi = np.array([b[1] for b in bnd])
    s = np.array([1.0, 0.1, 0.1])                     # escalas de (η, r, κ)
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


def _postprocess(fun, E, x, x_warm, args, bnd, info):
    """Pulido de Newton y, con punto anterior, variación mínima. info recibe E y x previos."""
    E_raw, x_raw = E, np.array(x, float)
    if HAS_JAX and POLISH:
        E, x = _polish_newton(fun, x, args, bnd)
    E_pol = E
    if HAS_JAX and MIN_VARIATION and x_warm is not None:
        E, x = _min_variation(fun, x, E, np.asarray(x_warm[:3], float), args, bnd)
    if info is not None:
        info.update(E_raw=E_raw, x_raw=x_raw, E_pol=E_pol)
    return E, x


def _jax_selftest(fun, bnd_of, s_val, rng):
    """máx|E_JAX − E_numpy| y máx|∇E_JAX − ∇E_FD| en puntos aleatorios dentro de las cotas."""
    E_j, g_j, _ = _ad()
    dE = dg = 0.0
    for Lam in (0.5, 2.0, 3.5, 5.0):
        b = bnd_of(Lam)
        lo = np.array([v[0] for v in b]); hi = np.array([v[1] for v in b])
        for _ in range(30):
            x = lo + (hi - lo)*rng.random(3)
            dE = max(dE, abs(float(E_j(x, Lam, s_val)) - fun(x, Lam, s_val)))
            h = 1e-6
            gfd = np.array([(fun(x + h*e, Lam, s_val) - fun(x - h*e, Lam, s_val))/(2*h) for e in np.eye(3)])
            dg = max(dg, float(np.max(np.abs(np.asarray(g_j(x, Lam, s_val)) - gfd))))
    return dE, dg


def minimize_cat(Lam, s, x_warm=None, info=None):
    """Minimiza el funcional cerrado. Devuelve (E, (η, r, κ)).
    Semillas (η, r) del original combinadas con κ ∈ [0, 1].
    Después de L-BFGS-B (con JAX): pulido de Newton y, si hay punto anterior (x_warm),
    continuación de variación mínima. 'info' (opcional) recibe E y x antes de ambos pasos."""
    base = [(0.0, 0.0), (0.0, -0.1), (Lam, 0.0), (Lam, -0.25), (0.5*Lam, -0.1)]
    kaps = (0.15, 0.0) if x_warm is not None else (0.15, 0.5, 0.0)
    bnd  = [(-Lam - 1.0, 2.0*Lam + 1.0), (-1.0, 0.0), (0.0, KAP_MAX)]
    seeds = [sd + (k,) for sd in base for k in kaps]
    if x_warm is not None:
        seeds = [tuple(x_warm[:3])] + seeds
    lo = np.array([b[0] for b in bnd]); hi = np.array([b[1] for b in bnd])
    bestE, bestx = np.inf, None
    for sd in seeds:
        x0  = np.clip(np.asarray(sd, float), lo, hi)
        res = minimize(energy_paps_closed, x0, args=(Lam, s), method='L-BFGS-B',
                       bounds=bnd, options={'ftol': 1e-15, 'gtol': 1e-11, 'maxiter': 1000})
        if res.fun < bestE:
            bestE, bestx = res.fun, res.x
    return _postprocess(energy_paps_closed, bestE, bestx, x_warm, (Lam, s), bnd, info)


def variational_cat_vector(Lam, x, sign, M):
    """Gato variacional (modo b, marco original, centrado):
    D(Λ/2)|φ↓> + sign·D(−Λ/2)|φ↑>, normalizado (x = (η, r, κ)).
    Equivale a |r, β> + sign·|r, −β>, β = (Λ−η)/2 (test C8)."""
    ops = get_ops(M)
    pu, pd = phis_paps(x, Lam, ops)
    psi = ops.apply(1, +0.5*Lam, pd) + sign*ops.apply(1, -0.5*Lam, pu)
    return psi/np.linalg.norm(psi)


# ══════════════════════════════════════════════════════════════════
#  EXACTO Y FIDELIDAD DEL GATO (parte exacta igual a la original)
# ══════════════════════════════════════════════════════════════════
def E_cat_exact(lam1, lam2, s, M=None):
    """GS exacto en detuning cero. Devuelve (E, evec, M)."""
    Lam = _Lambda(lam1, lam2)
    if M is None:
        M = _Mof_Lam(Lam)
    n1 = M + 1
    DL = disp_matrix(M, Lam)
    n  = np.diag(np.arange(n1)*omega0)
    H = np.zeros((2*n1, 2*n1))
    H[:n1, :n1] = n
    H[n1:, n1:] = n
    H[:n1, n1:] = s*DL
    H[n1:, :n1] = s*DL.T
    ev, V = eigh(H)
    return ev[0], V[:, 0], M


def _overlap_staircase(v1, v2, Lam, M):
    """<v1|D(Λ)|v2> con el corte escalonado en Fock del notebook original
    (D(Λ) truncada no es unitaria sobre Fock altos)."""
    LAM_CUTS = np.linspace(1.5, 4.0, 20)
    CUTOFFS  = np.linspace(0.999, 0.995, 20)
    n1 = M + 1
    n_passed = int(np.searchsorted(LAM_CUTS, Lam, side='right'))
    if n_passed == 0:
        return np.vdot(v1, disp_matrix(M, +Lam) @ v2).real
    c = CUTOFFS[min(n_passed - 1, len(CUTOFFS) - 1)]

    def _Mc(v):
        cs = np.cumsum(np.abs(v)**2) / max(np.vdot(v, v).real, 1e-30)
        return int(np.searchsorted(cs, c)) + 1

    Mc = min(max(_Mc(v1), _Mc(v2)), n1)
    return np.vdot(v1[:Mc], disp_matrix(Mc - 1, +Lam) @ v2[:Mc]).real


def _cat_overlap(lam1, lam2, x, evec_ex, M, sign=+1):
    """(|fid|, o_ex, o_var) para la rama sign del gato.
    o_ex = 2<φ1|D(Λ)|φ2> y o_var = <φ↑|D(Λ)|φ↓> se calculan con el MISMO corte
    escalonado en Fock del notebook original (_overlap_staircase), de modo que p± y β̃
    salen de la misma fórmula para el estado exacto y para el variacional.
    Parte variacional: |φ↑>, |φ↓> = Π|φ↑> del ansatz (P+AP)."""

    Lam = _Lambda(lam1, lam2)
    n1  = M + 1
    ops = get_ops(M)

    phi1 = evec_ex[:n1]
    phi2 = evec_ex[n1:]

    g_pa, g_ma = phis_paps(x, Lam, ops)     # |φ↑>, |φ↓> (marco de Lang-Firsov)

    DpL2 = disp_matrix(M, +0.5*Lam)
    DmL2 = disp_matrix(M, -0.5*Lam)

    cat_var = DpL2 @ g_ma + sign * (DmL2 @ g_pa)
    cat_var /= np.linalg.norm(cat_var)

    cat_ex = DpL2 @ phi2 + (-sign) * (DmL2 @ phi1)
    cat_ex /= np.linalg.norm(cat_ex)

    fid_overlap = np.vdot(cat_var, cat_ex)

    o_var = _overlap_staircase(g_pa, g_ma, Lam, M)          # <φ↑|D(Λ)|φ↓>
    o_ex  = 2.0 * _overlap_staircase(phi1, phi2, Lam, M)    # φ1, φ2 con norma² 1/2

    return abs(fid_overlap), o_ex, o_var


def exact_cat_state(lam1, lam2, evec_ex, M, sign=+1):
    Lam = _Lambda(lam1, lam2)
    n1  = M + 1
    phi1 = evec_ex[:n1]
    phi2 = evec_ex[n1:]
    DpL2 = disp_matrix(M, +0.5*Lam)
    DmL2 = disp_matrix(M, -0.5*Lam)
    cat_ex = DpL2 @ phi2 + (-sign) * (DmL2 @ phi1)
    return cat_ex / np.linalg.norm(cat_ex)


# ══════════════════════════════════════════════════════════════════
#  BARRIDO EN Λ (θ = π/4)
# ══════════════════════════════════════════════════════════════════
def cat_sweep(cv, Lam_arr):
    n = len(Lam_arr)
    out = {k: np.full(n, np.nan) for k in
           ('Lam', 'eta', 'r', 'kap', 'Evar', 'Efock', 'Eex', 'infid_p', 'infid_m',
            'o_ex', 'o_var', 'leak', 'pp_cf', 'pm_cf', 'th_cf',
            'Eraw', 'Epol', 'eta_raw', 'r_raw', 'kap_raw', 'pp_raw', 'th_raw')}
    warm = None
    for k, Lam in enumerate(Lam_arr):
        lam1, lam2 = _lam12_from_Lam_rho(Lam, RHO)
        M = _Mof_Lam(Lam)
        info = {}
        E, x = minimize_cat(Lam, cv, warm, info)
        warm = x
        # antes del pulido y de la variación mínima (solo para el test C11)
        out['Eraw'][k], out['Epol'][k] = info['E_raw'], info['E_pol']
        out['eta_raw'][k], out['r_raw'][k], out['kap_raw'][k] = info['x_raw']
        e0, r0, k0 = info['x_raw']
        F0  = np.exp(-2.0*(0.5*(Lam - e0))**2*np.exp(2.0*r0))
        N20 = 1.0/(1.0 + k0**2 + 2.0*k0*F0)
        out['pp_raw'][k] = 0.5*N20*(1.0 + k0)**2*(1.0 + F0)
        out['th_raw'][k] = np.sqrt(max(-0.5*np.log(N20*(2.0*k0 + (1.0 + k0**2)*F0)), 0.0))
        Ee, vec, _ = E_cat_exact(lam1, lam2, cv, M=M)
        ov_p, o_ex, o_var = _cat_overlap(lam1, lam2, x, vec, M, sign=+1)
        ov_m, _, _        = _cat_overlap(lam1, lam2, x, vec, M, sign=-1)
        Efk, _, pu, pd = cat_quantities(x, Lam, cv, get_ops(M))
        out['Lam'][k] = Lam
        out['eta'][k], out['r'][k], out['kap'][k] = x
        out['Evar'][k], out['Eex'][k], out['Efock'][k] = E, Ee, Efk
        out['infid_p'][k] = 1.0 - ov_p**2
        out['infid_m'][k] = 1.0 - ov_m**2
        out['o_ex'][k], out['o_var'][k] = o_ex, o_var
        # p± de (P+AP)S con la fórmula cerrada (traslape completo, sin corte en Fock):
        # p± = ½ N²(1 ± κ)²(1 ± e^{−2β²e^{2r}}),  β = (Λ − η)/2,  N^{-2} = 1 + κ² + 2κ e^{−2β²e^{2r}}
        eta_k, r_k, kap_k = x
        F_k  = np.exp(-2.0*(0.5*(Lam - eta_k))**2*np.exp(2.0*r_k))
        N2_k = 1.0/(1.0 + kap_k**2 + 2.0*kap_k*F_k)
        out['pp_cf'][k] = 0.5*N2_k*(1.0 + kap_k)**2*(1.0 + F_k)
        out['pm_cf'][k] = 0.5*N2_k*(1.0 - kap_k)**2*(1.0 - F_k)
        # β̃ de (P+AP)S con la fórmula cerrada: sqrt{−½ ln[N²(2κ + (1 + κ²) e^{−2β²e^{2r}})]}
        out['th_cf'][k] = np.sqrt(max(-0.5*np.log(N2_k*(2.0*kap_k + (1.0 + kap_k**2)*F_k)), 0.0))
        out['leak'][k] = max(np.sum(pu[-5:]**2), np.sum(pd[-5:]**2))
    return out


GRIDS = {'params': LAM_PARAMS, 'twin': LAM_INFID_TWIN,
         'single': LAM_INFID, 'cut': LAM_CUT}


def compute_sweeps():
    _OPS_CACHE.clear()
    keys = [(g, cv) for g in GRIDS for cv in cv_list]
    res  = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(cat_sweep)(cv, GRIDS[g]) for g, cv in keys)
    return dict(zip(keys, res))


# ══════════════════════════════════════════════════════════════════
#  WIGNER (funciones originales, numpy)
# ══════════════════════════════════════════════════════════════════
def wigner_cat(psi, xvec, pvec, verbose=True, n_jobs=-1):
    """W(α) = (2/π) Σ_n (−1)^n |<n|D(−α)|ψ>|², α = x + i p.
    Usa W(x,−p) = W(x,p) (lóbulos en el eje real, coeficientes reales)."""
    N = len(psi)
    lnf = np.zeros(N); lnf[1:] = np.cumsum(np.log(np.arange(1, N)))
    parity = (-1.0) ** np.arange(N)
    mp_idx = np.arange(N)[:, None]
    n_idx  = np.arange(N)[None, :]
    K       = np.abs(mp_idx - n_idx)
    upper   = mp_idx >= n_idx
    deg     = np.where(upper, n_idx, mp_idx)
    lnmin   = np.minimum(lnf[mp_idx], lnf[n_idx])
    lnmax   = np.maximum(lnf[mp_idx], lnf[n_idx])
    sqrtfac = np.exp(0.5 * (lnmin - lnmax))

    def _row(p):
        row = np.zeros(len(xvec))
        for ix, x in enumerate(xvec):
            alpha = -(x + 1j*p)
            ac    = np.conj(alpha)
            abs2  = (alpha * ac).real
            gauss = np.exp(-0.5 * abs2)
            L     = eval_genlaguerre(deg, K, abs2)
            phase = np.where(upper, alpha**K, (-ac)**K)
            D     = gauss * phase * sqrtfac * L
            c     = D @ psi
            row[ix] = (2.0/np.pi) * np.sum(parity * np.abs(c)**2)
        return row

    pvec = np.asarray(pvec)
    idx_pos = np.where(pvec >= 0.0)[0]
    p_pos   = pvec[idx_pos]
    if n_jobs == 1:
        rows_pos = np.array([_row(p) for p in p_pos])
    else:
        rows_pos = np.array(Parallel(n_jobs=n_jobs)(delayed(_row)(p) for p in p_pos))

    W = np.zeros((len(pvec), len(xvec)))
    pos_lookup = {round(pp, 12): rows_pos[k] for k, pp in enumerate(p_pos)}
    for ip, p in enumerate(pvec):
        key = round(abs(p), 12)
        W[ip, :] = pos_lookup[key] if key in pos_lookup else _row(p)
    return W


def wigner_pure_padded(psi, xvec, pvec):
    """W(α) = (2/π) Σ_n (−1)^n |<n|D(−α)|ψ>|², α = x + i p, para un estado puro.
    El estado se rellena con ceros hasta N_pad ≥ (R + √<n>_ψ + 6)², R = máx|α| de la malla,
    para que D(−α) truncado sea exacto en toda la malla aun cuando ψ tiene lóbulos lejos
    del origen. D(−α) ∝ D(−x) D(−i p) (la fase no afecta |·|²); cada factor se exponencia
    vía la diagonalización de su generador."""
    R = np.sqrt(np.max(np.abs(xvec))**2 + np.max(np.abs(pvec))**2)
    n_mean = np.sum(np.arange(len(psi)) * np.abs(psi)**2) / np.sum(np.abs(psi)**2)
    N = max(len(psi), int(np.ceil((R + np.sqrt(n_mean) + 6.0)**2)))
    v = np.zeros(N); v[:len(psi)] = psi
    b  = np.diag(np.sqrt(np.arange(1, N, dtype=float)), 1)
    w1, V1 = eigh(1j*(b.T - b))          # D(x)  = V1 e^{−i x w1} V1†
    w2, V2 = eigh(b.T + b)               # D(iy) = V2 e^{+i y w2} V2ᵀ
    V1h = V1.conj().T
    par = ((-1.0)**np.arange(N))[:, None]
    ph_x = np.exp(1j*np.outer(w1, np.asarray(xvec)))      # D(−x): e^{+i x w1}
    W = np.zeros((len(pvec), len(xvec)))
    for ip, y in enumerate(pvec):
        u  = V2 @ (np.exp(-1j*y*w2) * (V2.T @ v))         # D(−i y)|ψ>
        Dx = V1 @ (ph_x * (V1h @ u)[:, None])             # D(−x) D(−i y)|ψ>, todas las x
        W[ip] = (2.0/np.pi) * np.sum(par * np.abs(Dx)**2, axis=0)
    return W


def wigner_and_cuts_padded(psi, xvec, pvec):
    W = wigner_pure_padded(psi, xvec, pvec)
    ip0 = np.argmin(np.abs(pvec))
    ix0 = np.argmin(np.abs(xvec))
    return W, W[ip0, :], W[:, ix0]


def wigner_and_cuts_num(psi, xvec, pvec, verbose=True):
    W = wigner_cat(psi, xvec, pvec, verbose=verbose)
    ip0 = np.argmin(np.abs(pvec))
    ix0 = np.argmin(np.abs(xvec))
    return W, W[ip0, :], W[:, ix0]


def wigner_negativity(W, xvec, pvec):
    """δ = ½(∫|W| − 1) y ∫W (convención 2/π, coordenadas de α)."""
    norm_check = _trapz(_trapz(W, xvec, axis=1), pvec)
    delta = 0.5 * (_trapz(_trapz(np.abs(W), xvec, axis=1), pvec) - 1.0)
    return delta, norm_check


def _neg_point(Lam, sign, xvec, pvec, cv_val):
    """δ y ∫W del gato EXACTO (rama sign)."""
    lam1 = lam2 = Lam / np.sqrt(2.0)
    M = _Mof_Lam(Lam)
    _, vec, _ = E_cat_exact(lam1, lam2, cv_val, M=M)
    psi = exact_cat_state(lam1, lam2, vec, M, sign=sign)
    W = wigner_pure_padded(psi, xvec, pvec)
    return wigner_negativity(W, xvec, pvec)


def _neg_point_variational(Lam, sign, xvec, pvec, cv_val):
    """δ y ∫W del gato VARIACIONAL (rama sign), parámetros recalculados en Λ."""
    M = _Mof_Lam(Lam)
    _, x = minimize_cat(Lam, cv_val)
    psi = variational_cat_vector(Lam, x, sign, M)
    W = wigner_pure_padded(psi, xvec, pvec)
    return wigner_negativity(W, xvec, pvec)


def compute_negativity_exact(cv):
    _OPS_CACHE.clear()
    tasks = [(L, s) for L in LAM_NEG_EX for s in (+1, -1)]
    res = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(_neg_point)(L, s, X_NEG_EX, P_NEG_EX, cv) for L, s in tasks)
    res = np.array(res).reshape(len(LAM_NEG_EX), 2, 2)
    return dict(Lam=LAM_NEG_EX, dp=res[:, 0, 0], dm=res[:, 1, 0],
                np_=res[:, 0, 1], nm=res[:, 1, 1])


def compute_negativity_var(cv):
    _OPS_CACHE.clear()
    tasks = [(L, s) for L in LAM_NEG_VAR for s in (+1, -1)]
    res = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(_neg_point_variational)(L, s, X_NEG_VAR, P_NEG_VAR, cv)
        for L, s in tasks)
    res = np.array(res).reshape(len(LAM_NEG_VAR), 2, 2)
    return dict(Lam=LAM_NEG_VAR, dp=res[:, 0, 0], dm=res[:, 1, 0],
                np_=res[:, 0, 1], nm=res[:, 1, 1])


# ══════════════════════════════════════════════════════════════════
#  GATO BIMODAL (dos cavidades) CON EL ANSATZ (P+AP)
# ══════════════════════════════════════════════════════════════════
def cat_bimodal(lam1, lam2, cv, N, sector, sign):
    """Gato de dos modos en la base de sitio (vector, idx = m1*N + m2).
    Modo colectivo acoplado (notas): b_e = c â1 + s â2 (par), b_o = c â1 − s â2 (impar),
    c = λ1/Λ, s = λ2/Λ. Lóbulos: D_b(±β) S_b(r)|0,0>, β = (Λ−η)/2,
    es decir, desplazamientos (γ1, ±γ2) = (cβ, ±sβ) en las cavidades.
    Con (P+AP) el gato normalizado tiene esta misma forma (κ solo entra en el peso
    (1 ± κ) que se cancela al normalizar); (η, r) son los óptimos de (P+AP)S.
    Devuelve (psi, (η, r, γ1, γ2, κ))."""
    Lam = np.sqrt(lam1**2 + lam2**2)
    _, x = minimize_cat(Lam, cv)
    eta, r, kap = x
    beta = 0.5*(Lam - eta)
    c, s = lam1/Lam, lam2/Lam
    sig  = +1.0 if sector == 'even' else -1.0

    a  = sps.diags(np.sqrt(np.arange(1, N, dtype=float)), 1, format='csr')
    I  = sps.identity(N, format='csr')
    a1 = sps.kron(a, I, format='csr')
    a2 = sps.kron(I, a, format='csr')
    b  = c*a1 + sig*s*a2
    bd = b.T.tocsr()
    G1 = (bd - b).tocsr()
    G2 = (0.5*(b @ b - bd @ bd)).tocsr()

    vac = np.zeros(N*N); vac[0] = 1.0

    def lobe(bb):
        v = vac
        if r != 0.0:
            v = expm_multiply(r*G2, v)
        return expm_multiply(bb*G1, v)

    psi = lobe(+beta) + sign*lobe(-beta)
    psi = psi/np.linalg.norm(psi)
    return psi, (eta, r, c*beta, sig*s*beta, kap)


def wigner_joint_cut_exact(psi_ph, nph, uvec, vvec, plane='real'):
    """W(α1, α2) en un corte 2D (Re-Re o Im-Im), vector de dos modos idx = m1*nph + m2."""
    psi_mat = psi_ph.reshape(nph, nph)
    par = (-1.0) ** np.arange(nph)
    Par = np.outer(par, par)

    def _Ddag(val):
        a = (val + 0j) if plane == 'real' else (1j*val)
        return disp_matrix_complex(nph-1, -a)

    Ddag_u  = [_Ddag(u) for u in uvec]
    Ddag_vT = [_Ddag(v).T for v in vvec]
    pref = (2.0/np.pi)**2
    W = np.zeros((len(vvec), len(uvec)))
    for iu in range(len(uvec)):
        left = Ddag_u[iu] @ psi_mat
        for iv in range(len(vvec)):
            tpsi = left @ Ddag_vT[iv]
            W[iv, iu] = pref * np.sum(Par * (np.abs(tpsi)**2))
    return W


# ══════════════════════════════════════════════════════════════════
#  WIGNER DE LAS FIGURAS (los datos; las figuras las hace plot_variational_P_AP_cat.py)
# ══════════════════════════════════════════════════════════════════
def compute_wigner_stacks(cv):
    """W exacta y variacional (rama +) para cada λ de WIG_POINTS (λ₁ = λ₂ = λ). Devuelve
    {λ: dict(W_ex, W_v, x)}, malla WIG_GRID en x y en p (filas = p, columnas = x)."""
    xvec = pvec = WIG_GRID
    out = {}
    for lam_val, _ in WIG_POINTS:
        Lam = np.sqrt(2.0)*lam_val
        M   = _Mof_Lam(Lam)
        _, vec, _ = E_cat_exact(lam_val, lam_val, cv, M=M)
        psi_ex = exact_cat_state(lam_val, lam_val, vec, M, sign=+1)
        W_ex, _, _ = wigner_and_cuts_num(psi_ex, xvec, pvec, verbose=False)
        _, x = minimize_cat(Lam, cv)
        psi_v = variational_cat_vector(Lam, x, +1, M)
        W_v, _, _ = wigner_and_cuts_padded(psi_v, xvec, pvec)
        out[lam_val] = dict(W_ex=W_ex, W_v=W_v, x=x)
        print(f'Wigner exacto vs variacional, J/ω0={cv}, λ={lam_val}: (η, r, κ) = ({x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f})')
    return out


def compute_joint_wigner(cv):
    """W(α1, α2) del gato bimodal en los cortes Re-Re e Im-Im (malla UV_JOINT)."""
    lam1 = LAM_JOINT*np.cos(THETA_JOINT)
    lam2 = LAM_JOINT*np.sin(THETA_JOINT)
    psi, pars = cat_bimodal(lam1, lam2, cv, N_FOCK_JOINT, SECTOR_JOINT, SIGN_JOINT)
    print(f'J/ω0={cv}, {SECTOR_JOINT} sign={SIGN_JOINT}: η={pars[0]:.4f} r={pars[1]:.4f} κ={pars[4]:.4f} '
          f'γ1={pars[2]:.3f} γ2={pars[3]:.3f}', flush=True)
    W_re = wigner_joint_cut_exact(psi, N_FOCK_JOINT, UV_JOINT, UV_JOINT, plane='real')
    W_im = wigner_joint_cut_exact(psi, N_FOCK_JOINT, UV_JOINT, UV_JOINT, plane='imag')
    return dict(W_re=W_re, W_im=W_im, pars=pars)


# ══════════════════════════════════════════════════════════════════
#  GUARDADO (.txt en DATA_DIR; los lee plot_variational_P_AP_cat.py)
# ══════════════════════════════════════════════════════════════════
SWEEP_KEYS = ('Lam', 'eta', 'r', 'kap', 'Evar', 'Efock', 'Eex', 'infid_p', 'infid_m',
              'o_ex', 'o_var', 'leak', 'pp_cf', 'pm_cf', 'th_cf',
              'Eraw', 'Epol', 'eta_raw', 'r_raw', 'kap_raw', 'pp_raw', 'th_raw')


def joint_tag():
    return f'{SECTOR_JOINT}_{"minus" if SIGN_JOINT < 0 else "plus"}_Lam{LAM_JOINT:.3f}'


def save_sweeps(S):
    for (g, cv), d in S.items():
        np.savetxt(os.path.join(DATA_DIR, f'sweep_{g}_cv{cv}.txt'),
                   np.column_stack([d[k] for k in SWEEP_KEYS]), fmt='%.16e',
                   header=(f'barrido en Lambda (theta: rho = {RHO}), malla = {g}, J/omega0 = {cv}\n'
                           + ' '.join(SWEEP_KEYS)))


def save_negativity(NE, NV, cv):
    for name, N, X, P in (('exact', NE, X_NEG_EX, P_NEG_EX), ('variational', NV, X_NEG_VAR, P_NEG_VAR)):
        np.savetxt(os.path.join(DATA_DIR, f'negativity_{name}_cv{cv}.txt'),
                   np.column_stack([N['Lam'], N['dp'], N['dm'], N['np_'], N['nm']]), fmt='%.16e',
                   header=(f'delta_W del gato {name} (cv = {cv}), malla x[{X[0]}, {X[-1]}] ({len(X)}) '
                           f'x p[{P[0]}, {P[-1]}] ({len(P)})\n'
                           'Lam delta_plus delta_minus norm_plus norm_minus'))


def save_wigner_stacks(WS, cv):
    np.savetxt(os.path.join(DATA_DIR, 'wigner_grid.txt'), np.column_stack([WIG_GRID, WIG_GRID]),
               fmt='%.16e', header='malla de las Wigner de un modo (alpha = x + i p)\nx p')
    for lam_val, w in WS.items():
        eta, r, kap = w['x']
        hdr = (f'W(alpha), filas = p, columnas = x (wigner_grid.txt); lambda1 = lambda2 = {lam_val}, '
               f'cv = {cv}, rama +')
        np.savetxt(os.path.join(DATA_DIR, f'wigner_exact_lam{lam_val}_cv{cv}.txt'), w['W_ex'], fmt='%.10e',
                   header=hdr + ', exacto')
        np.savetxt(os.path.join(DATA_DIR, f'wigner_var_lam{lam_val}_cv{cv}.txt'), w['W_v'], fmt='%.10e',
                   header=hdr + f', (P+AP)S: eta = {eta:.10e} r = {r:.10e} kappa = {kap:.10e}')


def save_joint_wigner(WJ, cv):
    np.savetxt(os.path.join(DATA_DIR, 'wigner_joint_grid.txt'), UV_JOINT, fmt='%.16e',
               header='malla u = v de la Wigner conjunta')
    eta, r, g1, g2, kap = WJ['pars']
    hdr = (f'W(alpha1, alpha2), filas = v (alpha2), columnas = u (alpha1); Lambda = {LAM_JOINT}, '
           f'theta = {THETA_JOINT}, sector = {SECTOR_JOINT}, sign = {SIGN_JOINT:+d}, cv = {cv}, '
           f'N_FOCK = {N_FOCK_JOINT}; eta = {eta:.10e} r = {r:.10e} kappa = {kap:.10e}')
    np.savetxt(os.path.join(DATA_DIR, f'wigner_joint_{joint_tag()}_cv{cv}_ReRe.txt'), WJ['W_re'], fmt='%.10e',
               header=hdr + ', corte Re-Re')
    np.savetxt(os.path.join(DATA_DIR, f'wigner_joint_{joint_tag()}_cv{cv}_ImIm.txt'), WJ['W_im'], fmt='%.10e',
               header=hdr + ', corte Im-Im')


# ══════════════════════════════════════════════════════════════════
#  TESTS IMPRESOS
# ══════════════════════════════════════════════════════════════════
def _hr(title):
    print('\n' + '='*72 + f'\n {title}\n' + '='*72, flush=True)


def tests_pre():
    _hr('TESTS PREVIOS')
    rng = np.random.default_rng(7)
    # C1: funcional cerrado de (P+AP)S vs el mismo evaluado en Fock; con κ=0, vs el original D S
    for Lam in (1.0, 2.5, 4.0, 5.5):
        ops = get_ops(_Mof_Lam(Lam))
        d, d0 = [], []
        for _ in range(100):
            x = np.array([rng.uniform(0.0, Lam), rng.uniform(-0.6, 0.0), rng.uniform(-0.9, 1.0)])
            d.append(abs(energy_cat_fock(x, Lam, 1.0, ops) - energy_paps_closed(x, Lam, 1.0)))
            x0 = np.array([x[0], x[1], 0.0])
            d0.append(abs(energy_paps_closed(x0, Lam, 1.0) - energy_grad_cat(x0[:2], Lam, 1.0)[0]))
        print(f'[C1] Λ={Lam:3.1f}  máx|E_Fock - E_cerrada(P+AP)S| = {max(d):.2e}'
              f'   máx|E_cerrada(κ=0) - E_DS original| = {max(d0):.2e}')
    # C10: funcional JAX (derivadas del pulido y de la variación mínima) = funcional numpy
    if HAS_JAX:
        dE, dg = _jax_selftest(energy_paps_closed,
                               lambda L: [(-L - 1.0, 2.0*L + 1.0), (-1.0, 0.0), (0.0, KAP_MAX)], 1.0, rng)
        print(f'[C10] máx|E_JAX - E_numpy| = {dE:.2e}   máx|∇E_JAX - ∇E_FD| = {dg:.2e}   '
              f'(pulido = {POLISH}, variación mínima = {MIN_VARIATION}, δ = {MV_DELTA_REL:.0e}·|E*|)')
    else:
        print('[C10] JAX no está instalado: se omiten el pulido de Newton y la variación mínima')
    # C2: óptimo de (P+AP)S <= óptimo del D S original (familias anidadas)
    for cv in cv_list:
        for Lam in (0.5, 1.5, 2.5, 3.5, 4.5):
            l1, l2 = _lam12_from_Lam_rho(Lam, RHO)
            Ea, xa = _minimize_cat(l1, l2, cv)
            En, xn = minimize_cat(Lam, cv)
            print(f'[C2] cv={cv}  Λ={Lam:3.1f}  E_PAPS={En:+.10f}  E_DS={Ea:+.10f}  E_PAPS-E_DS={En-Ea:+.2e}'
                  f'  κ={xn[2]:.4f}')
    # C3: gato bimodal: centros de los lóbulos y norma
    lam1 = LAM_JOINT*np.cos(THETA_JOINT); lam2 = LAM_JOINT*np.sin(THETA_JOINT)
    psi, pars = cat_bimodal(lam1, lam2, 1.0, 30, 'even', +1)
    print(f'[C3] bimodal: norma={np.linalg.norm(psi):.12f}  '
          f'(η,r,κ)=({pars[0]:.4f},{pars[1]:.4f},{pars[4]:.4f})  γ1={pars[2]:.4f}  γ2={pars[3]:.4f}')
    # C4: paridad del gato variacional: <Π> = sign
    for Lam in (1.0, 3.0):
        M = _Mof_Lam(Lam)
        _, x = minimize_cat(Lam, 1.0)
        for sgn in (+1, -1):
            psi = variational_cat_vector(Lam, x, sgn, M)
            par = np.sum(((-1.0)**np.arange(len(psi)))*psi**2)
            print(f'[C4] Λ={Lam:3.1f} rama {sgn:+d}: <Π> = {par:+.12f}')
    # C8: el gato (P+AP)S tiene la forma |r,β> ± |r,−β> (κ solo en el peso) y
    #     p± = (1 ± o)/2 con o = <φ↑|D(Λ)|φ↓> coincide con ||D(Λ/2)φ↓ ± D(−Λ/2)φ↑||²/4
    for Lam in (1.0, 3.0):
        M = _Mof_Lam(Lam); ops = get_ops(M)
        _, x = minimize_cat(Lam, 1.0)
        eta, r, kap = x
        beta = 0.5*(Lam - eta)
        pu, pd = phis_paps(x, Lam, ops)
        o = pu @ ops.apply(1, Lam, pd)
        for sgn in (+1, -1):
            psi  = variational_cat_vector(Lam, x, sgn, M)
            ref  = ops.state(beta, r) + sgn*ops.state(-beta, r)
            ref /= np.linalg.norm(ref)
            raw  = ops.apply(1, +0.5*Lam, pd) + sgn*ops.apply(1, -0.5*Lam, pu)
            p_dir = 0.25*(raw @ raw)
            print(f'[C8] Λ={Lam:3.1f} rama {sgn:+d}: κ={kap:.4f}'
                  f'  ||Cat - Cat(β, r)|| = {np.linalg.norm(psi - ref):.2e}'
                  f'  |p - (1{"+" if sgn > 0 else "-"}o)/2| = {abs(p_dir - 0.5*(1 + sgn*o)):.2e}')


def tests_post(S, NE, NV):
    _hr('TESTS POSTERIORES')
    for (g, cv), d in S.items():
        gap = d['Evar'] - d['Eex']
        neg = d['kap'] < 1e-6
        line = (f'[C5] malla={g:6s} cv={cv}:  min(E_var - E_ex) = {np.nanmin(gap):+.2e}'
                f'  violaciones cota = {int(np.sum(gap < -1e-9))}'
                f'  máx. peso últimos 5 Fock = {np.nanmax(d["leak"]):.1e}'
                f'  κ ∈ [{np.nanmin(d["kap"]):.3f}, {np.nanmax(d["kap"]):.3f}]'
                f'  |κ| en la cota = {int(np.sum(np.abs(d["kap"]) > KAP_MAX - 1e-6))}'
                f'  κ en la cota inferior (0) = {int(np.sum(neg))}')
        if neg.any():
            line += f'  (Λ = {np.array2string(d["Lam"][neg], precision=3, separator=", ")})'
        print(line)
        print(f'[C6] malla={g:6s} cv={cv}:  máx|E_cerrada - E_Fock| en el óptimo = '
              f'{np.nanmax(np.abs(d["Evar"] - d["Efock"])):.2e}')
        # C11: pulido y variación mínima: la energía no sube más que δ, p± y β̃ (fórmula cerrada)
        #      no cambian, y variación total de (η, r, κ) a lo largo de Λ (antes -> después)
        up = (d['Evar'] - d['Epol'])/np.abs(d['Epol'])
        tv = lambda y: np.nansum(np.abs(np.diff(y)))
        print(f'[C11] malla={g:6s} cv={cv}:  E_pol - E_LBFGSB ∈ [{np.nanmin(d["Epol"] - d["Eraw"]):+.1e}, '
              f'{np.nanmax(d["Epol"] - d["Eraw"]):+.1e}]   máx (E - E_pol)/|E_pol| = {np.nanmax(up):.1e}'
              f'   máx|Δp+| = {np.nanmax(np.abs(d["pp_cf"] - d["pp_raw"])):.1e}'
              f'   máx|Δβ̃| = {np.nanmax(np.abs(d["th_cf"] - d["th_raw"])):.1e}'
              f'   variación total η: {tv(d["eta_raw"]):.3g} -> {tv(d["eta"]):.3g}'
              f'  r: {tv(d["r_raw"]):.3g} -> {tv(d["r"]):.3g}  κ: {tv(d["kap_raw"]):.3g} -> {tv(d["kap"]):.3g}')
        if g == 'cut':
            nocut = d['Lam'] < 1.5          # sin corte escalonado: ½(1 ± |o_var|) = fórmula cerrada
            print(f'[C9] cv={cv}:  máx|p+_cerrada + p-_cerrada - 1| = {np.nanmax(np.abs(d["pp_cf"] + d["pm_cf"] - 1)):.2e}'
                  f'   Λ < 1.5: máx|p+_cerrada - ½(1 + |o_var|)| = '
                  f'{np.nanmax(np.abs(d["pp_cf"][nocut] - 0.5*(1 + np.abs(d["o_var"][nocut])))):.2e}')
            th_nc = np.sqrt(np.clip(-0.5*np.log(np.clip(np.abs(d["o_var"][nocut]), 1e-300, None)), 0.0, None))
            print(f'[C9] cv={cv}:  Λ < 1.5: máx|β̃_cerrada - β̃(o_var)| = {np.nanmax(np.abs(d["th_cf"][nocut] - th_nc)):.2e}')
    for cv in NE:
        print(f'[C7] cv={cv}  máx|∫W - 1|: exacto (+)={np.nanmax(np.abs(NE[cv]["np_"]-1)):.2e} '
              f'(−)={np.nanmax(np.abs(NE[cv]["nm"]-1)):.2e} | variacional (+)={np.nanmax(np.abs(NV[cv]["np_"]-1)):.2e} '
              f'(−)={np.nanmax(np.abs(NV[cv]["nm"]-1)):.2e}')


# ══════════════════════════════════════════════════════════════════
#  EJECUCIÓN
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    tests_pre()

    _hr('CORRIDA: (P+AP)S, funcional cerrado')

    print('Barridos en Λ ...', flush=True)
    S = compute_sweeps()
    save_sweeps(S)

    NE, NV = {}, {}
    for cv in cv_list:
        print(f'Negatividad δ_W, J/ω0 = {cv} ...', flush=True)
        NE[cv] = compute_negativity_exact(cv)
        NV[cv] = compute_negativity_var(cv)
        save_negativity(NE[cv], NV[cv], cv)

        print(f'Wigner exacto vs variacional, J/ω0 = {cv} ...', flush=True)
        save_wigner_stacks(compute_wigner_stacks(cv), cv)

        print(f'Wigner conjunta de dos modos, J/ω0 = {cv} ...', flush=True)
        save_joint_wigner(compute_joint_wigner(cv), cv)

    tests_post(S, NE, NV)
    print(f'\nDatos guardados en: {DATA_DIR}')
    print('Figuras: python plot_variational_P_AP_cat.py')
