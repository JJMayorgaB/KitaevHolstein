# -*- coding: utf-8 -*-
"""
variational_tests/common.py
===========================
Núcleo compartido por los niveles 1–4 de la batería de pruebas del optimizador
variacional (P+AP)S.

  * Importa DIRECTAMENTE el script de producción ../variational_P_AP.py: el funcional
    cerrado, las semillas, las cotas, el exacto y los estados en Fock son los mismos
    que usan las figuras. Si algo falla aquí, falla en producción.
  * Reimplementa el MISMO funcional cerrado con diferenciación automática (JAX, o
    autograd como respaldo) para obtener gradiente, hessiano y ∂_Λ∇E exactos
    (niveles 3 y 4). `python common.py --selftest` verifica que coincide con producción.
  * Define los tres casos de prueba, la malla en Λ, las cantidades físicas que se
    comparan en el nivel 4 y la entrada/salida (un archivo JSON por punto, escritura
    atómica, reanudación automática).

Casos (ρ = 1, θ = π/4, λ₁ = λ₂ = Λ/√2):
    odd3   sector impar, funcional reducido por paridad de Rabi, x = (η↑, r, κ)  [producción]
    odd6   sector impar, funcional general SIN imponer la simetría, 6 parámetros
    even6  sector par, funcional general, 6 parámetros                         [producción]
Vector general: x = (η↑, η↓, r↑, r↓, κ↑, κ↓).
"""

import os
import sys

# Un hilo por proceso: el paralelismo lo hace joblib (evita sobresuscripción).
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false '
                                   'intra_op_parallelism_threads=1')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MPLBACKEND', 'Agg')

# Salida en UTF-8 aunque la consola o la tubería de Windows usen cp1252 (Λ, η, −, …).
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import argparse
import importlib.util
import json
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')

HERE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get('VT_RESULTS', os.path.join(HERE, 'results'))   # VT_RESULTS: guardar en otra carpeta

# ══════════════════════════════════════════════════════════════════
#  SCRIPT DE PRODUCCIÓN
# ══════════════════════════════════════════════════════════════════
PROD_PATH = os.environ.get('VPAP_PROD',
                           os.path.join(HERE, '..', 'variational_P_AP.py'))
if not os.path.exists(PROD_PATH):
    raise FileNotFoundError(f'No encuentro el script de producción: {PROD_PATH}\n'
                            'Copia variational_tests/ dentro de variational_P_AP/ o '
                            'define la variable de entorno VPAP_PROD.')
_spec = importlib.util.spec_from_file_location('vpap_prod', PROD_PATH)
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)
warnings.filterwarnings('ignore')

omega0 = P.omega0

# ══════════════════════════════════════════════════════════════════
#  DIFERENCIACIÓN AUTOMÁTICA
# ══════════════════════════════════════════════════════════════════
BACKEND = None
try:
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    BACKEND = 'jax'
except Exception:                                    # pragma: no cover
    try:
        import autograd
        import autograd.numpy as anp
        from autograd import grad as ag_grad, hessian as ag_hess, jacobian as ag_jac
        BACKEND = 'autograd'
    except Exception:
        BACKEND = None


# ══════════════════════════════════════════════════════════════════
#  CASOS, MALLA, PARÁMETROS DE SECTOR
# ══════════════════════════════════════════════════════════════════
CASES = {'odd3': ('odd', 3), 'odd6': ('odd', 6), 'even6': ('even', 6)}
PNAMES = {3: ['eta_up', 'r', 'kap'],
          6: ['eta_up', 'eta_dn', 'r_up', 'r_dn', 'kap_up', 'kap_dn']}
KAP_IDX = {3: [2], 6: [4, 5]}

DEFAULT_J    = (0.5, 1.0, 1.5)
DEFAULT_LMIN = 0.05
DEFAULT_LMAX = 5.65
DEFAULT_DLAM = 0.10


def lam_grid(lmin=DEFAULT_LMIN, lmax=DEFAULT_LMAX, dlam=DEFAULT_DLAM):
    return np.round(np.arange(lmin, lmax + 1e-9, dlam), 6)


def sector_of(case):
    return CASES[case][0]


def npar(case):
    return CASES[case][1]


def block(case, J, Lam):
    """(ε, ε', J) del bloque, exactamente como en producción (δλ = 0)."""
    lam1, lam2 = P.Lam_to_lam12(Lam, 0.0)
    return P.sector_params(sector_of(case), lam1, lam2, P.eps1_val, P.eps2_val, J, J)


def bounds(case, Lam):
    return P._bounds(Lam, npar(case))


def lohi(case, Lam):
    b = bounds(case, Lam)
    return np.array([v[0] for v in b]), np.array([v[1] for v in b])


def seeds(case, Lam):
    return [np.asarray(s, float) for s in P._seeds(Lam, npar(case), sector_of(case))]


def to_full(case, x):
    x = np.asarray(x, float)
    return P._sym_to_full(x) if npar(case) == 3 else x.copy()


def energy_prod(case, J, Lam):
    """Funcional de PRODUCCIÓN (numpy), como función de x."""
    eps, epsp, Jb = block(case, J, Lam)
    if npar(case) == 3:
        return lambda x: float(P.energy_closed_sym(x, eps, epsp, Jb, Lam))
    return lambda x: float(P.energy_closed(x, eps, epsp, Jb, Lam))


# ══════════════════════════════════════════════════════════════════
#  FUNCIONAL CERRADO EN FORMA DIFERENCIABLE (mismas fórmulas que producción)
# ══════════════════════════════════════════════════════════════════
def _eps_xp(sector, Lam):
    l2w = 0.5*Lam**2*omega0                          # λ₁²ω0 = λ₂²ω0 = Λ²ω0/2
    if sector == 'odd':
        return P.eps1_val - l2w, P.eps2_val - l2w
    return P.eps1_val + P.eps2_val - 2.0*l2w, 0.0*Lam


def _E_general_xp(x, Lam, J, sector, xp):
    eps, epsp = _eps_xp(sector, Lam)
    eu, ed, ru, rd, ku, kd = x[0], x[1], x[2], x[3], x[4], x[5]
    Dl = Lam + ed - eu
    ch = xp.cosh(ru - rd)
    F  = xp.exp(-Dl**2/(xp.exp(-2.0*ru) + xp.exp(-2.0*rd)))/xp.sqrt(ch)
    su, sd = xp.sinh(ru), xp.sinh(rd)
    au = Dl*xp.exp(rd)*su/ch
    ad = Dl*xp.exp(ru)*sd/ch
    Gu = F*((eu - au)*(ed + Lam + ad) + su*sd/ch)
    Gd = F*((ed + ad)*(eu - Lam - au) + su*sd/ch)
    nu = 1.0 + ku*ku + 2.0*ku*F
    nd = 1.0 + kd*kd + 2.0*kd*F
    A = eps  + omega0*(eu*eu + su**2 + ku*ku*((ed + Lam)**2 + sd**2) + 2.0*ku*Gu)/nu
    B = epsp + omega0*(ed*ed + sd**2 + kd*kd*((eu - Lam)**2 + su**2) + 2.0*kd*Gd)/nd
    T = J*(ku + kd + (1.0 + ku*kd)*F)/xp.sqrt(nu*nd)
    return 0.5*(A + B) - xp.sqrt(0.25*(A - B)**2 + T*T)


def _E_sym_xp(xs, Lam, J, sector, xp):
    eps, _ = _eps_xp(sector, Lam)
    eta_n, r, kap = 2.0*xs[0], xs[1], xs[2]
    F   = xp.exp(-0.5*(Lam - eta_n)**2*xp.exp(2.0*r))
    sh2 = xp.sinh(r)**2
    den = 1.0 + kap**2 + 2.0*kap*F
    num = (omega0*(0.25*eta_n**2 + (1.0 + kap**2)*sh2 + kap**2*(Lam - 0.5*eta_n)**2
                   + 2.0*kap*F*(0.25*Lam**2 - 0.25*(Lam - eta_n)**2*xp.exp(4.0*r) + sh2))
           - xp.abs(J)*xp.abs(2.0*kap + (1.0 + kap**2)*F))
    return eps + num/den


class AD:
    """E, ∇E, H = ∇∇E, ∂_Λ∇E y E vectorizada, exactos (sin diferencias finitas)."""

    def __init__(self, case):
        if BACKEND is None:
            raise ImportError('Ni JAX ni autograd están instalados (python -m pip install jax).')
        sector, n = CASES[case]
        self.n = n
        if BACKEND == 'jax':
            core = _E_sym_xp if n == 3 else _E_general_xp
            f = lambda x, L, J: core(x, L, J, sector, jnp)
            self._E  = jax.jit(f)
            self._g  = jax.jit(jax.grad(f, argnums=0))
            self._H  = jax.jit(jax.hessian(f, argnums=0))
            self._gL = jax.jit(jax.jacfwd(jax.grad(f, argnums=0), argnums=1))
            self._Ev = jax.jit(jax.vmap(f, in_axes=(0, None, None)))
        else:
            core = _E_sym_xp if n == 3 else _E_general_xp
            f = lambda x, L, J: core(x, L, J, sector, anp)
            self._E  = f
            self._g  = ag_grad(f, 0)
            self._H  = ag_hess(f, 0)
            self._gL = lambda x, L, J: ag_jac(lambda LL: ag_grad(f, 0)(x, LL, J))(L)
            self._Ev = lambda X, L, J: np.array([f(xx, L, J) for xx in X])

    def E(self, x, Lam, J):
        return float(self._E(np.asarray(x, float), float(Lam), float(J)))

    def grad(self, x, Lam, J):
        return np.asarray(self._g(np.asarray(x, float), float(Lam), float(J)), float)

    def hess(self, x, Lam, J):
        H = np.asarray(self._H(np.asarray(x, float), float(Lam), float(J)), float)
        return 0.5*(H + H.T)

    def dgrad_dLam(self, x, Lam, J):
        return np.asarray(self._gL(np.asarray(x, float), float(Lam), float(J)), float)

    def Evec(self, X, Lam, J, batch=200000):
        X = np.asarray(X, float)
        out = np.empty(len(X))
        for i in range(0, len(X), batch):
            out[i:i+batch] = np.asarray(self._Ev(X[i:i+batch], float(Lam), float(J)))
        return out


_AD_CACHE = {}


def get_ad(case):
    if case not in _AD_CACHE:
        _AD_CACHE[case] = AD(case)
    return _AD_CACHE[case]


# ══════════════════════════════════════════════════════════════════
#  OPTIMALIDAD (KKT + HESSIANO)
# ══════════════════════════════════════════════════════════════════
def optimality(case, J, Lam, x, tol_bound=1e-9):
    """Gradiente proyectado (KKT con cotas), autovalores del hessiano en las variables
    libres, variables en cota."""
    ad = get_ad(case)
    x = np.asarray(x, float)
    lo, hi = lohi(case, Lam)
    g = ad.grad(x, Lam, J)
    H = ad.hess(x, Lam, J)
    at_lo = x <= lo + tol_bound
    at_hi = x >= hi - tol_bound
    kkt_ok_lo = ~at_lo | (g >= -1e-10)               # en la cota inferior, ∂E ≥ 0
    kkt_ok_hi = ~at_hi | (g <= 1e-10)
    free = ~(at_lo | at_hi)
    pg = g.copy()
    pg[at_lo & (g > 0)] = 0.0
    pg[at_hi & (g < 0)] = 0.0
    if free.any():
        ev = np.linalg.eigvalsh(H[np.ix_(free, free)])
    else:
        ev = np.array([np.inf])
    evf = np.linalg.eigvalsh(H)
    return dict(grad=g.tolist(), projgrad=float(np.max(np.abs(pg))),
                kkt_sign_ok=bool(np.all(kkt_ok_lo & kkt_ok_hi)),
                at_lo=at_lo.tolist(), at_hi=at_hi.tolist(),
                eig_free=ev.tolist(), lam_min=float(ev.min()), lam_max=float(ev.max()),
                n_neg=int(np.sum(ev < -1e-10)), eig_full=evf.tolist())


# ══════════════════════════════════════════════════════════════════
#  CANTIDADES FÍSICAS
# ══════════════════════════════════════════════════════════════════
_EXACT_CACHE = {}


def exact(case, J, Lam):
    """Estado base exacto (producción: D(Λ) analítico en Fock) y D(Λ)."""
    key = (sector_of(case), float(J), float(Lam))
    if key not in _EXACT_CACHE:
        eps, epsp, Jb = block(case, J, Lam)
        M = P._Mof_Lam(Lam)
        E0, psi = P.exact_ground(eps, epsp, Jb, Lam, M)
        DL = P.disp_matrix(M, Lam)
        n1 = M + 1
        o_ex = 2.0*abs(psi[:n1] @ DL @ psi[n1:])
        _EXACT_CACHE[key] = dict(E0=float(E0), psi=psi, M=M, DL=DL, o_ex=float(o_ex))
        if len(_EXACT_CACHE) > 8:
            _EXACT_CACHE.pop(next(iter(_EXACT_CACHE)))
    return _EXACT_CACHE[key]


def _beta(o):
    o = abs(o)
    return float(np.sqrt(max(-0.5*np.log(max(o, 1e-300)), 0.0)))


def physics(case, J, Lam, x):
    """Cantidades que el nivel 4 compara entre métodos."""
    x  = np.asarray(x, float)
    xf = to_full(case, x)
    eps, epsp, Jb = block(case, J, Lam)
    E_case = energy_prod(case, J, Lam)(x)
    A, B, C = P.closed_ABC(xf, eps, epsp, Jb, Lam)
    c1, c2 = P.weights_2x2(A, B, C)
    ex  = exact(case, J, Lam)
    ops = P.get_ops(ex['M'])
    pu, pd = P.var_states(xf, Lam, ops)
    v = np.concatenate([c1*pu, c2*pd])
    fid  = float(abs(ex['psi'] @ v)**2)
    bdag = float(pu @ (ops.b.T @ pu))                # <φ↑| b† |φ↑>
    o_var = float(abs(2.0*c1*c2*C/Jb))               # 2 c↑ c↓ <φ↑|D(Λ)|φ↓>
    rec = dict(E=E_case, E_ex=ex['E0'], relE=float((E_case - ex['E0'])/abs(ex['E0'])),
               fid=fid, infid=1.0 - fid, bdag_up=bdag, c_up=float(c1), c_dn=float(c2),
               o_var=o_var, o_ex=ex['o_ex'],
               p_plus=0.5*(1 + o_var), p_minus=0.5*(1 - o_var),
               beta_var=_beta(o_var), beta_ex=_beta(ex['o_ex']),
               leak=float(max(np.sum(pu[-5:]**2), np.sum(pd[-5:]**2))),
               x_full=xf.tolist())
    if npar(case) == 6:
        rec['asym'] = float(abs(xf[0] + xf[1]) + abs(xf[2] - xf[3]) + abs(xf[4] - xf[5]))
    return rec


def full_record(case, J, Lam, x, **extra):
    rec = dict(case=case, J=float(J), Lam=float(Lam), x=np.asarray(x, float).tolist())
    rec.update(physics(case, J, Lam, x))
    if BACKEND is not None:
        rec.update(optimality(case, J, Lam, x))
    rec.update(extra)
    return rec


# ══════════════════════════════════════════════════════════════════
#  ENTRADA / SALIDA
# ══════════════════════════════════════════════════════════════════
def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def point_path(level, case, J, Lam, tag):
    d = os.path.join(RESULTS, level, case, f'J{J:.3f}')
    return os.path.join(d, f'L{Lam:08.4f}__{tag}.json')


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f'.tmp{os.getpid()}'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, default=_json_default)
    os.replace(tmp, path)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def exists(path):
    if not os.path.exists(path):
        return False
    try:
        load_json(path)
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════
#  LÍNEA DE COMANDOS Y PARALELISMO
# ══════════════════════════════════════════════════════════════════
def base_parser(desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument('--cases', default='odd3,odd6,even6',
                    help='casos separados por coma: odd3, odd6, even6')
    ap.add_argument('--J', default=','.join(str(j) for j in DEFAULT_J), help='J/ω0, separados por coma')
    ap.add_argument('--lmin', type=float, default=DEFAULT_LMIN)
    ap.add_argument('--lmax', type=float, default=DEFAULT_LMAX)
    ap.add_argument('--dlam', type=float, default=DEFAULT_DLAM)
    ap.add_argument('--lams', default='', help='lista explícita de Λ (anula lmin/lmax/dlam)')
    ap.add_argument('--cores', type=int, default=int(os.environ.get('VT_CORES', os.cpu_count() or 1)))
    ap.add_argument('--chunk', default='1/1', help='i/n: procesa solo el bloque i de n')
    ap.add_argument('--overwrite', action='store_true', help='recalcula aunque exista el archivo')
    ap.add_argument('--quick', action='store_true', help='prueba rápida: pocos Λ, un J')
    return ap


def parse_common(args):
    cases = [c.strip() for c in args.cases.split(',') if c.strip()]
    for c in cases:
        if c not in CASES:
            raise SystemExit(f'caso desconocido: {c}')
    Js = [float(j) for j in args.J.split(',') if j.strip()]
    if args.lams:
        lams = np.round(np.array([float(v) for v in args.lams.split(',')]), 6)
    else:
        lams = lam_grid(args.lmin, args.lmax, args.dlam)
    if args.quick:
        Js = Js[1:2] if len(Js) > 1 else Js
        lams = lams[np.linspace(0, len(lams) - 1, 5).astype(int)]
    i, n = (int(v) for v in args.chunk.split('/'))
    return cases, Js, lams, (i, n)


def take_chunk(tasks, chunk):
    i, n = chunk
    return tasks[i - 1::n]


def run_parallel(func, tasks, cores, label):
    from joblib import Parallel, delayed
    t0 = time.time()
    print(f'[{label}] {len(tasks)} tareas en {cores} núcleos  (backend AD: {BACKEND})', flush=True)
    if not tasks:
        return []
    if cores == 1:
        out = [func(*t) for t in tasks]
    else:
        out = Parallel(n_jobs=cores, backend='loky', verbose=5)(delayed(func)(*t) for t in tasks)
    print(f'[{label}] listo en {time.time() - t0:.0f} s', flush=True)
    return out


# ══════════════════════════════════════════════════════════════════
#  AUTOPRUEBA
# ══════════════════════════════════════════════════════════════════
def selftest(verbose=True):
    """Funcional AD = producción; ∇E y ∂_Λ∇E de AD = diferencias finitas; Fock = cerrado."""
    rng = np.random.default_rng(0)
    worst = dict(E=0.0, g=0.0, gL=0.0, fock=0.0)
    for case in CASES:
        ad = get_ad(case)
        for J in DEFAULT_J:
            for Lam in (0.3, 1.7, 3.2, 5.1):
                lo, hi = lohi(case, Lam)
                fp = energy_prod(case, J, Lam)
                for _ in range(6):
                    x = lo + (hi - lo)*rng.random(len(lo))
                    worst['E'] = max(worst['E'], abs(ad.E(x, Lam, J) - fp(x)))
                    h = 1e-6
                    gfd = np.array([(fp(x + h*e) - fp(x - h*e))/(2*h) for e in np.eye(len(x))])
                    worst['g'] = max(worst['g'], np.max(np.abs(ad.grad(x, Lam, J) - gfd)))
                    gLfd = (ad.grad(x, Lam + h, J) - ad.grad(x, Lam - h, J))/(2*h)
                    worst['gL'] = max(worst['gL'], np.max(np.abs(ad.dgrad_dLam(x, Lam, J) - gLfd)))
                x = seeds(case, Lam)[1]
                xf = to_full(case, x)
                eps, epsp, Jb = block(case, J, Lam)
                ops = P.get_ops(P._Mof_Lam(Lam))
                worst['fock'] = max(worst['fock'],
                                    abs(P.energy_fock(xf, eps, epsp, Jb, Lam, ops) - fp(x)))
    ok = worst['E'] < 1e-11 and worst['g'] < 1e-5 and worst['gL'] < 1e-5 and worst['fock'] < 1e-8
    if verbose:
        print(f'[selftest] backend AD = {BACKEND}   producción = {os.path.abspath(PROD_PATH)}')
        print(f'[selftest] máx|E_AD − E_prod|        = {worst["E"]:.2e}')
        print(f'[selftest] máx|∇E_AD − ∇E_FD|        = {worst["g"]:.2e}   (FD central, h = 1e-6)')
        print(f'[selftest] máx|∂Λ∇E_AD − FD|         = {worst["gL"]:.2e}')
        print(f'[selftest] máx|E_Fock − E_cerrada|   = {worst["fock"]:.2e}')
        print('[selftest] ' + ('OK' if ok else 'FALLÓ'))
    return ok


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        sys.exit(0 if selftest() else 1)
    print(__doc__)
