import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from scipy.special import factorial, eval_genlaguerre
from scipy.optimize import minimize
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, eigsh
from joblib import Parallel, delayed
import contextlib, joblib

import warnings
from math import comb as _C
warnings.filterwarnings('ignore')
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter


def get_M_CM(lam1, lam2, M_max=12):
    """Truncación adaptativa. El desplazamiento efectivo en los modos
    relativo/CoM es Λ = sqrt(λ₁²+λ₂²), por eso los umbrales se ajustan a Λ."""
    Lam = np.sqrt(lam1**2 + lam2**2)
    if   Lam < 0.5: return 5
    elif Lam < 1.5: return 9
    else:           return M_max


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
    """Elementos de matriz <m'|D(alpha)|n> en base de Fock, alpha complejo.
    D(alpha) = exp(alpha a^dag - alpha* a)
    """
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


def MofLam_sweet_exact(g1, g2, omega0=1.0, n_sigma=4, floor=15, M_cap=55):
    lam_max = max(abs(g1), abs(g2)) / omega0
    M = int(np.ceil(lam_max**2 + n_sigma*lam_max)) 
    M = max(M, floor)          # piso para λ pequeño
    return min(M, M_cap)

def build_LF_hamiltonians_sweet(M, t, Delta, omega1, omega2, g1, g2):
    """
    Hamiltoniano LF en base de SITIO, en el SWEET SPOT (ε̃₁=ε̃₂=0).
    En el sweet spot ε_i = g_i²/ω_i, así que ε_i_eff = ε_i - g_i²/ω_i = 0
    y los bloques electrónicos no llevan offset diagonal.
    """
    nph  = M + 1
    n_ph = nph * nph
    lam1 = g1 / omega1
    lam2 = g2 / omega2

    D1p = disp_matrix(M,  lam1)
    D1m = disp_matrix(M, -lam1)
    D2p = disp_matrix(M,  lam2)
    D2m = disp_matrix(M, -lam2)

    ph_E = np.array([m1*omega1 + m2*omega2
                     for m1 in range(nph) for m2 in range(nph)])

    T_hop  = t     * np.kron(D1p, D2m)
    T_pair = Delta * np.kron(D1m, D2m)

    # Sector par (ε̃₁=ε̃₂=0 → ambos bloques con solo energía fotónica)
    H_e = np.zeros((2*n_ph, 2*n_ph))
    H_e[:n_ph, :n_ph] = np.diag(ph_E)
    H_e[n_ph:, n_ph:] = np.diag(ph_E)
    H_e[:n_ph, n_ph:] = T_pair
    H_e[n_ph:, :n_ph] = T_pair.T

    # Sector impar (ε̃₁=ε̃₂=0)
    H_o = np.zeros((2*n_ph, 2*n_ph))
    H_o[:n_ph, :n_ph] = np.diag(ph_E)
    H_o[n_ph:, n_ph:] = np.diag(ph_E)
    H_o[:n_ph, n_ph:] = T_hop
    H_o[n_ph:, :n_ph] = T_hop.T

    return H_e, H_o, nph, n_ph


def diagonalize_LF_sweet(M, t, Delta, omega1, omega2, g1, g2):
    H_e, H_o, nph, n_ph = build_LF_hamiltonians_sweet(
        M, t, Delta, omega1, omega2, g1, g2)
    ev_e, evec_e = eigh(H_e)
    ev_o, evec_o = eigh(H_o)
    return ev_e, evec_e, ev_o, evec_o, nph, n_ph


def ground_state_LF_sweet(M, t, Delta, omega1, omega2, g1, g2, sector='odd'):
    """Estado base de UN sector, el mismo Hamiltoniano de build_LF_hamiltonians_sweet,
    con Lanczos (eigsh) y sin matrices densas: el bloque fuera de la diagonal es
    T = A⊗B y se aplica como (A⊗B) vec(X) = vec(A X Bᵀ)  (idx = m1*nph + m2).
      par:   T_pair = Δ D1(−λ1)⊗D2(−λ2)        impar: T_hop = t D1(λ1)⊗D2(−λ2)
    Memoria O(nph²) en lugar de las matrices de (2 nph²)² de diagonalize_LF_sweet.
    Devuelve (E0, vec0, nph, n_ph), vec0 = [φ1, φ2] como la columna 0 de eigh."""
    nph  = M + 1
    n_ph = nph * nph
    lam1 = g1 / omega1
    lam2 = g2 / omega2
    if sector == 'even':
        A = Delta * disp_matrix(M, -lam1); B = disp_matrix(M, -lam2)
    else:
        A = t * disp_matrix(M, lam1);      B = disp_matrix(M, -lam2)
    ph_E = np.array([m1*omega1 + m2*omega2
                     for m1 in range(nph) for m2 in range(nph)])

    def _mv(v):
        v  = np.asarray(v).ravel()
        p1 = v[:n_ph]; p2 = v[n_ph:]
        top = ph_E*p1 + (A @ p2.reshape(nph, nph) @ B.T).ravel()      # diag·φ1 + T φ2
        bot = ph_E*p2 + (A.T @ p1.reshape(nph, nph) @ B).ravel()      # Tᵀ φ1 + diag·φ2
        return np.concatenate([top, bot])

    H  = LinearOperator((2*n_ph, 2*n_ph), matvec=_mv, dtype=float)
    v0 = np.ones(2*n_ph) / np.sqrt(2*n_ph)
    ev, vec = eigsh(H, k=1, which='SA', tol=0.0, v0=v0, ncv=40, maxiter=100000)
    return ev[0], vec[:, 0], nph, n_ph

def wigner_joint_cut_exact(psi_ph, nph, uvec, vvec, plane='real', verbose=True, tag=''):
    """
    Wigner conjunta de dos modos W(α1,α2) en un corte 2D, para un
    vector de Fock de dos modos (dim nph², idx = m1*nph + m2).
      plane='real': Re(α1)=u vs Re(α2)=v   (Im=0)
      plane='imag': Im(α1)=u vs Im(α2)=v   (Re=0)
    W = (2/π)² Σ_{n1,n2}(-1)^{n1+n2}|<n1 n2|D1†(α1)⊗D2†(α2)|ψ>|²
    Con ψ como matriz [n1,n2]: D1† por la izquierda, D2†^T por la derecha.
    """
    psi_mat = psi_ph.reshape(nph, nph)          # [m1, m2]
    par = (-1.0) ** np.arange(nph)
    Par = np.outer(par, par)

    def _Ddag(val):
        a = (val + 0j) if plane == 'real' else (1j*val)
        return disp_matrix_complex(nph-1, -a)   # D†(α) = D(-α)

    Ddag_u = [_Ddag(u) for u in uvec]
    Ddag_vT = [_Ddag(v).T for v in vvec]

    pref = (2.0/np.pi)**2
    nu = len(uvec)
    step10 = max(1, nu // 10)

    W = np.zeros((len(vvec), len(uvec)))
    for iu in range(nu):
        left = Ddag_u[iu] @ psi_mat             # [n1', m2]
        for iv in range(len(vvec)):
            tpsi = left @ Ddag_vT[iv]           # [n1', n2']
            W[iv, iu] = pref * np.sum(Par * (np.abs(tpsi)**2))
        if verbose and ((iu+1) % step10 == 0 or (iu+1) == nu):
            print(f"      [{tag}] {100.0*(iu+1)/nu:5.1f}%", flush=True)
    return W


def wigner_joint_and_cuts_exact(psi_ph, nph, uvec, vvec, plane='real', verbose=True, tag=''):
    W = wigner_joint_cut_exact(psi_ph, nph, uvec, vvec, plane=plane, verbose=verbose, tag=tag)
    iv0 = np.argmin(np.abs(vvec))
    iu0 = np.argmin(np.abs(uvec))
    return W, W[iv0, :], W[:, iu0]

def reduced_dm_mode_exact(psi_ph, nph, mode):
    """Matriz densidad reducida de un modo, desde el vector de Fock de dos modos.
       psi_ph: vector dim nph² (idx = m1*nph + m2).
       mode=1 → ρ1 = Tr_2 (contrae m2);  mode=2 → ρ2 = Tr_1 (contrae m1).
       Devuelve matriz nph×nph."""
    psi_mat = psi_ph.reshape(nph, nph)          # [m1, m2]
    if mode == 1:
        rho = psi_mat @ psi_mat.conj().T        # ρ1[m1,m1'] = Σ_m2 ψ[m1,m2]ψ*[m1',m2]
    else:
        rho = psi_mat.T @ psi_mat.conj()        # ρ2[m2,m2'] = Σ_m1 ψ[m1,m2]ψ*[m1,m2']
    return rho


def entanglement_measures_pure(rho1, tol=1e-12):
    """Para un estado bipartito PURO: negatividad logarítmica, entropía de
       von Neumann y rango efectivo, todas del espectro de Schmidt de ρ1.
         E_N = log2 ||ρ^{T_2}||_1 = 2 log2( Σ_i sqrt(λ_i) )
         S   = -Σ_i λ_i log2 λ_i
       Nota: en la partición de SITIO el rango no está acotado a 2 —la rotación
       desde los modos colectivos reparte ambos modos entre las cavidades—, de
       modo que E_N y S pueden exceder 1 ebit."""
    ev = np.linalg.eigvalsh(rho1).real
    ev = ev[ev > tol]
    E_N = 2.0 * np.log2(np.sum(np.sqrt(ev)))
    S   = -np.sum(ev * np.log2(ev))
    rank_eff = int(np.sum(ev > 1e-4))
    return E_N, S, rank_eff


def wigner_dm_negativity(rho, xvec, pvec, verbose=False, n_jobs=-1):
    """Wigner de un modo de una matriz densidad ρ (nph×nph) + su negatividad.
       W(α) = (2/π) Σ_n (-1)^n <n|D†(α) ρ D(α)|n>
            = (2/π) Σ_n (-1)^n [D†(α) ρ D(α)]_{nn}
       Vectorizado por punto: para cada α, calcula D(-α) (=D†(α)) y la forma
       cuadrática. Reusa disp_matrix_complex."""
    N = rho.shape[0]
    parity = (-1.0) ** np.arange(N)

    def _row(p):
        row = np.zeros(len(xvec))
        for ix, x in enumerate(xvec):
            alpha = (x + 1j*p)
            Dd = disp_matrix_complex(N-1, -alpha)      # D†(α) = D(-α)
            # [D† ρ D]_{nn} = (D† ρ D)diag ; D = Dd†
            M_ = Dd @ rho @ Dd.conj().T
            row[ix] = (2.0/np.pi) * np.sum(parity * np.diag(M_).real)
        return row

    if n_jobs == 1:
        W = np.array([_row(p) for p in pvec])
    else:
        import joblib
        if verbose:
            print(f"   Wigner reducida: {len(pvec)} filas en paralelo ...", flush=True)
        W = np.array(joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_row)(p) for p in pvec))

    if not hasattr(np, 'trapz'):
        np.trapz = np.trapezoid
    norm_check = np.trapz(np.trapz(W, xvec, axis=1), pvec)
    delta = 0.5 * (np.trapz(np.trapz(np.abs(W), xvec, axis=1), pvec) - 1.0)
    return delta, norm_check

def pad_state(psi_ph, nph_old, nph_new):
    """Expande el estado de dos modos de nph_old² a nph_new² con ceros en Fock altos."""
    pm = psi_ph.reshape(nph_old, nph_old)
    pm_new = np.zeros((nph_new, nph_new), dtype=complex)
    pm_new[:nph_old, :nph_old] = pm
    return pm_new.reshape(-1)


def marginal_negativities_exact(psi_ph, nph, xvec, pvec, n_jobs=-1):
    """δ1, δ2 marginales + E_N, S y rango efectivo. El estado llega ya con la
       dimensión final (el padding se hace en exact_cat_bimodal, antes de los
       desplazamientos)."""
    pm = psi_ph.reshape(nph, nph)
    rho1 = pm @ pm.conj().T
    rho2 = pm.T @ pm.conj()

    # entrelazamiento inter-cavidad (estado bipartito puro)
    E_N, S, rank_eff = entanglement_measures_pure(rho1)

    d1, n1 = wigner_dm_negativity(rho1, xvec, pvec, n_jobs=n_jobs)
    d2, n2 = wigner_dm_negativity(rho2, xvec, pvec, n_jobs=n_jobs)
    return d1, d2, n1, n2, E_N, S, rank_eff


def exact_cat_bimodal(g1, g2, evec_ex, nph, n_ph, omega0=1.0, sign=+1,
                      sector='even', nph_out=None):
    """Gato bimodal exacto en base de sitio.
       El padding a nph_out se hace ANTES de aplicar los desplazamientos, para
       que D(α) no trunque la cola del estado desplazado.
       Usa vec(A X B^T) = (A⊗B) vec(X) para evitar construir los kron.
       Devuelve (cat, nph_final)."""
    lam1 = g1 / omega0
    lam2 = g2 / omega0

    phi1 = evec_ex[:n_ph]
    phi2 = evec_ex[n_ph:]

    if nph_out is not None and nph_out > nph:
        phi1 = pad_state(phi1, nph, nph_out)
        phi2 = pad_state(phi2, nph, nph_out)
        nph  = nph_out

    D1p = disp_matrix(nph-1, +0.5*lam1); D1m = disp_matrix(nph-1, -0.5*lam1)
    D2p = disp_matrix(nph-1, +0.5*lam2); D2m = disp_matrix(nph-1, -0.5*lam2)

    P1 = phi1.reshape(nph, nph)
    P2 = phi2.reshape(nph, nph)

    if sector == 'even':          # φ1 = |00>: (+λ1/2, +λ2/2)   φ2 = |11>: (-λ1/2, -λ2/2)
        catm = (D1p @ P1 @ D2p.T) + (-sign)*(D1m @ P2 @ D2m.T)
    else:                         # φ2: (+λ1/2, -λ2/2)   φ1: (-λ1/2, +λ2/2)
        catm = (D1p @ P2 @ D2m.T) + (-sign)*(D1m @ P1 @ D2p.T)

    cat = catm.reshape(-1)
    cat /= np.linalg.norm(cat)
    return cat, nph

# ══════════════════════════════════════════════════════════════════
#  EJECUCIÓN — δ1, δ2, E_N y S vs θ (Λ fijo, EXACTO en base de sitio)
#    θ ∈ (0, π);  λ1 = Λcosθ, λ2 = Λsinθ  →  g_i = λ_i·ω0
#  Paralelizado por (Λ, θ): cada worker hace su diagonalización + Wigner en serie.
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    omega0   = 1.0
    cv_val   = 1.0
    LAM_LIST = [2*np.sqrt(2), 3*np.sqrt(2)]      # Λ fijos
    sector   = 'odd'
    SIGNS    = (+1, -1)           # ramas |Cat_+> (sign=+1) y |Cat_-> (sign=-1), mismo estado base
    nph_wig  = 85
    N_JOBS   = 128                # cores a usar
    DENSE_EIGH = False            # True: eigh denso de ambos sectores (diagonalize_LF_sweet, ~5 GB por job)
                                  # False: solo el estado base del sector usado (Lanczos, pocos MB por job)

    nth      = 64
    theta_arr = np.linspace(0.0, np.pi, nth)

    xvec = np.linspace(-4.2, 4.2, 251)
    pvec = np.linspace(-3.0, 3.0, 251)

    HERE   = os.path.dirname(os.path.abspath(__file__))
    OUTDIR = os.path.join(HERE, 'Figures', 'cat', 'data')
    TMPDIR = {(L, sg): os.path.join(OUTDIR, f'_tmp_exact_sign{sg:+d}_{L:.6f}')
              for L in LAM_LIST for sg in SIGNS}
    for d in TMPDIR.values():
        os.makedirs(d, exist_ok=True)

    def _load_tmp(f_tmp, th):
        if os.path.exists(f_tmp):
            row = np.loadtxt(f_tmp).ravel()
            if row.size == 8 and abs(row[0] - th) < 1e-6:      # θ guardado con 8 cifras
                return tuple(row[1:])
        return None

    def _one_theta(Lam_fix, k, th):
        # si ambas ramas ya están en el guardado incremental (relanzamiento), no se repite
        f_tmp = {sg: os.path.join(TMPDIR[(Lam_fix, sg)], f'{k:03d}.txt') for sg in SIGNS}
        done  = {sg: _load_tmp(f_tmp[sg], th) for sg in SIGNS}
        if all(v is not None for v in done.values()):
            return Lam_fix, k, done

        g1 = Lam_fix * np.cos(th)
        g2 = Lam_fix * np.sin(th)
        M = 75
        if DENSE_EIGH:
            ev_e, evec_e, ev_o, evec_o, nph, n_ph = diagonalize_LF_sweet(
                M, cv_val, cv_val, omega0, omega0, g1, g2)
            vec = evec_e[:, 0] if sector == 'even' else evec_o[:, 0]
        else:
            _, vec, nph, n_ph = ground_state_LF_sweet(
                M, cv_val, cv_val, omega0, omega0, g1, g2, sector=sector)

        for sign in SIGNS:
            if done[sign] is not None:
                continue
            # padding ANTES de los desplazamientos
            psi_ph, nph_f = exact_cat_bimodal(g1, g2, vec, nph, n_ph, omega0,
                                              sign=sign, sector=sector,
                                              nph_out=nph_wig)

            d1, d2, n1, n2, E_N, S, rk = marginal_negativities_exact(
                psi_ph, nph_f, xvec, pvec, n_jobs=1)   # serial adentro

            # guardado incremental: si el job muere, no se pierde lo avanzado
            np.savetxt(f_tmp[sign], np.array([[th, d1, d2, n1, n2, E_N, S, rk]]), fmt='%.8e')
            done[sign] = (d1, d2, n1, n2, E_N, S, rk)

            print(f"  Λ={Lam_fix:.4f}  sign={sign:+d}  θ={th:5.3f}  δ1={d1:.4f} δ2={d2:.4f}  "
                  f"E_N={E_N:.4f} S={S:.4f}  rank={rk}  (∫W1={n1:.3f}, ∫W2={n2:.3f})", flush=True)
        return Lam_fix, k, done

    tasks = [(L, k, th) for L in LAM_LIST for k, th in enumerate(theta_arr)]
    print(f'Tareas: {len(tasks)}   ({len(LAM_LIST)} Λ × {nth} θ), ramas {SIGNS} en cada una', flush=True)

    _res = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(_one_theta)(L, k, th) for L, k, th in tasks)

    for Lam_fix in LAM_LIST:
        for sign in SIGNS:
            _out = np.full((nth, 8), np.nan)
            _out[:, 0] = theta_arr
            for L, k, done in _res:
                if L == Lam_fix:
                    _out[k, 1:] = done[sign]

            fname = os.path.join(OUTDIR, f'marginal_negativity_vs_theta_exact_sign{sign:+d}_{Lam_fix}.txt')
            np.savetxt(fname, _out, fmt='%.8e',
                       header=(f'exacto, sector={sector}, sign={sign:+d}, Lambda={Lam_fix}\n'
                               'theta  delta1  delta2  norm1  norm2  E_N  S  rank_eff'))
            print(f'Guardado: {fname}')

    print("Barrido marginal θ (exacto) listo.")
