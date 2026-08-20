import os
import numpy as np
from scipy.special import factorial, eval_genlaguerre
import warnings
warnings.filterwarnings('ignore')


def coherent_vec(N, alpha):
    """Estado coherente |α> en base de Fock truncada a N niveles."""
    n = np.arange(N)
    logc = -0.5*abs(alpha)**2 + n*np.log(alpha + 0j) - 0.5*np.array(
        [np.sum(np.log(np.arange(1, k+1))) if k > 0 else 0.0 for k in n])
    v = np.exp(logc)
    return v / np.linalg.norm(v)


# ══════════════════════════════════════════════════════════════════
#  TEST DE CONTROL — gato de coherentes PUROS por el mismo pipeline
# ══════════════════════════════════════════════════════════════════
Lam_fix = 3*np.sqrt(2)
th      = 1.047
nph     = 100
sign    = +1

lam1 = Lam_fix*np.cos(th)
lam2 = Lam_fix*np.sin(th)

# mismos desplazamientos que usa exact_cat_bimodal
b1 = 0.5*lam1
b2 = 0.5*lam2
print(f"beta1 = {b1:.4f}   beta2 = {b2:.4f}")

# gato par:  |+b1,+b2> - sign |-b1,-b2>
u_p = coherent_vec(nph, +b1); v_p = coherent_vec(nph, +b2)
u_m = coherent_vec(nph, -b1); v_m = coherent_vec(nph, -b2)

catm = np.outer(u_p, v_p) + (-sign)*np.outer(u_m, v_m)
catm /= np.linalg.norm(catm)

rho1 = catm @ catm.conj().T
ev = np.sort(np.linalg.eigvalsh(rho1).real)[::-1]

print("traza rho1 =", ev.sum())
print("10 mayores autovalores:")
for i, v in enumerate(ev[:10]):
    print(f"  lam_{i} = {v:.6e}")
print("componentes > 1e-12:", np.sum(ev > 1e-12))
print("componentes > 1e-8: ", np.sum(ev > 1e-8))
print("componentes > 1e-4: ", np.sum(ev > 1e-4))

# E_N y S de este gato ideal
s = np.sqrt(ev[ev > 1e-12])
print(f"\nE_N = {2*np.log2(np.sum(s)):.4f}")
lam = s**2
print(f"S   = {-np.sum(lam*np.log2(lam)):.4f}")