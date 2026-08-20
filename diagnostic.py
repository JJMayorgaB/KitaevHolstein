import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from marginal_negativities import (diagonalize_LF_sweet, exact_cat_bimodal,
                                   pad_state, MofLam_sweet_exact)

omega0  = 1.0
cv_val  = 1.0
Lam_fix = 3*np.sqrt(2)
th      = 1.047          # el ángulo donde E_N=1.17
nph_wig = 85

g1 = Lam_fix*np.cos(th)
g2 = Lam_fix*np.sin(th)
M  = min(MofLam_sweet_exact(g1, g2, omega0=omega0, M_cap=45) + 25, 55)

ev_e, evec_e, ev_o, evec_o, nph, n_ph = diagonalize_LF_sweet(
    M, cv_val, cv_val, omega0, omega0, g1, g2)
psi_ph = exact_cat_bimodal(g1, g2, evec_e[:, 0], nph, n_ph, omega0,
                           sign=+1, sector='even')

psi_ph = pad_state(psi_ph, nph, nph_wig)
pm = psi_ph.reshape(nph_wig, nph_wig)
rho1 = pm @ pm.conj().T

ev = np.sort(np.linalg.eigvalsh(rho1).real)[::-1]
print("M =", M, " nph =", nph)
print("traza rho1 =", ev.sum())
print("10 mayores autovalores de rho1:")
for i, v in enumerate(ev[:10]):
    print(f"  lam_{i} = {v:.6e}")
print("suma de los que superan 1e-12:", np.sum(ev > 1e-12), "componentes")