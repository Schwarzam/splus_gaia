import numpy as np
import pandas as pd

# ---------- Configuration ----------
W_nb_J0660 = 147.0  # Å, S-PLUS J0660 effective width
LAMBDA = {
    'u': 3536.0, 'J0378': 3770.0, 'J0395': 3940.0, 'J0410': 4094.0, 'J0430': 4292.0,
    'g': 4751.0, 'J0515': 5133.0, 'r': 6258.0, 'J0660': 6614.0, 'i': 7690.0,
    'J0861': 8611.0, 'z': 8831.0
}
LAM_HA = 6563.0

# Bands to use for continuum fit (exclude the Hα narrow-band!)
CONTINUUM_BANDS = ['u','J0378','J0395','J0410','J0430','g','J0515','r','i','J0861','z']

# Column name templates from your DataFrame
MAG_COL = {
    'u': 'mag_pstotal_u',   'g': 'mag_pstotal_g',   'r': 'mag_pstotal_r',   'i': 'mag_pstotal_i',   'z': 'mag_pstotal_z',
    'J0378': 'mag_pstotal_j0378', 'J0395': 'mag_pstotal_j0395', 'J0410': 'mag_pstotal_j0410',
    'J0430': 'mag_pstotal_j0430', 'J0515': 'mag_pstotal_j0515', 'J0660': 'mag_pstotal_j0660', 'J0861': 'mag_pstotal_j0861'
}
ERR_COL = {
    'u': 'err_mag_pstotal_u','g': 'err_mag_pstotal_g','r': 'err_mag_pstotal_r','i': 'err_mag_pstotal_i','z': 'err_mag_pstotal_z',
    'J0378': 'err_mag_pstotal_j0378','J0395': 'err_mag_pstotal_j0395','J0410': 'err_mag_pstotal_j0410',
    'J0430': 'err_mag_pstotal_j0430','J0515': 'err_mag_pstotal_j0515','J0660': 'err_mag_pstotal_j0660','J0861': 'err_mag_pstotal_j0861'
}

BANDS = ['u','J0378','J0395','J0410','J0430','g','J0515','r','J0660','i','J0861','z']


# ---------- Helpers ----------
def mags_to_flux_ln(m, dm):
    """
    Convert AB magnitudes to natural-log flux (arbitrary units) and its uncertainty.
    ln f = -0.4 * ln(10) * m + C ; sigma_ln f = (ln(10)/2.5) * dm
    The additive constant C cancels in ratios, so we ignore it.
    """
    k = np.log(10)/2.5
    ln_f = -0.4*np.log(10)*m
    s_ln_f = k*dm
    return ln_f, s_ln_f

def weighted_poly_fit_lnflux(lam, ln_f, s_ln_f, lam0=LAM_HA, deg=2, max_iter=3, sigma_clip=3.0):
    """
    Fit ln(f) = beta0 + beta1*t + beta2*t^2 (deg=2) with weights 1/sigma^2, where t = (lam - lam0)/1000.
    Iterative sigma-clipping on residuals in ln(f).
    Returns beta, Cov(beta), mask_used (bool array), and a flag 'ok'.
    """
    lam = np.asarray(lam); ln_f = np.asarray(ln_f); s_ln_f = np.asarray(s_ln_f)
    t = (lam - lam0)/1000.0
    Xfull = np.vstack([np.ones_like(t), t, t**2 if deg >= 2 else np.zeros_like(t)]).T[:, :deg+1]

    mask = np.isfinite(ln_f) & np.isfinite(s_ln_f) & np.isfinite(t) & (s_ln_f > 0)
    ok = False
    beta = None; Cov = None

    for _ in range(max_iter):
        if mask.sum() < (deg+2):  # need at least deg+2 points for a stable fit
            break
        X = Xfull[mask]
        y = ln_f[mask]
        w = 1.0 / (s_ln_f[mask]**2)
        # Weighted least squares: (X^T W X) beta = X^T W y
        XT_W = X.T * w
        A = XT_W @ X
        b = XT_W @ y
        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break

        # Residuals and robust sigma
        yhat = X @ beta
        res = y - yhat
        # Weighted residual variance estimate
        dof = max(1, mask.sum() - (deg+1))
        s2 = (w * res**2).sum() / w.sum()  # conservative
        s = np.sqrt(s2)
        new_mask = mask.copy()
        # sigma-clip on absolute residuals in ln(f)
        new_mask[np.where(mask)[0][np.abs(res) > sigma_clip * s]] = False
        if new_mask.sum() == mask.sum():
            # Converged — compute covariance with classic WLS formula
            # Use unweighted residual variance estimate s2_hat:
            s2_hat = (res**2).sum() / dof
            Cov = np.linalg.inv(A) * s2_hat
            ok = True
            break
        mask = new_mask

    return beta, Cov, mask, ok

def predict_lnflux_at(beta, Cov, lam_eval, lam0=LAM_HA):
    """Predict ln(f) and its variance at lam_eval from polynomial coefficients."""
    t0 = (lam_eval - lam0)/1000.0
    # Feature vector x0
    x0 = np.array([1.0, t0, t0**2])
    x0 = x0[:len(beta)]
    ln_f_hat = float(x0 @ beta)
    var_ln_f_hat = float(x0 @ Cov @ x0)
    return ln_f_hat, var_ln_f_hat

def ew_from_fluxes(f_nb, df_nb, f_c, df_c, W_nb=W_nb_J0660):
    R = f_nb / f_c
    dR = R * np.sqrt((df_nb/np.maximum(f_nb, 1e-300))**2 + (df_c/np.maximum(f_c, 1e-300))**2)
    EW = W_nb * (R - 1.0)
    dEW = W_nb * dR
    return EW, dEW

def ri_fallback(df_row):
    """Robust r–i linear interpolation fallback in flux (not ln-flux)."""
    # flux and errors (linear space) for r, i, J0660
    def mag_to_flux(m, dm):
        f = 10**(-0.4*m)
        df = (np.log(10)/2.5) * f * dm
        return f, df

    fr, dfr = mag_to_flux(df_row[MAG_COL['r']],  df_row[ERR_COL['r']])
    fi, dfi = mag_to_flux(df_row[MAG_COL['i']],  df_row[ERR_COL['i']])
    fnb, dfnb = mag_to_flux(df_row[MAG_COL['J0660']], df_row[ERR_COL['J0660']])

    lam_r, lam_i = LAMBDA['r'], LAMBDA['i']
    wr = (lam_i - LAM_HA) / (lam_i - lam_r)
    wi = 1.0 - wr
    fc = wr*fr + wi*fi
    dfc = np.sqrt((wr*dfr)**2 + (wi*dfi)**2)

    EW, dEW = ew_from_fluxes(fnb, dfnb, fc, dfc)
    snr = EW / np.maximum(dEW, 1e-30)
    return EW, dEW, snr

# ---------- Main function ----------
def detect_halpha_emitters(df,
                           ew_min=10.0, snr_min=3.0,
                           deg=2, max_iter=3, sigma_clip=3.0):
    """
    Compute EW(Hα) from S-PLUS photometry with a continuum fit (excluding J0660).
    Falls back to r–i interpolation if WLS fit is not possible for a source.

    Returns:
      results: DataFrame with ['EW_Ha','EW_Ha_err','SNR_EW','method'] aligned to df.index
      mask_emitters: boolean Series where selection passes (EW>ew_min & SNR>=snr_min)
    """
    # Pre-extract arrays for speed
    # ln-flux and uncertainties for all bands we use
    lnF = {}
    s_lnF = {}
    for b in CONTINUUM_BANDS + ['J0660']:
        m = df[MAG_COL[b]].values.astype(float)
        dm = df[ERR_COL[b]].values.astype(float)
        lnF[b], s_lnF[b] = mags_to_flux_ln(m, dm)

    # Build per-row arrays for continuum bands
    lam_all = np.array([LAMBDA[b] for b in CONTINUUM_BANDS], dtype=float)
    lnF_all = np.vstack([lnF[b] for b in CONTINUUM_BANDS]).T
    s_lnF_all = np.vstack([s_lnF[b] for b in CONTINUUM_BANDS]).T

    # J0660 (narrow-band)
    lnF_nb = lnF['J0660']
    s_lnF_nb = s_lnF['J0660']

    N = len(df)
    EW = np.full(N, np.nan)
    dEW = np.full(N, np.nan)
    SNR = np.full(N, np.nan)
    method = np.empty(N, dtype=object)

    # Per-object fit
    for i in range(N):
        lam = lam_all.copy()
        ln_f = lnF_all[i].copy()
        s_ln_f = s_lnF_all[i].copy()

        # Drop any NaNs per-row
        ok_mask = np.isfinite(ln_f) & np.isfinite(s_ln_f)
        lam_i = lam[ok_mask]; ln_f_i = ln_f[ok_mask]; s_ln_f_i = s_ln_f[ok_mask]

        beta, Cov, used_mask, ok = weighted_poly_fit_lnflux(
            lam_i, ln_f_i, s_ln_f_i, lam0=LAM_HA, deg=deg, max_iter=max_iter, sigma_clip=sigma_clip
        )

        # Narrow-band in linear flux:
        # f_nb = exp(ln f), df_nb via error in ln(f): sigma_f = f * sigma_ln f
        f_nb = np.exp(lnF_nb[i])
        df_nb = f_nb * s_lnF_nb[i]

        if ok:
            ln_fc, var_ln_fc = predict_lnflux_at(beta, Cov, LAM_HA, lam0=LAM_HA)
            f_c = np.exp(ln_fc)
            df_c = f_c * np.sqrt(max(var_ln_fc, 0.0))
            EW[i], dEW[i] = ew_from_fluxes(f_nb, df_nb, f_c, df_c, W_nb=W_nb_J0660)
            SNR[i] = EW[i]/np.maximum(dEW[i], 1e-30)
            method[i] = 'WLS-continuum'
        else:
            # Fallback to r-i interpolation
            ew_fb, dew_fb, snr_fb = ri_fallback(df.iloc[i])
            EW[i], dEW[i], SNR[i] = ew_fb, dew_fb, snr_fb
            method[i] = 'r-i-fallback'

    results = pd.DataFrame({
        'EW_Ha': EW,
        'EW_Ha_err': dEW,
        'SNR_EW': SNR,
        'method': method
    }, index=df.index)

    # Selection
    emitters = (results['EW_Ha'] > ew_min) & (results['SNR_EW'] >= snr_min)

    return results, emitters

# ---------- Example usage ----------
# results, emitters = detect_halpha_emitters(df, ew_min=10.0, snr_min=3.0)
# emitters_df = df[emitters].join(results)

dtypes = {
    "id": "string",
    "ra": "float32",
    "dec": "float32",
    "parallax": "float32",
    "gaia_ruwe": "float32",
    "gaia_parallax_over_error": "float32",
    "gaia_classprob_dsc_combmod_star": "float32",
    "gaia_in_qso_candidates": "int8",
    "gaia_in_galaxy_candidates": "int8",
    "gaia_phot_bp_rp_excess_factor": "float32",
    "mag_pstotal_r": "float32",
    "mag_pstotal_i": "float32",
    "mag_pstotal_u": "float32",
    "mag_pstotal_g": "float32",
    "mag_pstotal_z": "float32",
    "mag_pstotal_j0378": "float32",
    "mag_pstotal_j0395": "float32",
    "mag_pstotal_j0410": "float32",
    "mag_pstotal_j0430": "float32",
    "mag_pstotal_j0515": "float32",
    "mag_pstotal_j0660": "float32",
    "mag_pstotal_j0861": "float32",
    "err_mag_pstotal_r": "float32",
    "err_mag_pstotal_i": "float32",
    "err_mag_pstotal_u": "float32",
    "err_mag_pstotal_g": "float32",
    "err_mag_pstotal_z": "float32",
    "err_mag_pstotal_j0378": "float32",
    "err_mag_pstotal_j0395": "float32",
    "err_mag_pstotal_j0410": "float32",
    "err_mag_pstotal_j0430": "float32",
    "err_mag_pstotal_j0515": "float32",
    "err_mag_pstotal_j0660": "float32",
    "err_mag_pstotal_j0861": "float32",
}

print("Loading data...")
df = pd.read_csv(
    "/mnt/hdcasa/splus_gaia/oficial/SPLUS-s.csv",
    engine="pyarrow",
    dtype=dtypes,
    usecols=list(dtypes.keys())
)

print("Filtering data...")
df = df[(df['mag_pstotal_r'] > 12) & (df['mag_pstotal_r'] < 22)]

print("Selecting H-alpha emitters...")
results, emitters = detect_halpha_emitters(df, ew_min=30.0, snr_min=3.0)
emitters_df = df[emitters].join(results)

print("Applying final cuts...")
emitters_df = emitters_df[emitters_df['mag_pstotal_j0660'] < emitters_df['mag_pstotal_i']]

print("Saving results...")
emitters_df.to_csv("../data/halpha_emitters/SPLUS-s2.csv", index=False)