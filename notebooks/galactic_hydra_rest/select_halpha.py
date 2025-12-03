import os
import numpy as np
import pandas as pd

# ==============================
# Config (adjust as needed)
# ==============================
INPUT_CSV  = "/mnt/hdcasa/splus_gaia/oficial/MC.csv"
OUTPUT_CSV = "../data/halpha_emitters-gustabin/MC.csv"
CHUNK_SIZE = 250_000               # tune to your RAM (e.g. 100k–1M)
WRITE_HEADER = True                # auto-managed; leave True

# ---------- Photometric config ----------
W_nb_J0660 = 147.0  # Å, S-PLUS J0660 effective width
LAMBDA = {
    'u': 3536.0, 'J0378': 3770.0, 'J0395': 3940.0, 'J0410': 4094.0, 'J0430': 4292.0,
    'g': 4751.0, 'J0515': 5133.0, 'r': 6258.0, 'J0660': 6614.0, 'i': 7690.0,
    'J0861': 8611.0, 'z': 8831.0
}
LAM_HA = 6563.0

# Bands to use for continuum fit (exclude the Hα narrow-band!)
CONTINUUM_BANDS = ['u','J0378','J0395','J0410','J0430','g','J0515','r','i','J0861','z']
BANDS = ['u','J0378','J0395','J0410','J0430','g','J0515','r','J0660','i','J0861','z']

# Column name templates from your DataFrame (we’ll convert pstotal -> psf)
MAG_COL = {
    'u': 'mag_pstotal_u',   'g': 'mag_pstotal_g',   'r': 'mag_pstotal_r',
    'i': 'mag_pstotal_i',   'z': 'mag_pstotal_z',
    'J0378': 'mag_pstotal_j0378', 'J0395': 'mag_pstotal_j0395', 'J0410': 'mag_pstotal_j0410',
    'J0430': 'mag_pstotal_j0430', 'J0515': 'mag_pstotal_j0515', 'J0660': 'mag_pstotal_j0660',
    'J0861': 'mag_pstotal_j0861'
}
ERR_COL = {
    'u': 'err_mag_pstotal_u','g': 'err_mag_pstotal_g','r': 'err_mag_pstotal_r',
    'i': 'err_mag_pstotal_i','z': 'err_mag_pstotal_z',
    'J0378': 'err_mag_pstotal_j0378','J0395': 'err_mag_pstotal_j0395','J0410': 'err_mag_pstotal_j0410',
    'J0430': 'err_mag_pstotal_j0430','J0515': 'err_mag_pstotal_j0515','J0660': 'err_mag_pstotal_j0660',
    'J0861': 'err_mag_pstotal_j0861'
}

# replace all with psf
for key in list(MAG_COL.keys()):
    MAG_COL[key] = MAG_COL[key].replace("pstotal", "psf")
for key in list(ERR_COL.keys()):
    ERR_COL[key] = ERR_COL[key].replace("pstotal", "psf")

# ---------- DTypes (memory-friendly) ----------
dtypes = {
    "id": "string",
    "ra": "float32",
    "dec": "float32",
    "gaia_source_id": "string",
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
# replace pstotal with psf in dtypes
for key in list(dtypes.keys()):
    if "pstotal" in key:
        dtypes[key.replace("pstotal", "psf")] = dtypes.pop(key)

# Minimized usecols: only what we need for filters + EW calc + any metadata you want to keep
USECOLS = [
    "id", "ra", "dec", "gaia_source_id",
    # mags
    MAG_COL['r'], MAG_COL['i'], MAG_COL['u'], MAG_COL['g'], MAG_COL['z'],
    MAG_COL['J0378'], MAG_COL['J0395'], MAG_COL['J0410'], MAG_COL['J0430'],
    MAG_COL['J0515'], MAG_COL['J0660'], MAG_COL['J0861'],
    # errs
    ERR_COL['r'], ERR_COL['i'], ERR_COL['u'], ERR_COL['g'], ERR_COL['z'],
    ERR_COL['J0378'], ERR_COL['J0395'], ERR_COL['J0410'], ERR_COL['J0430'],
    ERR_COL['J0515'], ERR_COL['J0660'], ERR_COL['J0861'],
]

# ==============================
# Core math helpers (unchanged)
# ==============================
def mags_to_flux_ln(m, dm):
    k = np.log(10)/2.5
    ln_f = -0.4*np.log(10)*m
    s_ln_f = k*dm
    return ln_f, s_ln_f

def weighted_poly_fit_lnflux(lam, ln_f, s_ln_f, lam0=LAM_HA, deg=2, max_iter=3, sigma_clip=3.0):
    lam = np.asarray(lam); ln_f = np.asarray(ln_f); s_ln_f = np.asarray(s_ln_f)
    t = (lam - lam0)/1000.0
    Xfull = np.vstack([np.ones_like(t), t, t**2 if deg >= 2 else np.zeros_like(t)]).T[:, :deg+1]
    mask = np.isfinite(ln_f) & np.isfinite(s_ln_f) & np.isfinite(t) & (s_ln_f > 0)
    ok = False
    beta = None; Cov = None
    for _ in range(max_iter):
        if mask.sum() < (deg+2):
            break
        X = Xfull[mask]; y = ln_f[mask]; w = 1.0 / (s_ln_f[mask]**2)
        XT_W = X.T * w
        A = XT_W @ X
        b = XT_W @ y
        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break
        yhat = X @ beta
        res = y - yhat
        dof = max(1, mask.sum() - (deg+1))
        s2 = (w * res**2).sum() / w.sum()
        s = np.sqrt(s2)
        new_mask = mask.copy()
        new_mask[np.where(mask)[0][np.abs(res) > sigma_clip * s]] = False
        if new_mask.sum() == mask.sum():
            s2_hat = (res**2).sum() / dof
            Cov = np.linalg.inv(A) * s2_hat
            ok = True
            break
        mask = new_mask
    return beta, Cov, mask, ok

def predict_lnflux_at(beta, Cov, lam_eval, lam0=LAM_HA):
    t0 = (lam_eval - lam0)/1000.0
    x0 = np.array([1.0, t0, t0**2])[:len(beta)]
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

def detect_halpha_emitters(df,
                           ew_min=10.0, snr_min=3.0,
                           deg=2, max_iter=3, sigma_clip=3.0):
    # Pre-extract arrays for speed
    lnF = {}; s_lnF = {}
    needed = CONTINUUM_BANDS + ['J0660']
    # guard: if some columns are missing in this chunk (rare), fill with NaN arrays
    for b in needed:
        m = df.get(MAG_COL[b], pd.Series(np.nan, index=df.index)).astype("float32").values
        dm = df.get(ERR_COL[b], pd.Series(np.nan, index=df.index)).astype("float32").values
        lnF[b], s_lnF[b] = mags_to_flux_ln(m, dm)

    lam_all = np.array([LAMBDA[b] for b in CONTINUUM_BANDS], dtype=float)
    lnF_all = np.vstack([lnF[b] for b in CONTINUUM_BANDS]).T
    s_lnF_all = np.vstack([s_lnF[b] for b in CONTINUUM_BANDS]).T

    lnF_nb = lnF['J0660']; s_lnF_nb = s_lnF['J0660']

    N = len(df)
    EW = np.full(N, np.nan, dtype="float64")
    dEW = np.full(N, np.nan, dtype="float64")
    SNR = np.full(N, np.nan, dtype="float64")
    method = np.empty(N, dtype=object)

    for i in range(N):
        ln_f = lnF_all[i]; s_ln_f = s_lnF_all[i]
        ok_mask = np.isfinite(ln_f) & np.isfinite(s_ln_f)
        lam_i = lam_all[ok_mask]; ln_f_i = ln_f[ok_mask]; s_ln_f_i = s_ln_f[ok_mask]

        f_nb = np.exp(lnF_nb[i])
        df_nb = f_nb * s_lnF_nb[i]

        beta, Cov, used_mask, ok = weighted_poly_fit_lnflux(
            lam_i, ln_f_i, s_ln_f_i, lam0=LAM_HA, deg=deg, max_iter=max_iter, sigma_clip=sigma_clip
        )
        if ok and np.isfinite(f_nb):
            ln_fc, var_ln_fc = predict_lnflux_at(beta, Cov, LAM_HA, lam0=LAM_HA)
            f_c = np.exp(ln_fc)
            df_c = f_c * np.sqrt(max(var_ln_fc, 0.0))
            EW[i], dEW[i] = ew_from_fluxes(f_nb, df_nb, f_c, df_c, W_nb=W_nb_J0660)
            SNR[i] = EW[i] / np.maximum(dEW[i], 1e-30)
            method[i] = 'WLS-continuum'
        else:
            ew_fb, dew_fb, snr_fb = ri_fallback(df.iloc[i])
            EW[i], dEW[i], SNR[i] = ew_fb, dew_fb, snr_fb
            method[i] = 'r-i-fallback'

    results = pd.DataFrame({
        'EW_Ha': EW.astype("float32"),
        'EW_Ha_err': dEW.astype("float32"),
        'SNR_EW': SNR.astype("float32"),
        'method': method
    }, index=df.index)

    emitters = (results['EW_Ha'] > ew_min) & (results['SNR_EW'] >= snr_min)
    return results, emitters

# ==============================
# Chunked pipeline
# ==============================
def ensure_dir_for(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

def process_chunk(chunk, ew_min=30.0, snr_min=3.0):
    # Basic quality cut
    if MAG_COL['r'] in chunk.columns:
        chunk = chunk[(chunk[MAG_COL['r']] > 12) & (chunk[MAG_COL['r']] < 22)]
    else:
        # no r-band -> nothing to do
        return None

    if chunk.empty:
        return None

    # Compute EW per chunk
    results, emitters = detect_halpha_emitters(chunk, ew_min=ew_min, snr_min=snr_min)

    # Join and apply your final cut
    out = chunk[emitters].join(results)
    if out.empty:
        return None

    # Final cut: J0660 brighter than i (i.e., mag lower)
    mask_final = out[MAG_COL['J0660']] < out[MAG_COL['i']]
    out = out[mask_final]

    return out

def main():
    global WRITE_HEADER
    ensure_dir_for(OUTPUT_CSV)
    if os.path.exists(OUTPUT_CSV):
        # resume-friendly: append without header if file exists
        WRITE_HEADER = False

    total_in, total_out = 0, 0
    print("Streaming & processing in chunks...")

    # IMPORTANT: use engine='c' for chunksize support; pyarrow engine doesn't stream
    reader = pd.read_csv(
        INPUT_CSV,
        chunksize=CHUNK_SIZE,
        dtype=dtypes,
        usecols=USECOLS,
        low_memory=False,   # better type stability while chunking
        engine="c"          # iterator requires C engine
    )

    for ichunk, chunk in enumerate(reader, start=1):
        total_in += len(chunk)
        out = process_chunk(chunk, ew_min=30.0, snr_min=3.0)
        if out is not None and len(out):
            out.to_csv(OUTPUT_CSV, mode="a", index=False, header=WRITE_HEADER)
            WRITE_HEADER = False
            total_out += len(out)
        print(f"[{ichunk}] in:{len(chunk)} kept:{0 if out is None else len(out)} totals -> in:{total_in} kept:{total_out}")

    print("Done.")
    print(f"Total rows read: {total_in}")
    print(f"Total emitters written: {total_out}")
    print(f"Output: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()