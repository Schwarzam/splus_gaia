import numpy as np
import pandas as pd
from typing import Tuple, Dict, Sequence, Optional

# Optional tqdm import (doesn't crash if unavailable)
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# ---------- Configuration ----------
# Effective widths in Å (add others here if you plan to use different narrow bands)
BAND_WIDTH = {
    'J0660': 147.0,   # S-PLUS Hα narrow band
    # 'J0378': ???,   # add if you plan to detect in other narrow bands
}

# Central wavelengths (Å)
LAMBDA = {
    'u': 3536.0, 'J0378': 3770.0, 'J0395': 3940.0, 'J0410': 4094.0, 'J0430': 4292.0,
    'g': 4751.0, 'J0515': 5133.0, 'r': 6258.0, 'J0660': 6614.0, 'i': 7690.0,
    'J0861': 8611.0, 'z': 8831.0
}

# Broad bands for continuum bracketing in the generic fallback
BROAD_BANDS = ['u', 'g', 'r', 'i', 'z']

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
def mags_to_flux_ln(m: np.ndarray, dm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert AB magnitudes to natural-log flux (arbitrary units) and its uncertainty.

    ln f = -0.4 * ln(10) * m + C
    sigma[ln f] = (ln(10)/2.5) * dm

    The additive constant C cancels in ratios, so we ignore it.
    """
    k = np.log(10)/2.5
    ln_f = -0.4*np.log(10)*m
    s_ln_f = k*dm
    return ln_f, s_ln_f


def weighted_poly_fit_lnflux(lam: np.ndarray,
                             ln_f: np.ndarray,
                             s_ln_f: np.ndarray,
                             lam0: float,
                             deg: int = 2,
                             max_iter: int = 3,
                             sigma_clip: float = 3.0):
    """
    Fit ln(f) as a polynomial in t = (lam - lam0)/1000 with weights 1/sigma^2.

    Returns
    -------
    beta : np.ndarray or None
        Polynomial coefficients (length = deg+1) if successful.
    Cov : np.ndarray or None
        Covariance matrix of beta if successful.
    mask_used : np.ndarray (bool)
        Mask of points used in the final fit (relative to input arrays).
    ok : bool
        True if a stable solution was obtained.
    """
    lam = np.asarray(lam); ln_f = np.asarray(ln_f); s_ln_f = np.asarray(s_ln_f)
    t = (lam - lam0)/1000.0
    Xfull = np.vstack([np.ones_like(t), t, t**2 if deg >= 2 else np.zeros_like(t)]).T[:, :deg+1]

    mask = np.isfinite(ln_f) & np.isfinite(s_ln_f) & np.isfinite(t) & (s_ln_f > 0)
    ok = False
    beta = None; Cov = None

    for _ in range(max_iter):
        if mask.sum() < (deg+2):
            break
        X = Xfull[mask]
        y = ln_f[mask]
        w = 1.0 / (s_ln_f[mask]**2)

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


def predict_lnflux_at(beta: np.ndarray, Cov: np.ndarray, lam_eval: float, lam0: float) -> Tuple[float, float]:
    """Predict ln(f) and its variance at lam_eval from polynomial coefficients."""
    t0 = (lam_eval - lam0)/1000.0
    x0 = np.array([1.0, t0, t0**2])[:len(beta)]
    ln_f_hat = float(x0 @ beta)
    var_ln_f_hat = float(x0 @ Cov @ x0)
    return ln_f_hat, var_ln_f_hat


def ew_from_fluxes(f_nb: float, df_nb: float, f_c: float, df_c: float, W_nb: float) -> Tuple[float, float]:
    """Equivalent width and uncertainty from narrow-band and continuum fluxes."""
    R = f_nb / f_c
    dR = R * np.sqrt((df_nb/np.maximum(f_nb, 1e-300))**2 + (df_c/np.maximum(f_c, 1e-300))**2)
    EW = W_nb * (R - 1.0)
    dEW = W_nb * dR
    return EW, dEW


def _mag_to_flux(m: float, dm: float) -> Tuple[float, float]:
    """Single-value mag→flux with uncertainty."""
    f = 10**(-0.4*m)
    df = (np.log(10)/2.5) * f * dm
    return f, df


def generic_linear_fallback(df_row: pd.Series,
                            line_lambda: float) -> Tuple[float, float, float]:
    """
    Generic linear interpolation of the *continuum* in flux using the two nearest
    broad bands that bracket `line_lambda`. Uses the measured narrow-band for the line.

    Works anywhere in the optical as long as two broad bands bracket the line.
    """
    # Find the two bracketing broad bands
    broad_lams = np.array([LAMBDA[b] for b in BROAD_BANDS])
    order = np.argsort(broad_lams)
    bb = np.array(BROAD_BANDS)[order]
    bl = broad_lams[order]

    if not (line_lambda >= bl.min() and line_lambda <= bl.max()):
        # If outside the convex hull, fall back to nearest two by distance
        idx = np.argsort(np.abs(bl - line_lambda))[:2]
        b0, b1 = bb[idx[0]], bb[idx[1]]
        lam0, lam1 = bl[idx[0]], bl[idx[1]]
    else:
        # Proper bracket
        hi = np.searchsorted(bl, line_lambda)
        lo = hi - 1
        b0, b1 = bb[lo], bb[hi]
        lam0, lam1 = bl[lo], bl[hi]

    # Fluxes for the two broad bands
    f0, df0 = _mag_to_flux(df_row[MAG_COL[b0]], df_row[ERR_COL[b0]])
    f1, df1 = _mag_to_flux(df_row[MAG_COL[b1]], df_row[ERR_COL[b1]])

    # Interp weights at line_lambda
    w1 = (line_lambda - lam0) / (lam1 - lam0)
    w0 = 1.0 - w1

    fc = w0 * f0 + w1 * f1
    dfc = np.sqrt((w0*df0)**2 + (w1*df1)**2)

    return fc, dfc, (b0, b1)


def detect_narrowband_emitters(df: pd.DataFrame,
                               emit_band: str = 'J0660',
                               line_lambda: float = 6563.0,
                               band_width: float = 147.0,
                               ew_min: float = 10.0,
                               snr_min: float = 3.0,
                               deg: int = 2,
                               max_iter: int = 3,
                               sigma_clip: float = 3.0,
                               show_progress: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Compute EW for a line measured in a chosen narrow band using a continuum fit in ln-flux.
    Excludes `emit_band` from the continuum fit. Falls back to a generic linear interpolation
    between two *broad* bands bracketing `line_lambda` if the WLS fit is not possible.

    Parameters
    ----------
    df : DataFrame
        Input photometric table (magnitudes + errors).
    emit_band : str, default 'J0660'
        Band to be treated as the narrow line band.
    line_lambda : float, default 6563.0
        Line rest wavelength (Å) for continuum evaluation.
    band_width : float, default 147.0
        Effective width (Å) of the narrow band.
    ew_min : float, default 10.0
        Minimum EW threshold (Å) for the emitter selection.
    snr_min : float, default 3.0
        Minimum EW SNR threshold for the emitter selection.
    deg, max_iter, sigma_clip : int, int, float
        WLS continuum fit settings.
    show_progress : bool, default True
        Show tqdm progress bar.

    Returns
    -------
    results : DataFrame
        Columns: ['EW','EW_err','SNR_EW','method','fallback_bands'].
    mask_emitters : Series (bool)
        True where EW > ew_min and SNR_EW >= snr_min.
    """
    # Build continuum band list dynamically (exclude the emission band)
    continuum_bands = [b for b in BANDS if b != emit_band]

    # Pre-extract ln-flux for all bands used
    lnF: Dict[str, np.ndarray] = {}
    s_lnF: Dict[str, np.ndarray] = {}
    for b in continuum_bands + [emit_band]:
        m = df[MAG_COL[b]].values.astype(float)
        dm = df[ERR_COL[b]].values.astype(float)
        lnF[b], s_lnF[b] = mags_to_flux_ln(m, dm)

    lam_all = np.array([LAMBDA[b] for b in continuum_bands], dtype=float)
    lnF_all = np.vstack([lnF[b] for b in continuum_bands]).T
    s_lnF_all = np.vstack([s_lnF[b] for b in continuum_bands]).T

    lnF_nb = lnF[emit_band]
    s_lnF_nb = s_lnF[emit_band]

    N = len(df)
    EW = np.full(N, np.nan)
    dEW = np.full(N, np.nan)
    SNR = np.full(N, np.nan)
    method = np.empty(N, dtype=object)
    fallback_bands = np.empty(N, dtype=object)

    iterator = tqdm(range(N), total=N, desc="Fitting continuum", unit="obj") if show_progress else range(N)

    for i in iterator:
        lam = lam_all
        ln_f = lnF_all[i]
        s_ln_f = s_lnF_all[i]

        # Drop NaNs per-row
        ok_mask = np.isfinite(ln_f) & np.isfinite(s_ln_f)
        lam_i = lam[ok_mask]; ln_f_i = ln_f[ok_mask]; s_ln_f_i = s_ln_f[ok_mask]

        beta, Cov, used_mask, ok = weighted_poly_fit_lnflux(
            lam_i, ln_f_i, s_ln_f_i, lam0=line_lambda, deg=deg, max_iter=max_iter, sigma_clip=sigma_clip
        )

        # Narrow-band flux in linear space
        f_nb = np.exp(lnF_nb[i])
        df_nb = f_nb * s_lnF_nb[i]

        if ok and beta is not None and Cov is not None:
            ln_fc, var_ln_fc = predict_lnflux_at(beta, Cov, line_lambda, lam0=line_lambda)
            f_c = np.exp(ln_fc)
            df_c = f_c * np.sqrt(max(var_ln_fc, 0.0))
            EW[i], dEW[i] = ew_from_fluxes(f_nb, df_nb, f_c, df_c, W_nb=band_width)
            SNR[i] = EW[i] / np.maximum(dEW[i], 1e-30)
            method[i] = 'WLS-continuum'
            fallback_bands[i] = None
        else:
            # Generic broad-band interpolation fallback
            try:
                fc, dfc, (b0, b1) = generic_linear_fallback(df.iloc[i], line_lambda=line_lambda)
                EW[i], dEW[i] = ew_from_fluxes(f_nb, df_nb, fc, dfc, W_nb=band_width)
                SNR[i] = EW[i] / np.maximum(dEW[i], 1e-30)
                method[i] = 'broadband-linear'
                fallback_bands[i] = f"{b0}-{b1}"
            except Exception:
                method[i] = 'failed'
                fallback_bands[i] = None

    results = pd.DataFrame({
        'EW': EW,
        'EW_err': dEW,
        'SNR_EW': SNR,
        'method': method,
        'fallback_bands': fallback_bands
    }, index=df.index)

    emitters = (results['EW'] > ew_min) & (results['SNR_EW'] >= snr_min)
    return results, emitters


# ---------- CLI / main ----------
def build_default_dtypes() -> Dict[str, str]:
    """Default dtypes matching your previous script."""
    return {
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


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect emission-line candidates from S-PLUS photometry using a continuum fit."
    )
    parser.add_argument("--input", default="../data/all_0.7frac.csv", help="Input CSV path")
    parser.add_argument("--output", default="../data/splus_emitters.csv", help="Output CSV path")
    parser.add_argument("--emit-band", default="J0660", help="Band containing the emission line (default: J0660)")
    parser.add_argument("--line-lambda", type=float, default=6563.0, help="Line wavelength in Å (default: 6563)")
    parser.add_argument("--band-width", type=float, default=None,
                        help="Effective width (Å) of the narrow band. "
                             "If not provided, tries BAND_WIDTH[emit_band], else falls back to 147.")
    parser.add_argument("--ew-min", type=float, default=30.0, help="Minimum EW (Å) to select emitters (default: 30)")
    parser.add_argument("--snr-min", type=float, default=3.0, help="Minimum EW SNR (default: 3)")
    parser.add_argument("--deg", type=int, default=2, help="Polynomial degree for ln-flux continuum (default: 2)")
    parser.add_argument("--max-iter", type=int, default=3, help="Max iterations for sigma-clipped WLS (default: 3)")
    parser.add_argument("--sigma-clip", type=float, default=3.0, help="Sigma clip threshold (default: 3.0)")
    parser.add_argument("--r-mag-min", type=float, default=12.0, help="Lower r-mag cut (default: 12)")
    parser.add_argument("--r-mag-max", type=float, default=22.0, help="Upper r-mag cut (default: 22)")
    parser.add_argument("--extra-cut-jlessi", action="store_true",
                        help="Apply extra cut: mag(emit_band) < mag(i)")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")

    args = parser.parse_args()

    emit_band = args.emit_band
    if emit_band not in MAG_COL:
        raise ValueError(f"emit_band='{emit_band}' not found in MAG_COL/ERR_COL mappings.")

    band_width = args.band_width
    if band_width is None:
        band_width = BAND_WIDTH.get(emit_band, 147.0)

    # Build dtypes and ensure we load the needed columns
    dtypes = build_default_dtypes()
    needed_cols = set(dtypes.keys())
    # Ensure we have the chosen emit band columns
    needed_cols.add(MAG_COL[emit_band])
    needed_cols.add(ERR_COL[emit_band])

    print("Loading data...")
    df = pd.read_csv(
        args.input,
        engine="pyarrow",
        dtype=dtypes,
        usecols=list(needed_cols & set(dtypes.keys()))
    )

    print("Filtering data...")
    if ('mag_pstotal_r' in df.columns):
        df = df[(df['mag_pstotal_r'] > args.r_mag_min) & (df['mag_pstotal_r'] < args.r_mag_max)]

    print(f"Selecting emitters in {emit_band} around λ={args.line_lambda:.1f} Å ...")
    results, emitters = detect_narrowband_emitters(
        df,
        emit_band=emit_band,
        line_lambda=args.line_lambda,
        band_width=band_width,
        ew_min=args.ew_min,
        snr_min=args.snr_min,
        deg=args.deg,
        max_iter=args.max_iter,
        sigma_clip=args.sigma_clip,
        show_progress=not args.no_progress
    )
    out = df.join(results)

    if args.extra_cut_jlessi and 'mag_pstotal_i' in df.columns and MAG_COL.get(emit_band) in df.columns:
        print("Applying extra cut: mag(emit_band) < mag(i)")
        out = out[out[MAG_COL[emit_band]] < out['mag_pstotal_i']]

    print("Saving results...")
    out.to_csv(args.output, index=False)
    print(f"Done. Wrote: {args.output}")


if __name__ == "__main__":
    main()