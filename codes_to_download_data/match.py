#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gaia-first S-PLUS × Gaia positional match (streaming, memory-safe)

- Opens ONE Gaia shard at a time.
- For each overlapping field:
  - Loads S-PLUS for that field on first use (kept in a small LRU cache).
  - Filters Gaia rows to the field cone.
  - Crossmatches at 1".
  - Appends matches to /outdir/matches/{field}_match.csv immediately.
- No accumulating Gaia rows per field in memory.
"""

import os
import re
import math
import argparse
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

try:
    from astropy_healpix import HEALPix
    _USE_ASTROPY_HEALPIX = True
except Exception:
    _USE_ASTROPY_HEALPIX = False
    try:
        import healpy as hp
    except Exception as e:
        raise RuntimeError("Install either astropy-healpix or healpy") from e

from tqdm import tqdm

NSIDE = 256  # HEALPix L8
ORDER = 'nested'

# ----------------------------- Helpers -----------------------------

def parse_gaia_ranges(gaia_dir: Path) -> List[Tuple[int, int, Path]]:
    pat = re.compile(r"GaiaSource_(\d+)-(\d+)\.csv\.gz$")
    out = []
    for p in sorted(gaia_dir.glob("GaiaSource_*-*.csv.gz")):
        m = pat.search(p.name)
        if m:
            start, end = map(int, m.groups())
            out.append((start, end, p))
    if not out:
        raise FileNotFoundError(f"No GaiaSource_*-*.csv.gz files found under {gaia_dir}")
    return out

def level8_pixels_in_cone(ra_deg, dec_deg, radius_deg):
    if _USE_ASTROPY_HEALPIX:
        hpix = HEALPix(nside=NSIDE, order=ORDER, frame='icrs')
        center = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame='icrs')
        pix = hpix.cone_search_skycoord(center, radius_deg*u.deg)
        return set(map(int, np.asarray(pix)))
    else:
        theta = np.deg2rad(90.0 - dec_deg)
        phi = np.deg2rad(ra_deg)
        vec = hp.ang2vec(theta, phi)
        pix = hp.query_disc(nside=NSIDE, vec=vec, radius=np.deg2rad(radius_deg), nest=True)
        return set(map(int, np.asarray(pix)))

def read_splus_catalog(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"S-PLUS catalog not found: {path}")
    try:
        tab = Table.read(path, format='ascii')
        df = tab.to_pandas()
    except Exception:
        try:
            tab = Table.read(path)
            df = tab.to_pandas()
        except Exception:
            df = pd.read_csv(path, delim_whitespace=True, comment='#', engine='python')
    df.columns = [c.strip() for c in df.columns]
    
    # select only id,ra,dec,clarr_star_r,mag_pstotal_*,err_mag_pstotal_*
    pattern = re.compile(r'^(id|ra|dec|clarr_star_r|mag_psf_.*|err_mag_psf_.*)$', re.IGNORECASE)
    selected_cols = [c for c in df.columns if pattern.match(c)]
    df = df[selected_cols]
    
    return df

def find_ra_dec_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cols = {c.lower(): c for c in df.columns}
    candidates = [
        ('ra', 'dec'), ('ra_deg', 'dec_deg'), ('ra_icrs', 'dec_icrs'),
        ('raj2000', 'dej2000'), ('alpha', 'delta'), ('ra2000', 'dec2000')
    ]
    for low_ra, low_dec in candidates:
        if low_ra in cols and low_dec in cols:
            return cols[low_ra], cols[low_dec]
    ra_col = next((cols[c] for c in cols if 'ra' in c), None)
    dec_col = next((cols[c] for c in cols if 'dec' in c or 'de' in c), None)
    if ra_col and dec_col:
        return ra_col, dec_col
    raise ValueError("Could not infer RA/Dec columns in S-PLUS catalog")

def cone_filter(df, ra_col, dec_col, center_ra, center_dec, radius_deg):
    dra = radius_deg / max(1e-6, math.cos(math.radians(abs(center_dec))))
    ra = np.mod(df[ra_col].values, 360.0)
    dec = df[dec_col].values
    ra0 = center_ra % 360.0
    in_dec = (dec >= center_dec - radius_deg) & (dec <= center_dec + radius_deg)
    lo = (ra0 - dra) % 360.0
    hi = (ra0 + dra) % 360.0
    if lo <= hi:
        in_ra = (ra >= lo) & (ra <= hi)
    else:
        in_ra = (ra >= lo) | (ra <= hi)
    pre = in_ra & in_dec
    if not np.any(pre):
        return pd.Series([False]*len(df), index=df.index)
    sc = SkyCoord(ra=ra[pre]*u.deg, dec=dec[pre]*u.deg, frame='icrs')
    cen = SkyCoord(ra=ra0*u.deg, dec=center_dec*u.deg)
    sep = sc.separation(cen).deg
    mask_precise = sep <= radius_deg
    idx = df.index[pre]
    out = pd.Series([False]*len(df), index=df.index)
    out.loc[idx] = mask_precise
    return out

def propagate_gaia_to_epoch(gdf: pd.DataFrame,
                            source_epoch: float = 2016.0,
                            target_epoch: float = 2000.0,
                            ra_col: str = "ra",
                            dec_col: str = "dec",
                            pmra_col: str = "pmra",
                            pmdec_col: str = "pmdec") -> pd.DataFrame:
    """
    Propagate Gaia positions from source_epoch (J2016) to target_epoch (J2000)
    using proper motions. pmra is μ_α* cosδ, pmdec is μ_δ (both in mas/yr).
    Output columns: ra_prop, dec_prop (degrees).
    If PM columns are missing, returns input with ra_prop/dec_prop = ra/dec.
    """
    df = gdf.copy()
    years = (target_epoch - source_epoch)

    if pmra_col in df.columns and pmdec_col in df.columns:
        ra = np.asarray(df[ra_col].values, dtype="float64")
        dec = np.asarray(df[dec_col].values, dtype="float64")
        pmra = np.asarray(df[pmra_col].fillna(0.0).values, dtype="float64")   # mas/yr (μ_α* cosδ)
        pmdec = np.asarray(df[pmdec_col].fillna(0.0).values, dtype="float64") # mas/yr

        # Convert PM to degrees over 'years'
        # Δδ = pmdec * years / 3.6e6
        ddec_deg = (pmdec * years) / 3_600_000.0

        # Δα = (pmra / cosδ) * years / 3.6e6   (careful near poles)
        cosd = np.cos(np.deg2rad(np.clip(dec, -89.999999, 89.999999)))
        # avoid division by 0
        cosd[cosd == 0] = 1e-12
        dra_deg = (pmra * years) / (3_600_000.0 * cosd)

        ra_new = (ra + dra_deg) % 360.0
        dec_new = dec + ddec_deg
        # clip Dec to valid range (very high PM objects)
        dec_new = np.clip(dec_new, -90.0, 90.0)

        df["ra_prop"] = ra_new
        df["dec_prop"] = dec_new
    else:
        # No PMs → use original positions
        df["ra_prop"] = df[ra_col].values
        df["dec_prop"] = df[dec_col].values

    return df

def crossmatch_1arcsec(splus, gaia, splus_ra, splus_dec,
                       gaia_ra='ra', gaia_dec='dec', max_sep_arcsec=1.0):
    s_coord = SkyCoord(ra=splus[splus_ra].values*u.deg,
                       dec=splus[splus_dec].values*u.deg, frame='icrs')
    g_coord = SkyCoord(ra=gaia[gaia_ra].values*u.deg,
                       dec=gaia[gaia_dec].values*u.deg, frame='icrs')
    idx, sep2d, _ = s_coord.match_to_catalog_sky(g_coord)
    ok = sep2d.arcsec <= max_sep_arcsec
    s_sel = splus.loc[ok].reset_index(drop=True)
    g_sel = gaia.iloc[idx[ok]].reset_index(drop=True)
    g_sel = g_sel.rename(columns={c: f"gaia_{c}" for c in g_sel.columns})
    
    # add column of separation in arcsec
    s_sel = s_sel.copy()
    s_sel['separation_arcsec'] = sep2d.arcsec[ok]
    
    return pd.concat([s_sel, g_sel], axis=1)

# ------------------------- S-PLUS LRU cache -------------------------

class SplusCache:
    def __init__(self, capacity: int):
        self.capacity = max(1, capacity)
        self.cache: "OrderedDict[str, Tuple[pd.DataFrame,str,str,int]]" = OrderedDict()
        # value: (splus_df, ra_col, dec_col, n_rows)

    def get(self, field_name: str):
        if field_name in self.cache:
            self.cache.move_to_end(field_name)
            return self.cache[field_name]
        return None

    def put(self, field_name: str, df: pd.DataFrame, ra_col: str, dec_col: str):
        n_rows = len(df)
        self.cache[field_name] = (df, ra_col, dec_col, n_rows)
        self.cache.move_to_end(field_name)
        if len(self.cache) > self.capacity:
            evicted, _ = self.cache.popitem(last=False)
            print(f"[DEBUG] Evicted S-PLUS cache entry: {evicted}")

# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True)
    ap.add_argument("--gaia-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--radius", type=float, default=1.2)
    ap.add_argument("--gaia-cols", default="ra,dec,source_id")
    ap.add_argument("--splus-templates", default="/storage/splus/{field}.cat")
    ap.add_argument("--splus-template", default=None, help="Alias for single template; if set, overrides --splus-templates")
    ap.add_argument("--splus-cache", type=int, default=16, help="Max number of S-PLUS fields cached in memory")
    ap.add_argument("--max-fields", type=int, default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / "matches").mkdir(parents=True, exist_ok=True)
    (outdir / "logs").mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading field list…")
    fields_df = pd.read_csv(args.fields)
    for col in ("field", "ra", "dec"):
        if col not in fields_df.columns:
            raise ValueError(f"Missing column '{col}' in fields CSV")
    if args.max_fields is not None:
        fields_df = fields_df.head(args.max_fields).copy()

    # Templates (support both --splus-template and --splus-templates)
    if args.splus_template:
        templates = [args.splus_template]
    else:
        templates = [t.strip() for t in args.splus_templates.split(";") if t.strip()]

    def has_splus(field_name):
        for tmpl in templates:
            if Path(tmpl.format(field=field_name)).exists():
                return True
        return False

    fields_df = fields_df[fields_df["field"].apply(has_splus)].reset_index(drop=True)
    print(f"[INFO] {len(fields_df)} fields with S-PLUS catalogs found.")

    if len(fields_df) == 0:
        print("[WARN] No fields with S-PLUS catalogs. Exiting.")
        return

    # Build pixel map
    print("[INFO] Building HEALPix pixel map…")
    pixel2fields: Dict[int, List[int]] = {}
    for i, row in fields_df.iterrows():
        pix = level8_pixels_in_cone(row["ra"], row["dec"], args.radius)
        for p in pix:
            pixel2fields.setdefault(p, []).append(i)
    if not pixel2fields:
        print("[WARN] No pixels produced. Check RA/Dec and radius.")
        return

    pixels_sorted = np.array(sorted(pixel2fields.keys()), dtype=int)

    def candidate_fields_for_range(start, end):
        i = pixels_sorted.searchsorted(start, side='left')
        j = pixels_sorted.searchsorted(end,   side='right')
        if i >= j:
            return []
        cand = set()
        for p in pixels_sorted[i:j]:
            cand.update(pixel2fields[p])
        return sorted(cand)

    print("[INFO] Parsing Gaia shards…")
    ranges = parse_gaia_ranges(Path(args.gaia_dir))
    shard2fields: Dict[Path, List[int]] = {}
    for (start, end, path) in ranges:
        cand = candidate_fields_for_range(start, end)
        if cand:
            shard2fields[path] = cand
    print(f"[INFO] {len(shard2fields)} Gaia shards overlap at least one field "
          f"(from {len(ranges)} total).")

    # GAIA columns
    gaia_cols = [c.strip() for c in args.gaia_cols.split(",") if c.strip()]
    if "ra" not in gaia_cols or "dec" not in gaia_cols:
        raise ValueError("--gaia-cols must include 'ra' and 'dec'")

    # S-PLUS cache + running counters per field for summary
    splus_cache = SplusCache(capacity=args.splus_cache)
    summary = {}  # field_name -> dict

    # For appending CSVs with/without header
    created_output: Dict[str, bool] = {}

    # Walk shards; open only candidates; process field-by-field; append results immediately
    opened_shards = 0
    for (start, end, shard_path) in tqdm(ranges, desc="Gaia shards"):
        candidate_field_idxs = shard2fields.get(shard_path, [])
        if not candidate_field_idxs:
            continue

        opened_shards += 1
        print(f"[INFO] Opening Gaia shard {shard_path.name} for {len(candidate_field_idxs)} fields")

        # Prefer CSV (most Gaia shards), but allow ecsv fallback if needed
        gdf = None
        try:
            gdf = pd.read_csv(shard_path, usecols=gaia_cols, low_memory=False)
        except Exception as e_csv:
            try:
                gdf = Table.read(
                    shard_path,
                    format="ascii.ecsv",
                    guess=False,
                    fill_values=[('null', 99), ('nan', 99)],
                ).to_pandas()
            except Exception as e_ecsv:
                print(f"[WARN] Failed to read {shard_path.name} as CSV ({e_csv}) and ECSV ({e_ecsv}). Skipping.")
                continue

        if gdf is None or gdf.empty:
            print(f"[INFO] {shard_path.name}: 0 rows read, skipping.")
            continue
        
        before_cols = set(gdf.columns)
        gdf = propagate_gaia_to_epoch(
            gdf,
            source_epoch=2016.0,
            target_epoch=2000.0,
            ra_col="ra", dec_col="dec",
            pmra_col="pmra", pmdec_col="pmdec"
        )
        if not {"ra_prop","dec_prop"}.issubset(gdf.columns):
            raise RuntimeError("Failed to create ra_prop/dec_prop; check Gaia columns.")

        added = set(gdf.columns) - before_cols
        print(f"[DEBUG] {shard_path.name}: propagated to J2000; added columns {sorted(list(added))}")

        kept_total = 0

        for fid in candidate_field_idxs:
            field_name = str(fields_df.loc[fid, "field"])
            ra0 = float(fields_df.loc[fid, "ra"])
            dec0 = float(fields_df.loc[fid, "dec"])

            # Load S-PLUS for this field from cache or disk
            entry = splus_cache.get(field_name)
            if entry is None:
                # concatenate all existing templates
                splus_dfs = []
                for tmpl in templates:
                    p = Path(tmpl.format(field=field_name))
                    if p.exists():
                        try:
                            splus_dfs.append(read_splus_catalog(p))
                        except Exception as e:
                            print(f"[WARN] Failed to read S-PLUS file {p}: {e}")
                if not splus_dfs:
                    # shouldn't happen because we filtered up front, but be safe
                    print(f"[WARN] No S-PLUS catalogs present for {field_name} at processing time.")
                    continue
                splus_df = pd.concat(splus_dfs, ignore_index=True).drop_duplicates()
                try:
                    s_ra, s_dec = find_ra_dec_columns(splus_df)
                except Exception as e:
                    print(f"[WARN] Could not infer RA/Dec columns for {field_name}: {e}")
                    continue
                splus_cache.put(field_name, splus_df, s_ra, s_dec)
                entry = splus_cache.get(field_name)

            splus_df, s_ra, s_dec, s_rows = entry

            # Crossmatch immediately
            gaia_ra_col = "ra_prop"
            gaia_dec_col = "dec_prop"
            # Filter Gaia rows to the cone of this field
            mask = cone_filter(gdf, gaia_ra_col, gaia_dec_col, ra0, dec0, args.radius)
            if not mask.any():
                continue
            sel = gdf.loc[mask]
            
            if not mask.any():
                continue
            sel = gdf.loc[mask]

            matched_df = crossmatch_1arcsec(
                splus_df, sel,
                s_ra, s_dec,
                gaia_ra=gaia_ra_col, gaia_dec=gaia_dec_col,
                max_sep_arcsec=1.0
            )

            # Append to per-field CSV right away
            out_csv = outdir / "matches" / f"{field_name}_match.csv"
            write_header = not out_csv.exists() and not created_output.get(field_name, False)
            matched_df.to_csv(out_csv, mode='a', header=write_header, index=False)
            created_output[field_name] = True

            # Update summary counters
            s = summary.setdefault(field_name, {"gaia_rows": 0, "splus_rows": s_rows, "matched_rows": 0})
            s["gaia_rows"] += int(len(sel))
            s["matched_rows"] += int(len(matched_df))

            kept_total += int(len(sel))

        print(f"[INFO] {shard_path.name}: kept {kept_total} Gaia rows across {len(candidate_field_idxs)} fields")

    print(f"[INFO] Finished Gaia pass. Opened {opened_shards} shards.")
    if not summary:
        print("[WARN] No matches were produced.")
        # Still write an empty summary for bookkeeping
        pd.DataFrame([]).to_csv(outdir / "summary.csv", index=False)
        print(f"[INFO] Empty summary saved at {outdir/'summary.csv'}")
        return

    # Write summary
    rows = []
    for field_name, stats in summary.items():
        rows.append({
            "field": field_name,
            "gaia_rows_processed": stats["gaia_rows"],
            "splus_rows": stats["splus_rows"],
            "matched_rows": stats["matched_rows"],
            "status": "OK" if stats["matched_rows"] > 0 else "NO_MATCHES"
        })
    pd.DataFrame(rows).to_csv(outdir / "summary.csv", index=False)
    print(f"[INFO] Done. Summary saved at {outdir/'summary.csv'}")
    print(f"[INFO] Per-field CSVs saved under {outdir/'matches'}")

if __name__ == "__main__":
    main()