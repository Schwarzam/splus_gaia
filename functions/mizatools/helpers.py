import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, search_around_sky, Angle
import astropy.units as u


def unique_by_sep(ra_deg, dec_deg, min_sep_arcsec=2.0):
    """
    Remove objetos que estão a menos de min_sep_arcsec de algum objeto já mantido.
    Mantém o primeiro que aparece em cada vizinhança.

    Parameters
    ----------
    ra_deg, dec_deg : array-like (graus)
    min_sep_arcsec : float

    Returns
    -------
    keep_mask : np.ndarray (bool)
        True para objetos mantidos.
    """
    ra_deg = np.asarray(ra_deg, dtype=float)
    dec_deg = np.asarray(dec_deg, dtype=float)

    c = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame="icrs")
    seplim = (min_sep_arcsec * u.arcsec)

    # Lista de pares (i,j) com separação < seplim (exclui i=j)
    pairs = c.search_around_sky(c, seplim)
    idx1, idx2 = pairs[0], pairs[1]

    keep = np.ones(len(c), dtype=bool)

    # Se i<j e são próximos, drop o j (mantém o primeiro por ordem)
    for i, j in zip(idx1, idx2):
        if i == j:
            continue
        if i < j and keep[j]:
            # se i já foi removido, não faz sentido remover j por causa dele
            if keep[i]:
                keep[j] = False

    return keep
def match_emitters_to_clusters(
    clusters: pd.DataFrame,
    stars: pd.DataFrame,
    *,
    cluster_ra_col: str = "ra",
    cluster_dec_col: str = "dec",
    cluster_amaj_col: str = "amaj",
    stars_ra_col: str = "ra",
    stars_dec_col: str = "dec",
    amaj_unit: str = "arcmin",
    keep_cols_clusters: list | None = None,
    keep_cols_stars: list | None = None,
    stars_chunk_size: int = 250_000,
    return_columns: str = "full",  # "full" | "pairs" (só indices + separação)
    sort_by_sep: bool = True,
) -> pd.DataFrame:
    """
    Versão otimizada em memória:
    - clusters ficam fixos em memória
    - estrelas são processadas em chunks (evita SkyCoord gigante)
    - acumula resultados por chunk e concatena no final

    return_columns:
      - "pairs": retorna apenas cluster_index, star_index, sep_arcsec, sep_arcmin
      - "full" : adiciona colunas prefixadas cluster_/star_ como na sua função
    """
    if keep_cols_clusters is None:
        keep_cols_clusters = []
    if keep_cols_stars is None:
        keep_cols_stars = []

    # --------
    # 1) Pré-filtra clusters (sem cópias pesadas)
    # --------
    cl_req = [cluster_ra_col, cluster_dec_col, cluster_amaj_col] + keep_cols_clusters
    cl_df = clusters.loc[:, cl_req]

    # máscara booleana (bem mais leve que replace/dropna em grandes tabelas)
    cl_mask = (
        np.isfinite(cl_df[cluster_ra_col].to_numpy(dtype=float, copy=False)) &
        np.isfinite(cl_df[cluster_dec_col].to_numpy(dtype=float, copy=False)) &
        np.isfinite(cl_df[cluster_amaj_col].to_numpy(dtype=float, copy=False)) &
        (cl_df[cluster_amaj_col].to_numpy(dtype=float, copy=False) > 0)
    )
    cl_df = cl_df.loc[cl_mask]

    if cl_df.empty:
        return pd.DataFrame(columns=["cluster_index","star_index","sep_arcsec","sep_arcmin"])

    # SkyCoord dos clusters (normalmente pequeno)
    cl_ra = cl_df[cluster_ra_col].to_numpy(dtype=np.float64, copy=False)
    cl_dec = cl_df[cluster_dec_col].to_numpy(dtype=np.float64, copy=False)
    c_clusters = SkyCoord(ra=cl_ra * u.deg, dec=cl_dec * u.deg)

    unit = u.Unit(amaj_unit)
    cl_amaj = cl_df[cluster_amaj_col].to_numpy(dtype=np.float64, copy=False)
    max_amaj = float(np.nanmax(cl_amaj))
    seplimit = Angle(max_amaj, unit)

    # Para mapear índices internos do cl_df (0..n-1) para índices originais do clusters
    cl_index_orig = cl_df.index.to_numpy()

    # --------
    # 2) Itera estrelas em chunks
    # --------
    st_req = [stars_ra_col, stars_dec_col] + keep_cols_stars
    st_df_all = stars.loc[:, st_req]

    out_chunks = []
    n = len(st_df_all)

    for start in range(0, n, stars_chunk_size):
        end = min(start + stars_chunk_size, n)
        st_chunk = st_df_all.iloc[start:end]

        # filtra chunk (sem replace/dropna)
        st_ra = st_chunk[stars_ra_col].to_numpy(dtype=np.float64, copy=False)
        st_dec = st_chunk[stars_dec_col].to_numpy(dtype=np.float64, copy=False)
        st_mask = np.isfinite(st_ra) & np.isfinite(st_dec)

        if not np.any(st_mask):
            continue

        st_chunk_valid = st_chunk.loc[st_mask]
        st_ra_v = st_chunk_valid[stars_ra_col].to_numpy(dtype=np.float64, copy=False)
        st_dec_v = st_chunk_valid[stars_dec_col].to_numpy(dtype=np.float64, copy=False)

        c_stars = SkyCoord(ra=st_ra_v * u.deg, dec=st_dec_v * u.deg)

        # busca por chunk
        idx_cl, idx_st, sep2d, _ = search_around_sky(c_clusters, c_stars, seplimit=seplimit)
        if len(idx_cl) == 0:
            continue

        # corte par-a-par com amaj do cluster
        amaj_per_pair = Angle(cl_amaj[idx_cl], unit)
        keep = sep2d <= amaj_per_pair
        if not np.any(keep):
            continue

        idx_cl_k = idx_cl[keep]
        idx_st_k = idx_st[keep]
        sep_k = sep2d[keep]

        # índices originais (clusters e stars)
        pairs = pd.DataFrame({
            "cluster_index": cl_index_orig[idx_cl_k],
            "star_index": st_chunk_valid.index.to_numpy()[idx_st_k],
            "sep_arcsec": sep_k.arcsec,
            "sep_arcmin": sep_k.to(u.arcmin).value,
        })

        if return_columns == "pairs":
            out_chunks.append(pairs)
            continue

        # "full": anexa metadados (cuidado: isso pode ficar grande se houver muitos pares)
        cl_out = clusters.loc[pairs["cluster_index"], [cluster_ra_col, cluster_dec_col, cluster_amaj_col] + keep_cols_clusters]
        cl_out = cl_out.add_prefix("cluster_").reset_index(drop=True)

        st_out = stars.loc[pairs["star_index"], [stars_ra_col, stars_dec_col] + keep_cols_stars]
        st_out = st_out.add_prefix("star_").reset_index(drop=True)

        out_chunks.append(pd.concat([pairs.reset_index(drop=True), cl_out, st_out], axis=1))

    if not out_chunks:
        return pd.DataFrame(columns=["cluster_index","star_index","sep_arcsec","sep_arcmin"])

    result = pd.concat(out_chunks, ignore_index=True)

    if sort_by_sep and "sep_arcmin" in result.columns and len(result) > 1:
        result = result.sort_values("sep_arcmin", ascending=True, kind="mergesort").reset_index(drop=True)

    return result
def nome_antes_virgula_sem_espacos(nome):
    """Versão direta: retorna o que está antes da vírgula sem espaços"""
    if pd.isna(nome):
        return ""
    
    # Pegar tudo antes da primeira vírgula
    antes = str(nome).strip().split(',')[0].replace("-", "").replace(" ", "")
    
    # Remover todos os espaços
    return antes

