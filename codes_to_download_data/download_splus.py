import os
import adss
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from astropy.table import Table

# ---------------- Pretty prints ----------------
def print_log(message, level="INFO"):
    if level == "ERROR":
        color = "\033[91m"
    elif level == "WARNING":
        color = "\033[93m"
    else:
        color = "\033[92m"
    white_color = "\033[97m"
    print(f"{color}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {level}] {white_color}{message}")

# ---------------- FITS helper ----------------
def save_df_to_fits(df: pd.DataFrame, path: str):
    """
    Salva um DataFrame em FITS usando astropy.
    - Converte para astropy.Table para lidar com metadados e dtypes.
    - overwrite=True para substituir se existir.
    """
    tbl = Table.from_pandas(df)  # lida com strings e tipos automaticamente
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tbl.write(path, overwrite=True)

# ---------------- Main ----------------
def main(outfolder='data'):
    # Se quiser parametrizar via argparse, descomente e remova a classe Args.
    # import argparse
    # parser = argparse.ArgumentParser(description='Extrai dados S-PLUS campo a campo e salva em FITS')
    # parser.add_argument('--outfolder', '-o', type=str, default='data', help='Pasta de saída')
    # parser.add_argument('--username', '-u', type=str, default='', help='ADSS username')
    # parser.add_argument('--password', '-p', type=str, default='', help='ADSS password')
    # args = parser.parse_args()

    class Args:
        def __init__(self):
            self.username = ''  # preencha se necessário ou use tokens já válidos no adss
            self.password = ''
            self.outfolder = outfolder

    args = Args()

    print_log("Inicializando cliente ADSS...")
    client = adss.ADSSClient(
        base_url="https://splus.cloud/",
        username=args.username,
        password=args.password,
        verify_ssl=False
    )

    print_log("Carregando metadados do banco...")
    metadata = client.get_database_metadata()
    schema = metadata.get_schema("splus")
    table = schema.get_table("splus_idr6")
    print_log(f"  Esquema: splus | Tabela: splus_idr6")
    print_log(f"  Nº de colunas: {len(table.column_names())}")

    # --- Carregar ou consultar a lista de fields ---
    print_log("Carregando/consultando fields...")
    os.makedirs(args.outfolder, exist_ok=True)
    fields_cache = os.path.join(args.outfolder, "fields.csv")

    if not os.path.exists(fields_cache):
        print_log("  Consultando fields distintos em splus.splus_idr6 ...")
        result = client.query_and_wait("SELECT DISTINCT field FROM idr6.idr6")
        result.to_csv(fields_cache, index=False)
        fields = result.data["field"].tolist()
        print_log(f"  Fields consultados: {len(fields)}")
    else:
        fields = pd.read_csv(fields_cache)["field"].tolist()
        print_log(f"  Fields carregados do cache: {len(fields)}")

    # --- Processar cada field e salvar em FITS ---
    print_log(f"Salvando arquivos FITS em: {args.outfolder}")
    pbar = tqdm(enumerate(fields, 1), total=len(fields), desc="Processando fields", unit="field")

    for i, field in pbar:
        fits_path = os.path.join(args.outfolder, f"{field}.fits")
        if os.path.exists(fits_path):
            pbar.set_postfix_str("skip (já existe)")
            continue

        pbar.set_postfix_str(f"field={field}")
        try:
            # Se quiser limitar colunas, troque o SELECT * por colunas específicas
            query = f"""
                SELECT *
                FROM idr6.idr6
                WHERE field = '{field}'
            """
            result = client.query_and_wait(query)

            if result.data.empty:
                print_log(f"  Sem dados para field: {field}", level="WARNING")
                continue

            # Salvar em FITS
            save_df_to_fits(result.data, fits_path)

        except Exception as e:
            print_log(f"  Erro ao processar field {field}: {e}", level="ERROR")

    print_log("Concluído.")

# ---------------- Run ----------------
if __name__ == "__main__":
    # Ajuste a pasta de saída aqui:
    main(outfolder='../../../Mounts/Work1/Catalogues/SPLUS/iDR6/VAC_Catalogues/')