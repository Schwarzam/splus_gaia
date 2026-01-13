import splusdata

outdir = '/mnt/hdcasa/splus_gaia/mc_catalogs/'

conn = splusdata.Core()


for i in range(1, 150):
    print(f'Downloading MC{i:04d}')
    try:
        query = f"""
                select id,ra,dec,mag_psf*,err_mag_psf* from idr6.idr6 where field = 'MC{i:04d}'
            """
        mc = conn.query(
            query
        )
        mc.to_csv(f'{outdir}/MC{i:04d}.csv', index=False)
    except Exception as e:
        print(f'Error downloading MC{i:04d}: {e}')