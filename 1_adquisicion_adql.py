import os
import time
import pandas as pd
from astroquery.gaia import Gaia

def main():
    os.makedirs('data', exist_ok=True)
    csv_out = 'data/gaia_raw.csv'

    print("[FASE 1] Conectando con los servidores de Gaia DR3...")
    t0 = time.time()

    query_adql = """
    SELECT
        source_id, ra, dec, l, b,
        parallax, parallax_error,
        pmra, pmra_error, pmdec, pmdec_error,
        phot_g_mean_mag, bp_rp
    FROM
        gaiadr3.gaia_source
    WHERE
        l BETWEEN 290 AND 360
        AND b BETWEEN -25 AND 40
        AND parallax BETWEEN 5 AND 12
        AND parallax / parallax_error > 10
        AND phot_g_mean_mag < 15
    """

    print("[FASE 1] Enviando consulta ADQL asíncrona (esto puede tomar entre 1 y 3 minutos)...")
    job = Gaia.launch_job_async(query_adql, dump_to_file=False)
    
    print("[FASE 1] Servidor procesó la consulta. Descargando tabla de resultados...")
    results = job.get_results()
    df_gaia = results.to_pandas()
    
    elapsed = time.time() - t0
    print(f"[FASE 1] Descarga completada en {elapsed:.1f} segundos: {len(df_gaia):,} fuentes extraídas.")

    df_gaia.to_csv(csv_out, index=False)
    print(f"[FASE 1] Datos guardados exitosamente en {csv_out}")

if __name__ == '__main__':
    main()
