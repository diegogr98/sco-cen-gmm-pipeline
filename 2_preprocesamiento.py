import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def main():
    print("[FASE 2] Limpiando datos y estandarizando espacio cinemático...")
    df_raw = pd.read_csv('data/gaia_raw.csv')

    # Remoción de NaNs críticos
    cols_check = ['pmra', 'pmdec', 'parallax', 'bp_rp']
    df_limpio = df_raw.dropna(subset=cols_check).copy()
    print(f"[FASE 2] Fuentes tras remover NaNs: {len(df_limpio):,} (de {len(df_raw):,})")

    # Estandarización cinemática
    var_cinem = ['pmra', 'pmdec', 'parallax']
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(df_limpio[var_cinem])

    df_limpio['pmra_scaled'] = scaled_matrix[:, 0]
    df_limpio['pmdec_scaled'] = scaled_matrix[:, 1]
    df_limpio['parallax_scaled'] = scaled_matrix[:, 2]

    df_limpio.to_csv('data/gaia_preprocessed.csv', index=False)
    print("[FASE 2] Datos procesados guardados en data/gaia_preprocessed.csv")

if __name__ == '__main__':
    main()