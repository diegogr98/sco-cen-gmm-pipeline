import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def main():
    os.makedirs('output', exist_ok=True)
    df = pd.read_csv('data/gaia_preprocessed.csv')
    
    features = ['pmra_scaled', 'pmdec_scaled', 'parallax_scaled']
    X_scaled = df[features].values

    print("[FASE 3] Evaluando número óptimo de componentes con Criterio BIC (k=1..9)...")
    rango_k = np.arange(1, 10)
    valores_bic = []

    for k in rango_k:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(X_scaled)
        valores_bic.append(gmm.bic(X_scaled))

    k_optimo = int(rango_k[np.argmin(valores_bic)])
    print(f"[FASE 3] Componentes óptimos según BIC: k = {k_optimo}")

    # Ajuste del GMM final
    print(f"[FASE 3] Entrenando GMM definitivo con k = {k_optimo}...")
    gmm_final = GaussianMixture(n_components=k_optimo, random_state=42)
    gmm_final.fit(X_scaled)

    # -------------------------------------------------------------
    # SELECCIÓN MATEMÁTICA DEL CLÚSTER MÁS COMPACTO Y DENSO
    # -------------------------------------------------------------
    pesos = gmm_final.weights_
    covarianzas = gmm_final.covariances_

    densidades_fase = []
    for i in range(k_optimo):
        cov = covarianzas[i]
        # Determinante de covarianza (pseudovolumen cinemático)
        det_cov = np.linalg.det(cov)
        
        # Densidad = peso / volumen cinemático
        densidad = pesos[i] / np.sqrt(det_cov) if det_cov > 0 else 0
        densidades_fase.append(densidad)
        
        # Traza de covarianza (dispersión total)
        traza_cov = np.trace(cov)
        print(f"  Componente {i:2d} | Peso: {pesos[i]*100:5.2f}% | Dispersión (Tr): {traza_cov:6.3f} | Densidad Fase: {densidad:10.2f}")

    # El clúster objetivo es aquel que maximiza la densidad en el espacio de fases
    cluster_objetivo = int(np.argmax(densidades_fase))
    print(f"\n[FASE 3] Clúster seleccionado automáticamente: Componente {cluster_objetivo} (Máxima densidad cinemática)")

    # Asignación de etiquetas y probabilidades
    df['gmm_cluster'] = gmm_final.predict(X_scaled)
    probabilidades = gmm_final.predict_proba(X_scaled)
    df['probabilidad_maxima'] = np.max(probabilidades, axis=1)

    # Filtrar el grupo seleccionado matemáticamente con umbral de probabilidad > 0.80
    df_scocen = df[
        (df['gmm_cluster'] == cluster_objetivo) & 
        (df['probabilidad_maxima'] > 0.80)
    ].copy()

    # Cálculo fotométrico de magnitud absoluta en banda G
    df_scocen['magnitud_absoluta_g'] = (
        df_scocen['phot_g_mean_mag'] + 5 + 5 * np.log10(df_scocen['parallax'] / 1000.0)
    )

    print(f"[FASE 3] Fuentes aisladas para Sco-Cen: {len(df_scocen):,}")

    # Guardar gráfico Color-Magnitud
    plt.figure(figsize=(6, 8))
    plt.scatter(df_scocen['bp_rp'], df_scocen['magnitud_absoluta_g'], color='indigo', s=3, alpha=0.6)
    plt.gca().invert_yaxis()
    plt.title(f'Diagrama Color-Magnitud: Sco-Cen (Componente {cluster_objetivo})', weight='bold')
    plt.xlabel('Color ($G_{BP} - G_{RP}$)')
    plt.ylabel('Magnitud Absoluta ($M_G$)')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig('output/cmd_scocen.png', dpi=200)
    plt.close()

    # Guardar datos limpios para la fase 4
    df_scocen.to_csv('data/scocen_isolated.csv', index=False)
    print("[FASE 3] Datos exportados a data/scocen_isolated.csv")

if __name__ == '__main__':
    main()