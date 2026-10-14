import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

def main():
    os.makedirs('output', exist_ok=True)
    df_scocen = pd.read_csv('data/scocen_isolated.csv')
    print("[FASE 4] Segmentando subgrupos morfológicos (k=3)...")

    features_sub = ['l', 'b', 'parallax']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_scocen[features_sub])

    # Ajuste de GMM para 3 subgrupos
    gmm_sub = GaussianMixture(n_components=3, random_state=42, n_init=10)
    df_scocen['raw_cluster'] = gmm_sub.fit_predict(X_scaled)
    medias_reales = scaler.inverse_transform(gmm_sub.means_)

    # -------------------------------------------------------------
    # MAPEO AUTOMÁTICO BASADO EN COORDENADAS FÍSICAS (Longitud l)
    # -------------------------------------------------------------
    # Ordenamos los índices de los clústeres de menor a mayor longitud media
    orden_l = np.argsort(medias_reales[:, 0])  # [idx_menor_l, idx_medio_l, idx_mayor_l]

    mapa_subgrupos = {
        orden_l[0]: {'nombre': 'Lower Centaurus-Crux', 'color': 'orange'},
        orden_l[1]: {'nombre': 'Upper Centaurus-Lupus', 'color': 'green'},
        orden_l[2]: {'nombre': 'Upper Scorpius', 'color': 'indigo'}
    }

    df_scocen['subgrupo_nombre'] = df_scocen['raw_cluster'].map(lambda x: mapa_subgrupos[x]['nombre'])
    df_scocen['subgrupo_color'] = df_scocen['raw_cluster'].map(lambda x: mapa_subgrupos[x]['color'])
    df_scocen['subgrupo_id'] = df_scocen['raw_cluster'].map(
        lambda x: 0 if x == orden_l[0] else (1 if x == orden_l[1] else 2)
    )

    print("\n[FASE 4] Centroides espaciales identificados:")
    for idx_orig in orden_l:
        nombre = mapa_subgrupos[idx_orig]['nombre']
        l_m, b_m, plx_m = medias_reales[idx_orig]
        print(f"  - {nombre:<23}: l={l_m:6.2f}° | b={b_m:6.2f}° | plx={plx_m:5.2f} mas")

    # -------------------------------------------------------------
    # INTERPRETABILIDAD CON ÁRBOL DE DECISIÓN
    # -------------------------------------------------------------
    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(df_scocen[features_sub].values, df_scocen['subgrupo_id'].values)

    print("\n[FASE 4] Importancia relativa de variables en la segmentación:")
    for var, imp in zip(features_sub, clf.feature_importances_):
        print(f"  - {var:<12}: {imp*100:5.2f}%")

    # -------------------------------------------------------------
    # VISUALIZACIÓN FINAL
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for sub_id in [2, 1, 0]:  # Graficar en orden US -> UCL -> LCC
        sub_df = df_scocen[df_scocen['subgrupo_id'] == sub_id]
        nombre = sub_df['subgrupo_nombre'].iloc[0]
        color = sub_df['subgrupo_color'].iloc[0]

        # Panel 1: Espacial
        ax1.scatter(sub_df['l'], sub_df['b'], c=color, s=4, alpha=0.7, label=f"{nombre} (n={len(sub_df):,})")

        # Panel 2: CMD
        ax2.scatter(sub_df['bp_rp'], sub_df['magnitud_absoluta_g'], c=color, s=3, alpha=0.4)

    # Configuración Panel 1
    ax1.set_title('Separación Geográfica en el Plano Galáctico', weight='bold')
    ax1.set_xlabel('Longitud Galáctica ($l$) [°]')
    ax1.set_ylabel('Latitud Galáctica ($b$) [°]')
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc='upper right', framealpha=0.9)

    # Configuración Panel 2
    ax2.set_title('Estratificación en el CMD (Pre-Secuencia Principal)', weight='bold')
    ax2.set_xlabel('Color ($G_{BP} - G_{RP}$)')
    ax2.set_ylabel('Magnitud Absoluta ($M_G$)')
    ax2.invert_yaxis()
    ax2.set_xlim(-0.5, 4.0)
    ax2.set_ylim(12, -4)
    ax2.grid(True, linestyle=':', alpha=0.5)

    plt.suptitle('Segmentación Automatizada de Subgrupos en Scorpius-Centaurus', fontsize=13, weight='bold')
    plt.tight_layout()
    plt.savefig('output/subgrupos_scocen.png', dpi=200)
    plt.close()

    # Guardar conjunto clasificado final
    df_scocen.to_csv('data/scocen_final_classified.csv', index=False)
    print("\n[FASE 4] Proceso finalizado. Gráfico generado en output/subgrupos_scocen.png")

if __name__ == '__main__':
    main()