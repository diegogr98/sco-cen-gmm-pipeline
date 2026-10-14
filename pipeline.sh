#!/bin/bash
set -e

echo "=========================================================="
echo " Starting Scorpius-Centaurus GMM Clustering Pipeline "
echo "=========================================================="

python3 1_adquisicion_adql.py
python3 2_preprocesamiento.py
python3 3_clustering_gmm.py
python3 4_subgrupos_arbol.py

echo "=========================================================="
echo " Pipeline executed successfully! Outputs in output/ "
echo "=========================================================="