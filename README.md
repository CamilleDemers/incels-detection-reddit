# **Détecter le discours des *incels* sur Reddit : une approche par sac de communautés**<br/> 

## Description
Ce dépôt contient les scripts utilisés pour lire, nettoyer et échantillonner les données utilisées pour entraîner les modèles décrits dans l'article *Détecter le discours des incels sur Reddit : une approche par sac de communautés*.

Il contient également les fichiers de résultats obtenus pour chacune des configurations testées en phase d'apprentissage et de test. 

## Installation
```
git clone https://github.com/CamilleDemers/incels-detection-reddit.git
pip install -r requirements.txt
```

## Structure du répertoire
incels-detection-reddit/
├── data/                   # Dossier est absent du répertoire git, mais il devrait aborder la structure suivante
│   │
│   ├── incels/             # Données prétraitées pour la classe "incels"
│   │   │
│   │   └── the-eye_pushshift/  # Données brutes pour la classe "incels"
│   │
│   ├── neutrals/           # Données prétraitées pour la classe "neutres"
│   │   │
│   │   └── the-eye_pushshift/  # Données brutes pour la classe "neutres"
│   │
│   └── training_datasets/  # Jeux de données pour entraîner les modèles
│
├── src/                    # Scripts nécessaires au prétraitement des données et à l'entraînement des modèles
│   │
│   ├── utils/              # Fichiers utilitaires utilisés lors de l'exécution des scripts (ex. mots fonctionnels)
│
├── results/                # Fichiers de résultats générés lors de l'exécution des scripts
│
├── .gitignore              # Dossiers et fichiers à ignorer par git
├── README.md               # Description du projet
└── requirements.txt        # Dépendances nécessaires pour rouler les scripts


## Utilisation des scripts
```
# Lire les fichiers de données provenant de The-Eye / PushShift 
python scripts/read_incels_zst_to_csv.py
python scripts/read_incels_zst_to_csv.py

# Constituer les corpus d'apprentissage et de test en échantillonnat les données incels/neutres 
python scripts/build_train_test_datasets.py
 
# Entraîner les modèles et générer les résultats d'apprentissage et de test
python scripts/incels_detection_reddit.py

# Extraire les traits prédictifs des classes "incels" et "neutres"
python scripts/get_most_predictive_features.py
```

## Informations de contact 
- Camille Demers : camille.demers@umontreal.ca

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


**Références des sources des jeux de données utilisés pour entraîner les modèles** <br/>
- Ribeiro, M. H., Blackburn, J., Bradlyn, B., de Cristofaro, E., Stringhini, G., Long, S., Greenberg, S. et Zannettou, S. (2020). *Dataset for: The Evolution of the Manosphere Across the Web* (version 1.0) [Ensemble de données]. Zenodo. https://doi.org/10.5281/zenodo.4007913 
- Baumgartner, J., Zannettou, S., Keegan, B., Squire, M. et Blackburn, J. (2020). The Pushshift Reddit Dataset. Proceedings of the International AAAI Conference on Web and Social Media, 14, 830‑839.
- stuck_in_the_matrix, R., Watchful1. (2024). Reddit comments/submissions 2005-06 to 2023-12. Academic Torrents. https://academictorrents.com/details/9c263fc85366c1ef8f5bb9da0203f4c8c8db75f4
- Watchful1. (2024). Subreddit comments/submissions 2005-06 to 2023-12. Academic Torrents. https://academictorrents.com/details/56aa49f9653ba545f48df2e33679f014d2829c10


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

