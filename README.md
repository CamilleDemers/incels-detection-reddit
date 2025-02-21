# **Comparaison de méthodes pour la détection des *incels* sur Reddit**<br/> 

## Description
Ce dépôt contient les scripts utilisés pour entraîner les modèles décrits dans l'article *Comparaison de méthodes pour la détection du discours des incels sur Reddit*.    
  
Cette étude compare la performance de différents systèmes de détection du discours incel en utilisant une approche d’apprentissage par sacs de communautés. Les expérimentations menées
permettent de comparer l’efficacité de diverses représentations vectorielles pour entraîner différents algorithmes d’apprentissage supervisé à détecter le discours incel dans un corpus de
commentaires provenant de Reddit.

## Installation
```
git clone https://github.com/CamilleDemers/incels-detection-reddit.git
pip install -r requirements.txt
```

## Structure du répertoire
Pour exécuter les scripts, le répertoire devrait arborer la structure suivante :
```
incels-detection-reddit/
├── data/                 
│   │
│   ├── incels/            # Données prétraitées pour la classe "incels"
│   │   │
│   │   └── the-eye_pushshift/  # Données brutes pour la classe "incels"
│   │
│   ├── neutrals/          # Données prétraitées pour la classe "neutres"
│   │   │
│   │   └── the-eye_pushshift/  # Données brutes pour la classe "neutres"
│   │
│   └── training_datasets/  # Jeux de données pour entraîner les modèles
│
├── src/                   # Scripts pour le prétraitement des données et l'entraînement des modèles
│   │
│   ├── utils/             # Fichiers utilitaires utilisés lors de l'exécution des scripts
│
├── results/               
│   │
│   ├── results_test/      # Fichiers de résultats générés en phase de test
│   │      
│   └── results_training/  # Fichiers de résultats générés en phase d'apprentissage
│  
├── word2vec_models/       #  Modèles Word2Vec entraînés et sauvegardés
│   
├── .gitignore             # Dossiers et fichiers à ignorer par git
├── README.md              # Description du projet
└── requirements.txt       # Dépendances nécessaires pour rouler les scripts
```

## Utilisation des scripts
```
# Lire les fichiers de données provenant de The-Eye / PushShift 
python src/lire_pretraiter_incels_zst.py
python src/lire_pretraiter_neutres_zst.py

# Constituer les corpus d'apprentissage et de test 
python src/creation_corpus_apprentissage_test.py
 
# Entraîner les modèles et générer les résultats d'apprentissage
python src/entrainer_sauvegarder_modeles_word2vec.py
python src/detection_incels_apprentissage.py 'tfidf'
python src/detection_incels_apprentissage.py 'word2vec'
python src/detection_incels_apprentissage.py 'sbert'
python src/concatener_resultats_apprentissage.py

# Générer les résultats de test (macro et par classe)
python src/detection_incels_test.py

# Extraire les traits prédictifs des classes "incels" et "neutres"
ipython extraction_coefficients_regression_logistique.ipynb
```

## Informations de contact 
Camille Demers : camille.demers@umontreal.ca

## Citation
```
@article{demers_forest2025,
  author = {Demers, Camille and Forest, Dominic},
  title = {Comparaison de méthodes pour la détection du discours des incels sur Reddit},
  journal = {Revue TAL},
  volume = {65},
  issue = {3},
  year = {2025},
}
```

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


**Références des jeux de données utilisés pour entraîner les modèles** <br/>
- Ribeiro, M. H., Blackburn, J., Bradlyn, B., de Cristofaro, E., Stringhini, G., Long, S., Greenberg, S. et Zannettou, S. (2020). *Dataset for: The Evolution of the Manosphere Across the Web*. Zenodo. https://doi.org/10.5281/zenodo.4007913 
- Baumgartner, J., Zannettou, S., Keegan, B., Squire, M. et Blackburn, J. (2020). *The Pushshift Reddit Dataset*. Proceedings of the International AAAI Conference on Web and Social Media, 14, 830‑839.
- stuck_in_the_matrix, RaiderBDev, Watchful1. (2024). *Reddit comments/submissions 2005-06 to 2023-12*. Academic Torrents. https://academictorrents.com/details/9c263fc85366c1ef8f5bb9da0203f4c8c8db75f4
- Watchful1. (2024). *Subreddit comments/submissions 2005-06 to 2023-12*. Academic Torrents. https://academictorrents.com/details/56aa49f9653ba545f48df2e33679f014d2829c10


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

