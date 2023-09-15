**Automatic Misogyny Detection on Social Media Platforms: The Case of Reddit and Incels communities**<br/> 
Camille Demers, Isabelle Fontaine, Audrée Frappier et Dominic Forest

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Ce dépôt contient : 
- L'entièreté du corpus d'apprentissage et de test, de sa forme brute prétraitée (lue et convertie en sous-fichiers au format tableur) jusqu'à l'échantillon filtré que nous avons retenu pour développer notre modèle. 
- Les scripts utilisés pour lire, filtrer, nettoyer et échantillonner aléatoirement différentes proportions de données d'apprentissage
- L'ensemble des modèles développés dans l'outil Wordstat (fichiers .ppj) à partir de différentes proportions de données issues de chacune des catégories
- Le détail des paramètres testés en phase d'apprentissage : 
   - Les ratios "incels"/"neutres" testés
   - Les critères relatifs au filtrage  du corpus d'apprentissage (liste d'exclusion de termes ; fréquence minimale d'occurrence, nombre maximal de traits discriminants à retenir, etc.)
   - Les différents nombre de traits discriminants testés 
   - Les paramètres des algorithmes utilisés lors des expérimentations (statistiques utilisées pour la pondération des traits discriminants, valeurs de K testées pour l'algorithme des K plus proches voisins, etc.)
- L'ensemble des résultats au test mené sur différents ratios de données "incels"/"neutres" avec les deux classifieurs KNN retenus (où K = 1 et 2, respectivement)
- Les différentes annexes mentionnées dans le corps du texte de l'article

**Jeu de données utilisé pour développer notre modèle** <br/>
Ribeiro, M. H., Blackburn, J., Bradlyn, B., de Cristofaro, E., Stringhini, G., Long, S., Greenberg, S. et Zannettou, S. (2020). *Dataset for: The Evolution of the Manosphere Across the Web* (version 1.0) [Ensemble de données]. Zenodo. https://doi.org/10.5281/zenodo.4007913 

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Structure des dossiers contenus dans ce dépôt
Des précisions sur le contenu de chaque dossier&nbsp;/&nbsp;fichier sont indiquées au besoin (** Précisions)<br/>
(Utiliser les flèches gauche et droite du clavier pour faire défiler le texte horizontalement)

```text
+---1-corpus
|   |   tailles_corpus_appr_test.xlsx  
|   |   ** Fichier tableur indiquant les ratios de données incels/neutres utilisées dans chacun des corpus d'apprentissage et de test
|   |   
|   +---corpus_apprentissage
|   |   |   neutres_60k_nettoyes.xlsx
|   |   |   ** Fichier tableur contenant 60 000 données neutres, utilisées pour constituer les différents ratios de données neutres au sein des différents corpus d'apprentissage et de test 
|   |   |   ribeiro_subreddits_categories.xlsx
|   |   |   ** Liste des subreddits catégorisés comme étant significativement fréquentés par les membres des communautés incels selon les travaux de Ribeiro et al. (2020)
|   |   | 
|   |   +---corpus_incel_ribeiro_filtre
|   |   |   ** 60 fichiers au format tableur contenant approximativement 500 000 données incels issues du corpus de Ribeiro et al. (2020) ; filtrées selon le subreddits (doivent appartenir à un
|   |   |      subreddit catégorisé "incel" et la date de publication (2015-2019)
|   |   | 
|   |   |       incels_1.xlsx
|   |   |       incels_2.xlsx
|   |   |       incels_3.xlsx
   ...               ...
|   |   |       incels_57.xlsx
|   |   |       incels_58.xlsx
|   |   |       incels_59.xlsx
|   |   |       
|   |   \---corpus_incel_ribeiro_filtre_samples
|   |       ** Ce dossier contient des fichiers au format tableur contenant des échantillons de taille variable de données incels extraits aléatoirement des 58 premiers fichiers 
|   |	       du dossier --corpus_incels_ribeiro_filtre afin de constituer les différents ratios de données incels/neutre de nos corpus d'apprentissage (le 59e fichier sert à mettre
|   | 	       de côté un ensemble de données incels pour la phase de test)
|   |   
|   |           corpus_incels_10k.xlsx
|   |           corpus_incels_20k.xlsx
|   |           corpus_incels_30k.xlsx
|   |           corpus_incels_40k.xlsx
|   |           corpus_incels_45k.xlsx
|   |           corpus_incels_50k.xlsx
|   |           
|   \---corpus_test
|   |   ** Ce dossier contient des fichiers au format tableur contenant les données utilisées pour la constitution des différents ratios de données de nos corpus test. 
|   |      Les données incels proviennent du 59e fichier du dossier --corpus_incel_ribeiro_filtre qui a été mis de côté pour la phase de test;
|   |      Les 15 000 données incels ont été exclues du fichier neutres_60k_nettoyes.xlsx en vue de la phase de test
|
|           test_incels_10k.xlsx
|           test_neutres_15k_nettoyes.xlsx
|           
+---2-experimentation
|   |   exclusion_v2.stop
|   |   ** Ce fichier .stop contient la liste des mots fonctionnels, des artefacts HTML et des autres types de données bruitées qui ont été exclus du corpus à l'étape de filtrage
|   | 
|   |   parametres_experimentations.xlsx
|   |   ** Ce classeur documente les paramètres utilisés dans Wordstat pour mener les expérimentations lors de la phase d'apprentissage
|   |
|   |   resultats_apprentissage_graphs.xlsx
|   |   ** Ce classeur contient les graphiques synthétisant les résultats obtenus en phase d'apprentissage pour l'ensemble des différents ratios de données testés avec différents nombres de traits discriminants et différents algorithmes ; il synthétise les résultats contenus dans les fichiers table.xlsx des sous-dossiers suivant
|   |
|   |   ** Chacun des sous-dossiers suivant contient les modèles d'apprentissage développés dans Wordstat pour les différents ratio de données incels/neutres testés : 
|   |      > Le fichier .xlsx contient les données brutes (post, catégorie) du corpus d'apprentissage généré pour le ratio correspondant ; 
|   |      > Le fichier .ppj correspond au projet créé par Wordstat une fois les données importées
|   |      > Le fichier table.xlsx contient les résultats de l'expérimentation menée (exactitude, rappel, précision, mesure-F) pour chaque algorithme testé (Naive Bayes, différentes valeurs de KNN) et selon un nombre variable de traits discriminants retenus (allant de 100 à 3700)              
|   | 
|   |   ** Les fichiers ont étés nommés de manière à indiquer le ratio de données incels et de données neutres qu'ils contiennent ainsi que la taille du corpus (en nombre de posts) ;     
|   |   Par exemple, le fichier app_10_90_50k.ppj contient 10% de données incels, 90% de données neutres et compte 50 000 (50K) commentaires Reddit au total    
|   | 
|   +---10_90_500k
|   |   ** Ce sous-dossier a dû être exclu parce qu'il était trop volumineux pour être hébergé sur Github ; il est possible de nous contacter pour en avoir une copie au besoin.
|   |       appr_10_90_500k.ppj
|   |       appr_10_90_500k.xlsx
|   |       
|   |       
|   +---10_90_50k
|   |       app_10_90_50k.ppj
|   |       app_10_90_50k.xlsx
|   |       app_10_90_50k_table.xlsx
|   |       
|   +---20_80_50k
|   |       appr_20_80_50k.ppj
|   |       appr_20_80_50k.xlsx
|   |       appr_20_80_50k_table.xlsx
|   |       
|   +---30_70_50k
|   |       appr_30_70_50k.ppj
|   |       appr_30_70_50k.xlsx
|   |       appr_30_70_50k_table.xlsx
|   |       
|   +---50_50_50k
|   |       appr_50_50_50k.ppj
|   |       appr_50_50_50k.xlsx
|   |       appr_50_50_50k_table.xlsx
|   |       
|   +---70_30_50k
|   |       appr_70_30_50k.ppj
|   |       appr_70_30_50k.xlsx
|   |       appr_70_30_50k_table.xlsx  
|   |       
|   \---90_10_50k
|           appr_90_10_50k.ppj
|           appr_90_10_50k.xlsx
|           appr_90_10_50k_table.xlsx         
|           
+---3-test
|   |   resultats_test_50k.xlsx
|   |   ** Ce classeur contient l'ensemble des résultats (exactitude, rappel, précision, mesure-F) des test menés sur les deux systèmes retenus, soit les classifieurs KNN, où K = 1 et K = 2 respectivement, tenant compte de 3200 traits discriminants, pour différentes proportions de données incels testées.
|   |
|   |   resultats_test_50k-500k.xlsx
|   |   ** Ce classeur contient les résultats comparant les performances obtenues par le système retenu (KNN = 1, 3200 trait discrimimnants, 10% de données incels) entraîné avec 50k données d'apprentissage vs 500k données d'apprentissage.
|   |
|   +---donnees_test
|   |   ** Chacun des sous-dossiers suivant contient les corpus de données mises de côtés pour la phase de test, pour les différents ratio de données incels/neutres testés 
|   |      > Le fichier .xlsx contient les données brutes (post, catégorie) du corpus test généré pour le ratio correspondant ; 
|   |      > Le fichier .ppj correspond au projet créé par Wordstat une fois les données importées  
|   |
|   |   +---10_90_10k
|   |   |       test_10_90_10k.ppj
|   |   |       test_10_90_10k.xlsx      
|   |   |       
|   |   +---20_80_10k
|   |   |       test_20_80_10k.ppj
|   |   |       test_20_80_10k.xlsx     
|   |   |       
|   |   +---30_70_10k
|   |   |       test_30_70_10k.ppj
|   |   |       test_30_70_10k.xlsx      
|   |   |       
|   |   +---50_50_10k
|   |   |       test_50_50_10k.ppj
|   |   |       test_50_50_10k.xlsx
|   |   |       
|   |   |       
|   |   +---70_30_10k
|   |   |       test_70_30.ppj
|   |   |       test_70_30.xlsx
|   |   |       
|   |   |       
|   |   \---90_10_10k
|   |           test_90_10.ppj
|   |           test_90_10.xlsx          
|   |           
|   +---resultats_KNN_1
|   |   ** Ce dossier contient les fichiers au format tableur des résultats au test pour les différents ratio de données incels/neutres testés, en utilisant l'algorithme KNN, où K= 1, avec 3200 traits discriminants.
|   |      Chaque classeur contient la liste des publications sur lesquelles l'algorithme a été testé, la catégorie prédite par l'algorithme, sa catégorie réelle ainsi que les mesures d'exactitude, 
|   | 	   de rappel, de précision et de mesure F résultantes.
|   |      
|   |       knn-1_resultats_10_90_10k.xlsx
|   |       knn-1_resultats_10_90_500k_10k.xlsx (nous avons ici testé un apprentissage basé sur 500 000 données plutôt que 50 000)
|   |       knn-1_resultats_20_80_10k.xlsx
|   |       knn-1_resultats_30_70_10k.xlsx
|   |       knn-1_resultats_50_50_10k.xlsx
|   |       knn-1_resultats_70_30_10k.xlsx
|   |       knn-2_resultats_90_10_10k.xlsx
|   |       
|   \---resultats_KNN_2
|   |   ** Ce dossier contient les fichiers au format tableur des résultats au test pour les différents ratio de données incels/neutres testés, en utilisant l'algorithme KNN, où K= 1, avec 3200 traits discriminants.
|   |      Chaque classeur contient la liste des publications sur lesquelles l'algorithme a été testé, la catégorie prédite par l'algorithme, sa catégorie réelle ainsi que les mesures d'exactitude, 
|   | 	   de rappel, de précision et de mesure F résultantes.
|   | 
|           knn-2_resultats_10_90_10k.xlsx
|           knn-2_resultats_20_80_10k.xlsx
|           knn-2_resultats_30_70_10k.xlsx
|           knn-2_resultats_50_50_10k.xlsx
|           knn-2_resultats_70_30_10k.xlsx
|           knn-2_resultats_90_10_10k.xlsx  
|         
+---4-annexes
|   |   ** Ce dossier contient les annexes mentionnées dans l'article, soit les listes des subreddits incels et neutres d'où proviennent les données d'apprentissage ;
|   | 
|       liste_subreddits_incels.xlsx
|       liste_subreddits_neutres.xlsx
|       
\---5-scripts
|   |  ** Ce dossier contient l'ensemble des scripts qui ont été utilisés lors de la constitution du corpus 
|   |  -Un script Python permettant de lire un fichier au format .ndjson très volumineux (pour les données incels) et de le morceler en un nombre paramétrable de sous-fichiers au format .CSV 
|   |  -Une macro-commande VBA permettant de lire en boucle les données provenant de chacun des fichiers CSV résultant de l'applicaiton du premier script et de les filtrer afin de ne retenir que les subreddits pertinents ainsi que
|   |   la plage temporelle d'intérêt (2015-2019)
|   |  -Une macro-commande VBA permettant de colliger au sein d'un même fichier un nombre paramétrable de données extraites aléatoirement des fichiers filtrés résultant de l'application de la première macro-commande
|   |  -Un script Python permettant de lire les N premières lignes d'une archive .bzt très volumineuse (pour les données neutres) et de les stocker dans un fichier au format CSV 
|   |
        1_lire_chunker_ndjson.py
        2_filtrer_incels_2015-2019.bas
        3_sampler_incels.bas
        4-bzt_n-lines_to-csv_neutres.py
```

