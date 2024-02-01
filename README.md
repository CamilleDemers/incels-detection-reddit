**Détection automatique de propos misogynes en ligne: le cas de Reddit et des communautés Incels**<br/> 
Camille Demers, Isabelle Fontaine, Audrée Frappier et Dominic Forest

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Ce dépôt contient : 
- L'entièreté du corpus d'apprentissage et de test prétraitée et filtré 
- Les scripts utilisés pour lire, filtrer, nettoyer et échantillonner aléatoirement différentes proportions de données d'apprentissage
- L'ensemble des modèles développés avec *sk-learn* à partir de différentes proportions de données issues de chacune des catégories
- Le détail des paramètres testés en phase d'apprentissage : 
   - Les ratios "incels"/"neutres" testés
   - Les critères relatifs au filtrage  du corpus d'apprentissage (liste d'exclusion de termes ; fréquence minimale d'occurrence, nombre maximal de traits discriminants à retenir, etc.)
   - Les différents nombre de traits discriminants testés 
   - Les paramètres des algorithmes utilisés lors des expérimentations (statistiques utilisées pour la pondération des traits discriminants, valeurs de K testées pour l'algorithme des K plus proches voisins, etc.)

**Jeux de données utilisés pour entraîner les modèles** <br/>
Ribeiro, M. H., Blackburn, J., Bradlyn, B., de Cristofaro, E., Stringhini, G., Long, S., Greenberg, S. et Zannettou, S. (2020). *Dataset for: The Evolution of the Manosphere Across the Web* (version 1.0) [Ensemble de données]. Zenodo. https://doi.org/10.5281/zenodo.4007913 

Baumgartner, J., Zannettou, S., Keegan, B., Squire, M. et Blackburn, J. (2020). The Pushshift Reddit Dataset. Proceedings of the International AAAI Conference on Web and Social Media, 14, 830‑839.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

