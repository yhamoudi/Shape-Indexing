# Shape-Indexing
A binary shape indexing/retrieval system

### Installation

Nos outils sont développés en python 3. Ils requièrent l'installation des packages suivants (pip3 install nom_du_package --user) : 
  - numpy
  - matplotlib
  - Pillow
  - scipy
  - PyAudio
  - scikit-learn

Notre programme nécessite également une base de données de valeurs propres `eigenvalues.db` située dans le dossier `eigenvalues`. Il y a 3 possibilités pour récupérer cette base : 
  - la reconstruire grâce à la commande : `python3 src/eigenvalues.py database --output eigenvalues/eigenvalues.db --ncpus 4` (remplacer 4 par le nombre de coeurs disponibles). Cette opération prend plus d'une heure.
  - télécharger la base de donnée et l'ajouter manuellement : https://filesender.ens-lyon.fr/?vid=6feca009-e562-49c8-fbbb-000078889ff9
  - télécharger le dépôt GitHub (qui intègre la base de données) : https://github.com/yhamoudi/Shape-Indexing

Il est également conseillé d'ajouter les images (au format pgm) dans le dossier database.

### Utilisations des scripts

Calculer la similarité (0 : peu similaires, 1 : très similaires) entre 2 images (cette opération prend jusqu'à 30s) :
```
  ./similarity.sh database/apple-1.pgm database/apple-3.pgm 
```

Calculer une probabilité d'appartenance à chaque classe pour une image donnée (cette opération prend jusqu'à 30s) :
```
  ./classification.sh database/apple-5.pgm 
```

Entendre le son associé à l'image database/beetle-11.pgm :
```
  ./sound.sh database/beetle-11.pgm
```

Essayer le mini-jeu de reconnaissance des sons : 
```
  ./soung_game.sh
```

### Description des fichiers python (dans src/)

 - `image.py` : contient la classe Image et l'ensemble des outils permettant de manipuler les images (charger une image au format PGM, appliquer une  rotation, ajouter du bruit...)
 - `eigenvalues.py` : permet de calculer les valeurs propres d'un ensemble d'images : 
    * `compute_eigenvalues` prend en entrée une image au format numpy array et renvoie la liste de ses valeurs propres
    * `python3 src/eigenvalues.py database --output save.db --ncpus 4` calcule sur 4 coeurs en parallèle les valeurs propres de chaque image contenue dans le dossier `database` et stocke le résultat final (une map qui associe au nom de chaque image ses valeurs propres) dans `save.db`
 - `database.py` : permet d'évaluer les performances de l'algorithme
    * `python3 src/database.py eigenvalues/eigenvalues.db --classes classes.csv --niters 20` : répartit les images en un train set (environ 80% des images) et un test set (images restantes). Calcule pour chaque image du test set l'image du train set la plus proche. Cette expérience est répétée 20 fois, et renvoie la moyenne du nombre de réussites (images correctement classifiées) et la variance.
 - `similarity.py` : utilisé pour calculer la similarité entre 2 images (`similarity.sh`) : 
    * calculer les valeurs propres puis le descripteur de chaque image 
    * renvoyer la similarité entre les 2 descripteurs (par défaut : 1 - cosine/2)
 - `classify.py` : utilisé pour classer une image dans une des 70 catégories (`classify.sh`)
    * calculer les valeurs propres puis le descripteur de l'image 
    * calculer pour chaque image de la base de donnée la distance de son descripteur à celui de l'image précédente
    * renvoyer la catégorie de l'image de la base de donnée la plus de proche de l'image donnée en entrée
 - `sound.py` : permet de produire un bruit à partir d'une image (`sound.sh`)
 - `sound_game.py` : jeu de reconnaissance d'images à partir des sons (`sound_game.sh`)

