# Shape-Indexing
A binary shape indexing/retrieval system

Nos outils sont développés en python 3. Ils requièrent l'installation des packages suivants (pip3 install nom_du_package --user) : 
  - numpy
  - matplotlib
  - Pillow
  - scipy
  - PyAudio
  - scikit-learn

### Utilisations des scripts

Calculer la distance entre 2 images (cette opération prend jusqu'à 30s) :
```
  ./distance.sh database/apple-1.pgm database/apple-3.pgm 
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
    * `python3 eigenvalues.py database --output save.db -- ncpus 4` calcule sur 4 coeurs en parallèle les valeurs propres de chaque image contenue dans le dossier `database` et stocke le résultat final (une map qui associe au nom de chaque image ses valeurs propres) dans `save.db`
 - `database.py` : .......................
 - `distance.py` : utilisé pour calculer la distance entre 2 images (`distance.sh`) : 
    * calculer les valeurs propres puis le descripteur de chaque image 
    * renvoyer la distance entre les 2 descripteurs (la métrique par défaut est cosine)
 - `classify.py` : utilisé pour classer une image dans une des 70 catégories (`classify.sh`)
    * calculer les valeurs propres puis le descripteur de l'image 
    * calculer pour chaque image de la base de donnée la distance de son descripteur à celui de l'image précédente
    * renvoyer la catégorie de l'image de la base de donnée la plus de proche de l'image donnée en entrée
 - `sound.py` : permet de produire un bruit à partir d'une image (`sound.sh`)
 - `sound_game.py` : jeu de reconnaissance d'images à partir des sons (`sound_game.sh`)

