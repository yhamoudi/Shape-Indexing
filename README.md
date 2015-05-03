# Shape-Indexing
A binary shape indexing/retrieval system

Nos outils sont développés en python 3. Ils requièrent l'installation des packages suivant (pip3 install nom_du_package --user) : 
  - numpy
  - matplotlib
  - Pillow
  - scipy
  - PyAudio
  - scikit-learn

### Utilisations des scripts

Entendre le son associé à l'image database/beetle-11.pgm :
  ./sound database/beetle-11.pgm

Essayer le mini-jeu basé sur les sons associés aux images : 
  ./soung_game

### Description des fichiers python

- La classe Image (dans src/image.py) contient l'ensemble des outils que nous avons utilisés pour manipuler les images (charger une image au format PGM, appliquer une rotation, appliquer du bruit...)
