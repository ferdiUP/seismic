# Imagerie sismique

Ces fichiers viennent en complément de mon rapport de stage au sein du LMA de Poitiers (traitement d'images sismiques).
*Les données ne sont pas fournies*.

- correlation.py : calcul de la correlation (au moyen de la transformée de Fourier rapide) entre deux images sismiques (2D) puis entre deux coupes de ces images (1D). L'idée est d'en extraire un champ de déplacement (non réalisé ici).
- denoising.py : débruitage de signaux sismiques 2D par noyau gaussien, puis par variation totale (méthode de A. Chambolle). On trace également le ratio signal/bruit en fonction du poids de débruitage pour cette méthode, l'écart moyen entre les transformées de Hilbert des deux méthodes, l'amplitude de la transformée de Hilbert du signal 2D (et d'une coupe 1D), et la transformée de Fourier du signal.
- dense_opticalflow.py : utilisation d'une méthode dense de flot optique (ici, l'approximation polynomiale de G. Färneback, implémentée dans cv2).
- operators.py : implémentation des deux opérateurs discrets utilisés dans le débruitage par variation totale (gradient, divergence).
- tests.py : tests statistiques sur le bruit.
