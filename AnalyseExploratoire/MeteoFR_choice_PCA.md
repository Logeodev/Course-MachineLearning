# Choix entre les deux méthodes d’agrégations

Dans les deux cas explorés dans les notebooks `MeteoFR_mois_jour_EDA.ipynb` et `MeteoFR_semaine_jour_EDA.ipynb`, on obtient à la fin du processus PCA + règle de Kaiser **4 dimensions** sélectionnées.
Néanmoins, le PCA génère combine en 5 dimensions dans l'analyse comprenant le mois contre 6 dimensions dans l'analyse comprenant les semaines.

Ainsi on évince plus de dimensions dans l'analyse par semaines, avec la règle de Kaiser. Donc je vais préférer pour ma part l'agrégation sur les mois, dont le PCA combine des dimensions de plus grande variabilité, que l'on garde.