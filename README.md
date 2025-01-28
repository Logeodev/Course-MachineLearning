<table>
<tr>                                                                                   
     <th>
         <div style='padding:15px;color:#030aa7;font-size:240%;text-align: center;font-style: italic;font-weight: bold;font-family: Georgia, serif'>Formation MachineLearning</div>
     </th>
     <th><img src="https://raw.githubusercontent.com/rbizoi/MachineLearning/refs/heads/master/images/ml_logo.png" width="96"></th>
 </tr>
<tr>                                                                                   
     <th><img src="https://raw.githubusercontent.com/rbizoi/MachineLearning/refs/heads/master/images/Machine-Learning.jpg" width="1024"></th>
 </tr>    
</table>

<b><div style='padding:15px;background-color:#d8dcd6;color:#030aa7;font-size:120%;text-align: left'>Installation</div></b>




<table>
    <tr>                                                                                   
         <th><a href="https://www.anaconda.com/download/success">
               <img src="https://raw.githubusercontent.com/rbizoi/MachineLearning/refs/heads/master/images/anaconda.png" width="512">
             </a>
         </th>
    </tr>    
</table>
<a href="https://www.anaconda.com/download/success">Installation Anaconda</a>

<div style='padding:15px;color:#030aa7;font-size:100%;text-align: left;font-family: Georgia, serif'>Mise à jour des librairies de l’environnement de base</div>

```
conda activate root
conda update --all
python -m pip install --upgrade pip
```
<div style='padding:15px;color:#030aa7;font-size:100%;text-align: left;font-family: Georgia, serif'>Création de l’environnement <b>cours</b> </div>
<div style='padding:15px;color:#030aa7;font-size:100%;text-align: left;font-family: Georgia, serif'><b>Windows</b> </div>

```
# conda remove -n cours --all -y
conda create -n cours -c conda-forge  python==3.10 ipython ipython-sql jupyter notebook numpy pandas pyarrow matplotlib seaborn portpicker biopython flatbuffers redis colour pydot pygraphviz pyyaml pyspark folium scikit-image scikit-learn yellowbrick lightgbm xgboost catboost plotly imgaug tifffile imagecodecs optuna kneed imbalanced-learn

conda activate cours

pip install sql psycopg cx_Oracle opencv-python-headless dtreeviz readfcs
```

<div style='padding:15px;color:#030aa7;font-size:100%;text-align: left;font-family: Georgia, serif'><b>Linux</b> </div>

```
# conda remove -n cours --all -y
conda create -p /home/utilisateur/anaconda3/envs/cours -c conda-forge  python==3.10 ipython ipython-sql jupyter notebook numpy pandas pyarrow matplotlib seaborn portpicker biopython flatbuffers redis colour pydot pygraphviz pyyaml pyspark folium scikit-image scikit-learn yellowbrick lightgbm xgboost catboost plotly imgaug tifffile imagecodecs optuna kneed imbalanced-learn

conda activate cours

pip install sql psycopg cx_Oracle opencv-python-headless dtreeviz readfcs
```


