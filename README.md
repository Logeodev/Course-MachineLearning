<table>
<tr>                                                                                   
     <th>
         <div>Formation MachineLearning</div>
     </th>
     <th><img src="https://raw.githubusercontent.com/rbizoi/MachineLearning/refs/heads/master/images/ml_logo.png" width="96"></th>
 </tr>
<tr>                                                                                   
     <th><img src="https://raw.githubusercontent.com/rbizoi/MachineLearning/refs/heads/master/images/Machine-Learning.jpg" width="1024"></th>
 </tr>    
</table>

<b><div>Installation</div></b>




<table>
    <tr>                                                                                   
         <th><a href="https://www.anaconda.com/download/success">
               <img src="https://raw.githubusercontent.com/rbizoi/MachineLearning/refs/heads/master/images/anaconda.png" width="512">
             </a>
         </th>
    </tr>    
</table>
<br>
<b></b><a href="https://www.anaconda.com/download/success">Installation Anaconda</a></b>
<br>
<div>Mise à jour des librairies de l’environnement <b>base</b></div>

```
conda activate root
conda update --all
python -m pip install --upgrade pip
```
<div>Création de l’environnement <b>cours</b> </div>
<br>
<div><b>Windows</b> </div>
<br>

```
# conda remove -n cours --all -y
conda create -n cours -c conda-forge  python==3.10 ipython ipython-sql jupyter notebook numpy pandas pyarrow matplotlib seaborn portpicker biopython flatbuffers redis colour pydot pygraphviz pyyaml pyspark folium scikit-image scikit-learn yellowbrick lightgbm xgboost catboost plotly imgaug tifffile imagecodecs optuna kneed imbalanced-learn

conda activate cours

pip install sql psycopg cx_Oracle opencv-python-headless dtreeviz readfcs
```
<br>
<div><b>Linux</b> </div>
<br>

```
# conda remove -n cours --all -y
conda create -p /home/utilisateur/anaconda3/envs/cours -c conda-forge  python==3.10 ipython ipython-sql jupyter notebook numpy pandas pyarrow matplotlib seaborn portpicker biopython flatbuffers redis colour pydot pygraphviz pyyaml pyspark folium scikit-image scikit-learn yellowbrick lightgbm xgboost catboost plotly imgaug tifffile imagecodecs optuna kneed imbalanced-learn

conda activate cours

pip install sql psycopg cx_Oracle opencv-python-headless dtreeviz readfcs
```


