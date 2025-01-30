<table>
<tr>                                                                                   
     <th>
         <div style='padding:15px;color:#030aa7;font-size:240%;text-align: center;font-style: italic;font-weight: bold;font-family: Georgia, serif'><a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud">Détection de fraude à la carte de crédit</a></div>
     </th>
     <th><img src="https://raw.githubusercontent.com/rbizoi/MachineLearning/refs/heads/master/images/creditCardFraud-log.jpg" width="96"></th>
 </tr>
<tr>                                                                                   
     <th><img src="https://raw.githubusercontent.com/rbizoi/MachineLearning/refs/heads/master/images/creditCardFraud.png" width="512"></th>
 </tr>    
</table>

```
donnees = pd.read_csv('../donnees/creditCard.csv').dropna().drop(columns='Unnamed: 0')
donnees.Time = donnees.Time.astype('int32')
donnees['DateOrigine'] = pd.to_datetime(donnees.Time, unit='s',origin=pd.Timestamp('2013-09-01T00:00:04.000000000'))
donnees['JourSem']        = donnees.DateOrigine.dt.day_of_week
donnees['Heure24']     = donnees.DateOrigine.dt.hour
donnees['HeureMinute'] = donnees['Heure24'] + donnees.DateOrigine.dt.minute.apply(lambda x: round(x,-1))/100
donnees.reset_index(inplace=True)
donnees.set_index(['index','DateOrigine','Time','Class'],inplace=True)
donnees.to_parquet('../donnees/CreditCardFraud/creditCard.parquet',compression='gzip', engine='pyarrow')
```

