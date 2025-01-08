<table>
<tr>                                                                                   
     <th>
         <div style='padding:15px;color:#030aa7;font-size:240%;text-align: center;font-style: italic;font-weight: bold;font-family: Georgia, serif'><a href="https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset">Diabetes Health Indicators</a></div>
     </th>
     <th><img src="https://raw.githubusercontent.com/rbizoi/MachineLearning/refs/heads/master/images/diabetes.jpg" width="96"></th>
 </tr>
</table>

<div style='text-align: center'>
<img src="https://raw.githubusercontent.com/rbizoi/MachineLearning/refs/heads/master/images/diabetesindicators.jpeg" width="512">
</div>

<div style='padding:15px;color:#030aa7;font-size:100%;text-align: left;font-family: Georgia, serif'>L'ensemble de données sur les indicateurs de santé du diabète contient des statistiques sur les soins de santé et des informations sur le mode de vie des personnes en général ainsi que leur diagnostic de diabète.</div>

<div style='padding:15px;color:#030aa7;font-size:100%;text-align: left;font-family: Georgia, serif'><a href="https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators">Veuillez vous référer à la page <span style="font-weight: bold; color: blue">UC Irvine Machine Learning Repository</span>
 officielle pour plus de détails.</a></div>

<table>
        <tr>                                                                                   
             <th  style="text-align:left;background-color:#053061;color:white;">Fichier de données</th>
             <th  style="text-align:left;background-color:#053061;color:white;">Description</th>
        </tr> 
        <tr>
            <th  style="text-align:left">diabetes_012_health_indicators_BRFSS2015.csv</th>               
            <th  style="text-align:left">Jeu de données de <span style="font-weight: bold; color: blue">253680</span> observations<br><br>La variable cible Diabetes_012 comporte 3 classes :<span style="font-style: italic; color: blue"><br>0 - absence de diabète ou uniquement pendant la grossesse<br>1 - prédiabète<br>2 - diabète<br></span>Il existe un déséquilibre entre les trois modalités.</th>
        </tr> 
        <tr>
            <th  style="text-align:left">diabetes_binary_5050split_health_indicators_BRFSS2015.csv</th>               
            <th  style="text-align:left">Jeu de données de <span style="font-weight: bold; color: blue">70692</span> observations<br><br>La variable cible Diabetes_binary comporte 2 classes :<span style="font-style: italic; color: blue"><br>0 - absence de diabète ou uniquement pendant la grossesse<br>1 - prédiabète ou diabète<br></span>Il existe un équilibre entre les deux modalités.</th>
        </tr> 
        <tr>
            <th  style="text-align:left">diabetes_binary_health_indicators_BRFSS2015.csv</th>               
            <th  style="text-align:left">Jeu de données de <span style="font-weight: bold; color: blue">253680</span> observations<br><br>La variable cible Diabetes_binary comporte 2 classes :<span style="font-style: italic; color: blue"><br>0 - absence de diabète ou uniquement pendant la grossesse<br>1 - prédiabète ou diabète<br></span>Il existe un déséquilibre entre les deux modalités.</th>
        </tr> 
</table>

<table>
        <tr>                                                                                   
             <th style='padding:15px;color:#030aa7;font-size:150%;text-align: left;font-weight: bold;font-family: Georgia, serif'>diabetes_binary_health_indicators_BRFSS2015.csv</th>
             <th><img src="https://raw.githubusercontent.com/rbizoi/MachineLearning/refs/heads/master/images/diabetes.jpg" width="128"></th>
        </tr>  
<table>

<table>
        <tr>                                                                                   
             <th  style="text-align:left;background-color:#053061;color:white;"> </th>
             <th  style="text-align:left;background-color:#053061;color:white;">Colonne initiale </th>
             <th  style="text-align:left;background-color:#053061;color:white;">Description</th>
        </tr>    
    <tr>
        <th  style="text-align:left">0 </th>                            
        <th  style="text-align:left;color:red;font-style: italic">Diabetes_binary </th> 
        <th  style="text-align:left;color:red;font-style: italic">you have diabetes (0,1)</th>
    </tr>    
    <tr>
        <th  style="text-align:left">1 </th>                            
        <th  style="text-align:left">HighBP </th>                            
        <th  style="text-align:left">Adults who have been told they have high blood pressure by a doctor, nurse, or other health professional (0,1)</th>
    </tr>    
    <tr>
        <th  style="text-align:left">2 </th>                            
        <th  style="text-align:left">HighChol </th>                          
        <th  style="text-align:left">Have you EVER been told by a doctor, nurse or other health professional that your blood cholesterol is high? (0,1)</th>
    </tr>    
    <tr>
        <th  style="text-align:left">3 </th>                            
        <th  style="text-align:left">CholCheck </th>                         
        <th  style="text-align:left">Cholesterol check within past five years (0,1)</th>
    </tr>    
    <tr>
        <th  style="text-align:left">4 </th>                            
        <th  style="text-align:left">BMI </th>                               
        <th  style="text-align:left">Body Mass Index (BMI)</th>
    </tr> 
    <tr>
        <th  style="text-align:left">5 </th>                            
        <th  style="text-align:left">Smoker </th>                            
        <th  style="text-align:left">Have you smoked at least 100 cigarettes in your entire life? [Note : 5 packs = 100 cigarettes] (0,1)</th>
    </tr>  
    <tr>
        <th  style="text-align:left">6 </th>                            
        <th  style="text-align:left">Stroke </th>                            
        <th  style="text-align:left">(Ever told) you had a stroke. (0,1)</th>
    </tr>    
    <tr>
        <th  style="text-align:left">7 </th>                            
        <th  style="text-align:left">HeartDiseaseorAttack </th>              
        <th  style="text-align:left">Respondents that have ever reported having coronary heart disease (CHD) or myocardial infarction (MI) (0,1)</th>
    </tr>  
    <tr>
        <th  style="text-align:left">8 </th>                            
        <th  style="text-align:left">PhysActivity </th>                      
        <th  style="text-align:left">Adults who reported doing physical activity or exercise during the past 30 days other than their regular job (0,1)</th>
    </tr>   
    <tr>
        <th  style="text-align:left">9 </th>                            
        <th  style="text-align:left">Fruits </th>                            
        <th  style="text-align:left">Consume Fruit 1 or more times per day (0,1)</th>
    </tr> 
    <tr>
        <th  style="text-align:left">10 </th>                            
        <th  style="text-align:left">Veggies </th>                           
        <th  style="text-align:left">Consume Vegetables 1 or more times per day (0,1)</th>
    </tr>    
    <tr>
        <th  style="text-align:left">11 </th>                            
        <th  style="text-align:left">HvyAlcoholConsump </th>                 
        <th  style="text-align:left">Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week)(0,1)</th>
    </tr> 
    <tr>
        <th  style="text-align:left">12 </th>                            
        <th  style="text-align:left">AnyHealthcare </th>                     
        <th  style="text-align:left">Do you have any kind of health care coverage, including health insurance, prepaid plans such as HMOs, or government plans such as Medicare, or Indian Health Service? (0,1)</th>
    </tr>  
    <tr>
        <th  style="text-align:left">13 </th>                            
        <th  style="text-align:left">NoDocbcCost </th>                       
        <th  style="text-align:left">Was there a time in the past 12 months when you needed to see a doctor but could not because of cost? (0,1)</th>
    </tr>    
    <tr>
        <th  style="text-align:left">14 </th>                            
        <th  style="text-align:left">GenHlth </th>                           
        <th  style="text-align:left">Would you say that in general your health is  rate (1 ~ 5)</th>
    </tr>    
    <tr>
        <th  style="text-align:left">15 </th>                            
        <th  style="text-align:left">MentHlth </th>                          
        <th  style="text-align:left">Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good? (0 ~ 30)</th>
    </tr>
    <tr>
        <th  style="text-align:left">16 </th>                            
        <th  style="text-align:left">PhysHlth </th>                          
        <th  style="text-align:left">Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? (0 ~ 30)</th>
    </tr>    
    <tr>
        <th  style="text-align:left">17 </th>                            
        <th  style="text-align:left">DiffWalk </th>                          
        <th  style="text-align:left">Do you have serious difficulty walking or climbing stairs? (0,1)</th>
    </tr>
    <tr>
        <th  style="text-align:left">18 </th>                            
        <th  style="text-align:left">Sex </th>                               
        <th  style="text-align:left">Indicate sex of respondent (0,1) (Female or Male)</th>
    </tr>    
    <tr>
        <th  style="text-align:left">19 </th>                            
        <th  style="text-align:left">Age </th>                               
        <th  style="text-align:left">Fourteen-level age category (1 ~ 14)</th>
    </tr>
    <tr>
        <th  style="text-align:left">20 </th>                            
        <th  style="text-align:left">Education </th>                         
        <th  style="text-align:left">What is the highest grade or year of school you completed? (1 ~ 6)</th>
    </tr>
    <tr>
        <th  style="text-align:left">21 </th>                            
        <th  style="text-align:left">Income </th>                            
        <th  style="text-align:left">Is your annual household income from all sources  (If respondent refuses at any income level, code "Refused.") (1 ~ 8)</th>
    </tr>    
</table>