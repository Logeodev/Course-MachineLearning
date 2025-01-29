<table>
<tr>                                                                                   
     <th>
         <div style='padding:15px;color:#030aa7;font-size:240%;text-align: center;font-style: italic;font-weight: bold;font-family: Georgia, serif'><a href="https://www.kaggle.com/datasets/uciml/mushroom-classification">Classification des Champignons</a></div>
     </th>
     <th><img src="https://raw.githubusercontent.com/rbizoi/MachineLearning/refs/heads/master/images/champignon.jpg" width="96"></th>
 </tr>
</table>

<div style='text-align: center'>
<img src="https://raw.githubusercontent.com/rbizoi/MachineLearning/refs/heads/master/images/schéma_champignon.jpg" width="512">
</div>


<div style='padding:15px;color:#030aa7;font-size:100%;text-align: left;font-family: Georgia, serif'><a href="https://archive.ics.uci.edu/dataset/73/mushroom">Veuillez vous référer à la page <span style="font-weight: bold; color: blue">UC Irvine Machine Learning Repository</span>
 officielle pour plus de détails.</a></div>
 
<table>
    <CAPTION style='padding:15px;color:#030aa7;font-size:150%;text-align: left;font-weight: bold;font-family: Georgia, serif'>mushrooms.csv</CAPTION>    
<tr>                                                                                   
     <th>
        <table>
        <tr>                                                                                   
             <th  style="text-align:left;background-color:#053061;color:white;">Colonne initiale </th>
             <th  style="text-align:left;background-color:#053061;color:white;">Description</th>
        </tr>
        <tr>
            <th  style="text-align:left;color:red;font-style: italic">poisonous</th>               
            <th  style="text-align:left;color:red;font-style: italic">edible=e, poisonous=p</th>
        </tr>    
        <tr>
            <th  style="text-align:left">cap-shape</th>               
            <th  style="text-align:left">bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s<br><img src="https://raw.githubusercontent.com/rbizoi/MachineLearning/refs/heads/master/images/mushroom-cap-shape.jpg" width="512"></th>
        </tr>    
        <tr>
            <th  style="text-align:left">cap-surface</th>             
            <th  style="text-align:left">fibrous=f,grooves=g,scaly=y,smooth=s<br><img src="https://raw.githubusercontent.com/rbizoi/MachineLearning/refs/heads/master/images/mushroom-cap-surface.jpg" width="512"></th>
        </tr>    
        <tr>
            <th  style="text-align:left">cap-color</th>               
            <th  style="text-align:left">brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y</th>
        </tr>    
        <tr>
            <th  style="text-align:left">bruises</th>                 
            <th  style="text-align:left">bruises=t,no=f</th>
        </tr>    
        <tr>
            <th  style="text-align:left">odor</th>                    
            <th  style="text-align:left">almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s</th>
        </tr>    
        <tr>
            <th  style="text-align:left">gill-attachment</th>         
            <th  style="text-align:left">attached=a,descending=d,free=f,notched=n<br><img src="https://raw.githubusercontent.com/rbizoi/MachineLearning/refs/heads/master/images/mushroom-gill-attachment.png" width="512"></th>
        </tr>    
        <tr>
            <th  style="text-align:left">gill-spacing</th>            
            <th  style="text-align:left">close=c,crowded=w,distant=d<br><img src="https://raw.githubusercontent.com/rbizoi/MachineLearning/refs/heads/master/images/mushroom-gill-spacing.jpg" width="512"></th>
        </tr>    
        <tr>
            <th  style="text-align:left">gill-size</th>               
            <th  style="text-align:left">broad=b,narrow=n</th>
        </tr>    
        <tr>
            <th  style="text-align:left">gill-color</th>              
            <th  style="text-align:left">black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y</th>
        </tr>    
        <tr>
            <th  style="text-align:left">stalk-shape</th>             
            <th  style="text-align:left">enlarging=e,tapering=t</th>
        </tr>    
        <tr>
            <th  style="text-align:left">stalk-root</th>              
            <th  style="text-align:left">bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=?<br><img src="https://raw.githubusercontent.com/rbizoi/MachineLearning/refs/heads/master/images/mushroom-stalk.jpg" width="512"></th>
        </tr>    
        <tr>
            <th  style="text-align:left">stalk-surface-above-ring</th>
            <th  style="text-align:left">fibrous=f,scaly=y,silky=k,smooth=s</th>
        </tr>    
        <tr>
            <th  style="text-align:left">stalk-surface-below-ring</th>
            <th  style="text-align:left">fibrous=f,scaly=y,silky=k,smooth=s</th>
        </tr>    
        <tr>
            <th  style="text-align:left">stalk-color-above-ring</th>  
            <th  style="text-align:left">brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y</th>
        </tr>    
        <tr>
            <th  style="text-align:left">stalk-color-below-ring</th>  
            <th  style="text-align:left">brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y</th>
        </tr>    
        <tr>
            <th  style="text-align:left">veil-type</th>               
            <th  style="text-align:left">partial=p,universal=u</th>
        </tr>    
        <tr>
            <th  style="text-align:left">veil-color</th>              
            <th  style="text-align:left">brown=n,orange=o,white=w,yellow=y</th>
        </tr>    
        <tr>
            <th  style="text-align:left">ring-number</th>             
            <th  style="text-align:left">none=n,one=o,two=t</th>
        </tr>    
        <tr>
            <th  style="text-align:left">ring-type</th>               
            <th  style="text-align:left">cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z<br><img src="https://raw.githubusercontent.com/rbizoi/MachineLearning/refs/heads/master/images/mushroom-ring-type.jpg" width="512"></th>
        </tr>    
        <tr>
            <th  style="text-align:left">spore-print-color</th>       
            <th  style="text-align:left">black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y</th>
        </tr>    
        <tr>
            <th  style="text-align:left">population</th>              
            <th  style="text-align:left">abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y</th>
        </tr>    
        <tr>
            <th  style="text-align:left">habitat</th>                 
            <th  style="text-align:left">grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d</th>
        </tr>    
        </table>
     </th>
     <th style="vertical-align: top"><img src="https://raw.githubusercontent.com/rbizoi/MachineLearning/refs/heads/master/images/champignons_structure.jpeg" width="1024"></th>
 </tr>
</table>


```
dictValeurs = {}
dictValeurs['cible']                   ={'comestible':'e','toxique':'p'} 
dictValeurs['cap_shape']               ={'bell':'b','conical':'c','convex':'x','flat':'f','knobbed':'k','sunken':'s'}
dictValeurs['cap_surface']             ={'fibrous':'f','grooves':'g','scaly':'y','smooth':'s'}
dictValeurs['cap_color']               ={'brown':'n','buff':'b','cinnamon':'c','gray':'g','green':'r','pink':'p','purple':'u','red':'e','white':'w','yellow':'y'}
dictValeurs['bruises']                 ={'bruises':'t','no':'f'}
dictValeurs['odor']                    ={'almond':'a','anise':'l','creosote':'c','fishy':'y','foul':'f','musty':'m','none':'n','pungent':'p','spicy':'s'}
dictValeurs['gill_attachment']         ={'attached':'a','descending':'d','free':'f','notched':'n'}
dictValeurs['gill_spacing']            ={'close':'c','crowded':'w','distant':'d'}
dictValeurs['gill_size']               ={'broad':'b','narrow':'n'}
dictValeurs['gill_color']              ={'black':'k','brown':'n','buff':'b','chocolate':'h','gray':'g','green':'r','orange':'o','pink':'p','purple':'u','red':'e','white':'w','yellow':'y'}
dictValeurs['stalk_shape']             ={'enlarging':'e','tapering':'t'}
dictValeurs['stalk_root']              ={'bulbous':'b','club':'c','cup':'u','equal':'e','rhizomorphs':'z','rooted':'r','missing':'?'}
dictValeurs['stalk_surface_above_ring']={'fibrous':'f','scaly':'y','silky':'k','smooth':'s'}
dictValeurs['stalk_surface_below_ring']={'fibrous':'f','scaly':'y','silky':'k','smooth':'s'}
dictValeurs['stalk_color_above_ring']  ={'brown':'n','buff':'b','cinnamon':'c','gray':'g','orange':'o','pink':'p','red':'e','white':'w','yellow':'y'}
dictValeurs['stalk_color_below_ring']  ={'brown':'n','buff':'b','cinnamon':'c','gray':'g','orange':'o','pink':'p','red':'e','white':'w','yellow':'y'}
dictValeurs['veil_type']               ={'partial':'p','universal':'u'}
dictValeurs['veil_color']              ={'brown':'n','orange':'o','white':'w','yellow':'y'}
dictValeurs['ring_number']             ={'none':'n','one':'o','two':'t'}
dictValeurs['ring_type']               ={'cobwebby':'c','evanescent':'e','flaring':'f','large':'l','none':'n','pendant':'p','sheathing':'s','zone':'z'}
dictValeurs['spore_print_color']       ={'black':'k','brown':'n','buff':'b','chocolate':'h','green':'r','orange':'o','purple':'u','white':'w','yellow':'y'}
dictValeurs['population']              ={'abundant':'a','clustered':'c','numerous':'n','scattered':'s','several':'v','solitary':'y'}
dictValeurs['habitat']                 ={'grasses':'g','leaves':'l','meadows':'m','paths':'p','urban':'u','waste':'w','woods':'d'}
```

