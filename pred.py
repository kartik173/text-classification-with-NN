# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 23:29:57 2019

@author: kartik
"""

import pandas as pd
import json

depart_doc=pd.read_csv('data/document_departments.csv')

li=[]
m=depart_doc["Document ID"]
print(len(m))
for i in range(len(m)):
    #print(i)
    st='data/docs/'+str(m[i])+".json"
    li.append([depart_doc["Department"][i]])
    li[i].append(m[i])
    # remove at end
    with open(st) as f:
        dat = json.load(f)
        des=dat["jd_information"]["description"]
        if des=="":
            des=pd.NaT
        li[i].append(des)
    
    
data=pd.DataFrame(columns=["ID","num","Description"],data=li)
# remove num at end

data.count()

data1=data.dropna()

data1.groupby("ID").count()
