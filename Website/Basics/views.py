from django.shortcuts import render
import pandas as pd
import sklearn 
from sklearn import tree
import numpy as np
def bill(request):
     if(request.method=="POST"):
        data=request.POST
        d=data.get('diagonal')
        hl=data.get('height_left')
        hr=data.get('height_right')
        ml=data.get('margin_low')
        mu=data.get('margin_up')
        l=data.get('length')
        path="C:\\Users\\mf879\\OneDrive\\Desktop\\Fake_bill_prediction\\fake_bills.csv"
        data1=pd.read_csv(path)
        medianvalue=data1.margin_low.median()
        data1.margin_low=data1.margin_low.fillna(medianvalue)
        inputs=data1.drop('is_genuine','columns')
        outputs=data1.drop(['diagonal','height_left','height_right','margin_low','margin_up','length'],'columns')
        model=tree.DecisionTreeClassifier()
        model.fit(inputs,outputs)
        result=model.predict([[d,hl,hr,ml,mu,l]])
        if result==True:
            info="The currency is real"
        else:
            info="The currecny is fake"
        return render(request,"bill.html",context={'info':info})

     return render(request,'bill.html')

        
# Create your views here.
