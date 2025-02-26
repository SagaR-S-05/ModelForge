import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import warnings
import json
from pprint import pprint
warnings.filterwarnings("ignore")

# selected_model=input("Enter the model you want to test: ")

all_scores={}

plot_data=[]

models_list=open("./static/Models_list.txt","r").readlines()

for model in models_list:
    if model.startswith("#") or model.startswith("from sklearn.datasets"):
        continue
    if model.endswith("\n"):
        model=model.replace("\n","")
    model.strip()
    exec(model)
    model_name=model.split("import ")[1]
    model_class = eval(model_name)
    if "lightgbm" in model.split("from ")[1]:
        model_object = model_class(verbosity=-1)
    else:
        model_object = model_class()
    
    
    dataset_name=models_list[32]
    dataset_name=dataset_name.replace("\n","")
    exec(dataset_name)
    
    dataset_object=eval(dataset_name.split("import ")[1])
    
    df=pd.DataFrame(dataset_object().data,columns=dataset_object().feature_names)
    
    df['target']=dataset_object().target
    
    X_train,X_test,y_train,y_test=train_test_split(df.drop('target',axis=1),df['target'],test_size=0.2,random_state=42)
    
    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy()
    model_object.fit(X_train_np, y_train_np)
    
    y_pred = model_object.predict(X_test.to_numpy())
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, hue=y_test, palette='Set1')
    plt.title(f'{model_name} Predictions vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.tight_layout()
    plt.savefig(f'../static/images/{model_name}_scatter.png')
    plt.close()
    
    all_scores[model_name]={"score":int(model_object.score(X_test.to_numpy(),y_test.to_numpy())*100),"plot":f'../static/images/{model_name}_scatter.png'}
    plot_data.append({
        'model': model_name,
        'accuracy': all_scores[model_name]["score"]
    })

pprint(all_scores)
with open('model_results.json', 'w') as f:
    json.dump(all_scores, f, indent=4)