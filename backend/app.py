from flask import Flask,render_template,request,session
from werkzeug.datastructures import FileStorage
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor   
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
app=Flask(__name__)

app.secret_key="secret"
app.config["SESSION_PERMANENT"] = False

def load_data(file:FileStorage):
    if file.filename.endswith(".csv") or file.filename.endswith(".xls"):
        df=pd.read_csv(file)
    elif file.filename.endswith(".xlsx"):
        df=pd.read_excel(file,engine="openpyxl")
    else:
        df=pd.DataFrame(["please upload a csv or excel file"],columns=["Invalid file type"])
    return df

def label_encoder(df,out):
    if out not in df.columns:
        print(df.columns)
        return df.columns
    for col in df.columns:
        if df[col].dtype=="object":
            df[col]=df[col].astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    return df

def init_model(model_type):
    models_list=open("../Models_list.txt","r").readlines()
    for model in models_list:
        if model.startswith("#") or model.startswith("from sklearn.datasets"):
            continue
        if model.endswith("\n"):
            model=model.replace("\n","")
        model.strip()
        if model_type in model:
            exec(model)
            if "lightgbm" in model:
                return eval(model_type)(verbosity=-1)
            else:
                return eval(model_type)()
    return RandomForestRegressor()

def train_model(df,out,model_type):
    model=init_model(model_type)
    model.fit(df.drop(columns=[out]),df[out])
    model_path=f"./static/ML_files/{model.__class__.__name__}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    return model,model_path

def save_image(df,out,model):
    plt.ioff()
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df[out], y=model.predict(df.drop(columns=[out])), hue=df[out], palette='Set1')
    model_name=model.__class__.__name__
    plt.title(f'{model_name} Predictions vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.tight_layout()
    image_name=f"./static/images/{model_name}_scatter.png"
    plt.savefig(image_name)
    plt.close()
    return image_name


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/file",methods=["POST"])
def file():
    f=request.files["file"]
    df=load_data(f)
    data=df.copy()
    with open(f"data/{f.filename}.data","wb") as file:
        pickle.dump(data,file)
    session["filename"]=f.filename
    return render_template("file.html",data=data,filename=f.filename)

@app.route("/file/result",methods=["POST"])
def file_result():
    filename=session["filename"]
    with open(f"data/{filename}.data","rb") as file:
        data=pickle.load(file)
    feature=request.form["feature"]
    model_type=request.form["model"]
    print(session)
    df=label_encoder(df=data,out=feature)
    if type(df) is not pd.DataFrame:
        return render_template("error.html",error=[feature,df,filename])
    model,model_path=train_model(df,feature,model_type)
    image="."+save_image(df,feature,model)
    print(image)
    predict=model.predict(df.drop(columns=[feature]))
    actual=df[feature]
    score=model.score(df.drop(columns=[feature]),df[feature])
    max_value=max(df[feature])
    min_value=min(df[feature])
    return render_template( "file.html",data=data,filename=filename,
                            image=image,model_path="."+model_path,
                            score=score,predict=predict,feature=feature,len=len
                            ,activate=True,max_value=max_value,min_value=min_value)


if __name__=="__main__":
    app.run(debug=True)
    