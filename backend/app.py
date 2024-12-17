from flask import Flask,render_template,request,session
from werkzeug.datastructures import FileStorage
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from model import *
app=Flask(__name__)

app.secret_key="secret"
app.config["SESSION_PERMANENT"] = False

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/file",methods=["POST"])
def file():
    f=request.files["file"]
    df=load_data(f)
    data=df.copy()
    with open(f"Test-data/{f.filename}.data","wb") as file:
        pickle.dump(data,file)
    session["filename"]=f.filename
    return render_template("index.html",data=data,filename=f.filename,render=True,models=Model_list())

@app.route("/file/result",methods=["POST"])
def file_result():
    filename=session["filename"]
    with open(f"Test-data/{filename}.data","rb") as file:
        data=pickle.load(file)
    feature=request.form["feature"]
    model_type=request.form["model"]
    print(session)
    df=label_encoder(df=data,out=feature)
    if type(df) is not pd.DataFrame:
        return render_template("error.html",error=[feature,df,filename])
    model,model_path=train_model(df,feature,model_type)
    image="."+save_image(df,feature,model)
    image_con=False
    if "Classifier" in model.__class__.__name__:
        image_con="."+conf_mat(df,feature,model)
    print(image,image_con)
    predict=model.predict(df.drop(columns=[feature]))
    actual=df[feature]
    score=model.score(df.drop(columns=[feature]),df[feature])
    max_value=max(df[feature])
    min_value=min(df[feature])
    return render_template( "index.html",data=data,filename=filename,
                            image=image,model_path="."+model_path,
                            score=score,predict=predict,feature=feature,len=len
                            ,activate=True,max_value=max_value,min_value=min_value,
                            confusionMatrix=image_con
                            ,render=True,
                            )


if __name__=="__main__":
    app.run(debug=True)
    # print(Model_list())
    