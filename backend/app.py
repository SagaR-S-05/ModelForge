from flask import Flask,render_template,request,session
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
from model import *
app=Flask(__name__)

app.secret_key="secret"
app.config["SESSION_PERMANENT"] = False

@app.route("/")
def index():
    return render_template("upload_page.html")

@app.route("/file",methods=["POST"])
def file():
    f=request.files["file"]
    model_based=True if request.form.get("model_based") is not None else False
    multi_column=True if request.form.get("multi_column") is not None else False
    
    df=load_data(f)
    data=df.copy()
    with open(f"Test-data/{f.filename}.data","wb") as file:
        pickle.dump(data,file)
    session["filename"]=f.filename
    return render_template("file.html",data=data,filename=f.filename,render=True,model_based=model_based,multi_column=multi_column,models=Model_list())

@app.route("/file/result",methods=["POST"])
def file_result():
    filename=session["filename"]
    with open(f"Test-data/{filename}.data","rb") as file:
        data=pickle.load(file)
    feature=request.form["feature"]
    model_type1=request.form["model1"]
    model_type2=request.form.get("model2")
    
    com_fea1=request.form.get("com_fea1")
    com_fea2=request.form.get("com_fea2")
    
    compare_image=None
    
    if com_fea1 is not None and com_fea2 is not None:
        print(com_fea1,com_fea2)
        compare_image="."+compare_columns(data,com_fea1,com_fea2)
        print(compare_image)
    
    if model_type2 is not None:
        print(session)
        print(model_type1,model_type2)
        df=label_encoder(df=data,out=feature)
        if type(df) is not pd.DataFrame:
            return render_template("error.html",error=[feature,df,filename])
        
        model1,model_path1=train_model(df,feature,model_type1)
        model2,model_path2=train_model(df,feature,model_type2)
        image1="."+save_image(df,feature,model1,1)
        image2="."+save_image(df,feature,model2,2)
        image_con1=False
        image_con2=False
        if "Classifier" in model1.__class__.__name__:
            image_con1="."+conf_mat(df,feature,model1)
        if "Classifier" in model2.__class__.__name__:
            image_con2="."+conf_mat(df,feature,model2)
        print(image1,image_con1)
        print(image2,image_con2)
        predict1=model1.predict(df.drop(columns=[feature]))
        predict2=model2.predict(df.drop(columns=[feature]))
        actual=df[feature]
        score1=model1.score(df.drop(columns=[feature]),df[feature]) 
        score2=model2.score(df.drop(columns=[feature]),df[feature])
        max_value=max(df[feature])
        min_value=min(df[feature])
        return render_template( "index1.html",filename=filename,
                                image1=image1,image2=image2
                                ,model_path="."+model_path1,
                                score1=score1,score2=score2,
                                predict2=predict2,predict1=predict1,feature=feature,
                                confusionMatrix1=image_con1,
                                confusionMatrix2=image_con2,
                                compare_image=compare_image
                                )
    print(session)
    df=label_encoder(df=data,out=feature)
    if type(df) is not pd.DataFrame:
        return render_template("error.html",error=[feature,df,filename])
        
    model1,model_path1=train_model(df,feature,model_type1)
    image1="."+save_image(df,feature,model1,1)
    image_con1=False
    image_con2=False
    if "Classifier" in model1.__class__.__name__:
        image_con1="."+conf_mat(df,feature,model1)
    print(image1,image_con1)
    predict1=model1.predict(df.drop(columns=[feature]))
    actual=df[feature]
    score1=model1.score(df.drop(columns=[feature]),df[feature]) 
    max_value=max(df[feature])
    min_value=min(df[feature])
    return render_template( "index1.html",filename=filename,
                            image1=image1,model_path="."+model_path1,
                            score1=score1,predict1=predict1,feature=feature,
                            confusionMatrix1=image_con1,
                            compare_image=compare_image
                            )


if __name__=="__main__":
    app.run(debug=True)
    # print(Model_list())
    