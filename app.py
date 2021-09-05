import numpy as np
import pickle
import requests
import json
import os
import pandas as pd 
from flask import Flask, render_template, redirect, url_for, request,jsonify
from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple


current_dir = os.getcwd()
_model_loc =  os.path.join(current_dir,'models')



def _load_model(path=os.path.join(_model_loc,"word2vec")):
    model = gensim.models.Word2Vec.load(os.path.join(_model_loc,"word2vec"))
    return model

def _load_matrix(path=os.path.join(_model_loc,"df_top10_word2vec.csv")):
    df = pd.read_csv(path)    
    return df 

app=Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.config["DEBUG"] = False




# instantiate index page
@app.route("/")
def index():
   	return render_template("index.html")
    
# return model predictions
@app.route("/get_predictions", methods=["GET","POST"])   
def predict(): 
    
    print("Start Predict")
    msg_data={}
    for k in request.args.keys():
        val=request.args.get(k)
        msg_data[k]=val
    print(msg_data["productid"])

    item_id = msg_data["productid"]
    df = _load_matrix() 
    inp = df.loc[df['index']== item_id]
    print(inp)
    print({ str(k) :  str(inp[k].values[0]) for k in df.columns })
    return{ str(k) :  str(inp[k].values[0]) for k in df.columns }
    
 
if __name__ == "__main__":
    run_simple("localhost", 5000, app)   
    
