import sys
import pandas as pd
import datetime
from flask import Flask, Response, render_template, request, redirect, stream_with_context, send_file, url_for
from backend import *

app = Flask(__name__)

@app.route("/", methods = ['GET','POST'])
def homepage():
    if request.method == 'POST':
        modelTag = request.form['modelDropdown']
        #value = tag_lookup(tag)
        print(modelTag)
        datasetTag = request.form['datasetDropdown']
        #value = tag_lookup(tag)
        print(datasetTag)
        
        #Dependent on number of params REMEMBER TO IMPLEMENT LATER
        
        paramNum = 5
        paramLabels = [0] * (paramNum)
        paramTag = [0] * (paramNum)
        
        #for i in range(1, paramNum):
        #    paramLabels[i] = request.get['param' + str(i) + 'label']
        
        #print(paramLabels)
        
        for i in range(1, paramNum + 1):
            paramTag[i - 1] = request.values['param' + str(i) + 'value']
            
        
        
        #paramTag = request.values['param1value']
        print(paramTag)
        mlInstance = MLThing(modelTag, datasetTag, paramTag)
        

        output = pd.read_csv("emissions.csv")
        output['duration'] = str(datetime.timedelta(seconds=output['duration'].iloc[0]))
        return render_template('results.html', model=modelTag, dataset=datasetTag, parameters=paramTag, output=output)
    
    return render_template('homepage.html')

# http://127.0.0.1:5000/results?model=model_name&dataset=mydata&parameters=11
@app.route("/results", methods=['GET'])
def results():
    model = request.args.get('model')
    dataset = request.args.get('dataset')
    parameters = request.args.get('parameters')
    output = pd.read_csv("emissions.csv")
    output['duration'] = str(datetime.timedelta(seconds=output['duration'].iloc[0]))
    return render_template('results.html', model=model, dataset=dataset, parameters=parameters, output=output)

@app.route("/emissions.csv", methods=['GET'])
def emissions():
    return send_file('emissions.csv')

# flask --app app run --debug   