from flask import Flask, render_template, request
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
        
        mlInstance = MLThing(modelTag, datasetTag, 5)
        print(mlInstance.params)
    
    return render_template('homepage.html')



# http://127.0.0.1:5000/results?model=model_name&dataset=mydata&parameters=11
@app.route("/results", methods=['GET'])
def results():
    model = request.args.get('model')
    dataset = request.args.get('dataset')
    parameters = request.args.get('parameters')
    return render_template('results.html', model=model, dataset=dataset, parameters=parameters)



# flask --app app run --debug   