import sys
from flask import Flask, Response, render_template, request, redirect, stream_with_context, url_for
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
        return render_template('results.html', model=modelTag, dataset=datasetTag, parameters=mlInstance.params)
    
    return render_template('homepage.html')

# @app.route("/output")
# def output():
#     def get_stdout():
#         for i in range(10):
#             yield str(i)
#     return Response(get_stdout(), mimetype="text/event-stream")

# http://127.0.0.1:5000/results?model=model_name&dataset=mydata&parameters=11
@app.route("/results", methods=['GET'])
def results():
    model = request.args.get('model')
    dataset = request.args.get('dataset')
    parameters = request.args.get('parameters')
    return render_template('results.html', model=model, dataset=dataset, parameters=parameters)



# flask --app app run --debug   