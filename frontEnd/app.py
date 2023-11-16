import sys
import io
import pandas as pd
import datetime
from flask import (
    Flask,
    Response,
    render_template,
    request,
    redirect,
    stream_with_context,
    send_file,
    url_for,
    jsonify
)
from backend import *

app = Flask(__name__)



mlInstance = MLThing()

@app.route("/", methods=["GET", "POST"])
def homepage():
    if request.method == "POST":
        modelTag = request.form["modelDropdown"]
        # value = tag_lookup(tag)
        print(modelTag)
        datasetTag = request.form["datasetDropdown"]
        # value = tag_lookup(tag)
        print(datasetTag)

        # subset name
        subsetTag = request.form["subsetTextbox"]
        if subsetTag == "None":
            print(subsetTag)

        # Dependent on number of params REMEMBER TO IMPLEMENT LATER

        paramNum = 5
        paramLabels = [0] * (paramNum)
        paramTag = [0] * (paramNum)

        for i in range(1, paramNum + 1):
            paramTag[i - 1] = request.values["param" + str(i) + "value"]
        print(paramTag)

        peftType = request.form["peftType"]
        print(peftType)

        #print(LabelArray)
        #print(ParamArray)

        mlInstance = None
        if peftType == "None":
            mlInstance.setVars(modelTag, datasetTag, paramTag) #= MLThing(modelTag, datasetTag, paramTag)
        else:
            mlInstance.setVars(modelTag, datasetTag, paramTag, peftType=peftType) #= MLThing(modelTag, datasetTag, paramTag, peftType=peftType)

        mlInstance.run()
        output = pd.read_csv("emissions.csv").iloc[-1]

        output["duration"] = str(datetime.timedelta(seconds=output["duration"]))
        return render_template(
            "results.html",
            model=modelTag,
            dataset=datasetTag,
            parameters=paramTag,
            peftType=peftType,
            output=output,
        )

    return render_template("homepage.html")


# http://127.0.0.1:5000/results?model=model_name&dataset=mydata&parameters=11
@app.route("/results", methods=["GET"])
def results():
    model = request.args.get("model")
    dataset = request.args.get("dataset")
    parameters = request.args.get("parameters")
    output = pd.read_csv("emissions.csv").iloc[-1]
    output["duration"] = str(datetime.timedelta(seconds=output["duration"]))
    return render_template(
        "results.html",
        model=model,
        dataset=dataset,
        parameters=parameters,
        peftType="None",
        output=output,
    )


# send the most recent result from emissions.csv (new runs are appended every time training is done, so emissions.csv contains entire history of runs)
@app.route("/emissions.csv", methods=["GET"])
def emissions():
    with open("emissions.csv", "rb") as f:
        lines = f.readlines()
        first_line = lines[0]
        last_line = lines[-1]
        file_obj = io.BytesIO(first_line + last_line)
        return send_file(file_obj, download_name="emissions.csv", mimetype="text/csv")
    # return send_file("emissions.csv")

# flask --app app run --debug   


@app.route('/fetchLabels', methods=['POST']) 
def process_labels(): 
    data = request.json['LabelArray'] 
    #result = sum(data) 
    mlInstance.setLbArray(data)
    
    #print(data)
    #print(result)
    return jsonify({'result': 5}) 

@app.route('/fetchParamValues', methods=['POST']) 
def process_params(): 
    data = request.json['ParamValues'] 
    #result = sum(data) 
    mlInstance.setPArray(data)
    #ParamArray = data
    #print(data)
    #print(result)
    return jsonify({'result': 5}) 

# flask --app app run --debug
