from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def homepage():
    return render_template('homepage.html')

# http://127.0.0.1:5000/results?model=model_name&dataset=mydata&parameters=11
@app.route("/results", methods=['GET'])
def results():
    model = request.args.get('model')
    dataset = request.args.get('dataset')
    parameters = request.args.get('parameters')
    return render_template('results.html', model=model, dataset=dataset, parameters=parameters)

# flask --app app run --debug