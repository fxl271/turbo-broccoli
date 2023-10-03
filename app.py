from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def homepage():
    return render_template('homepage.html')

# flask --app app run --debug

# model name, dataset, parameters