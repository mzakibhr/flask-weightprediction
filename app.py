from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def weight_prediction():
    if request.method == 'GET':
        return render_template("weight-height-prediction.html")
    elif request.method == 'POST':
        print(dict(request.form))
        weight_features = dict(request.form).values()
        weight_features = np.array([float(x) for x in weight_features])
        model = joblib.load(
            "model-development/weight-height-classification-using-linear-regression.pkl")
        weight_features = [weight_features]
        print(weight_features)
        result = model.predict(weight_features)
        result = np.round(result, 2)
        result = float(result[0])
        return render_template('weight-height-prediction.html', result=result)
    else:
        return "Unsupported Request Method"


if __name__ == '__main__':
    app.run(port=5000, debug=True)
