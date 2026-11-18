
from flask import Flask, render_template, request # type: ignore
import pandas as pd # type: ignore
import joblib # type: ignore

app = Flask(__name__)

# Load trained model
model = joblib.load("house_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    try:
        area = float(request.form["area"])
        bedrooms = int(request.form["bedrooms"])
        bathrooms = int(request.form["bathrooms"])
        age = int(request.form["age"])
        location = request.form["location"]

        if area <= 0 or bedrooms <= 0 or bathrooms <= 0 or age < 0:
            return render_template(
                "index.html",
                error="Please enter valid house details."
            )

        input_data = pd.DataFrame([{
            "Area": area,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Age": age,
            "Location": location
        }])

        prediction = model.predict(input_data)[0]

        return render_template(
            "index.html",
            prediction=round(prediction, 2)
        )

    except (ValueError, TypeError):
        return render_template(
            "index.html",
            error="Please enter valid numeric values."
        )

if __name__ == "__main__":
    app.run(debug=True)
