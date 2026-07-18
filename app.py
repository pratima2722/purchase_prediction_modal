import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the pickle model safely
MODEL_PATH = "naive_model_pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# HTML & CSS UI Layout with an attractive color scheme (Deep Indigo & Emerald)
HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Naive Bayes Prediction Dashboard</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --accent-color: #6366f1;
            --accent-hover: #4f46e5;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --success-color: #10b981;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            background-color: var(--card-bg);
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
            width: 100%;
            max-width: 480px;
            border: 1px solid #334155;
        }

        h2 {
            text-align: center;
            margin-bottom: 8px;
            color: var(--text-main);
            font-weight: 600;
        }

        .subtitle {
            text-align: center;
            color: var(--text-muted);
            font-size: 14px;
            margin-bottom: 30px;
        }

        .form-group {
            margin-bottom: 20px;
        }

        label {
            display: block;
            margin-bottom: 8px;
            color: var(--text-main);
            font-size: 14px;
            font-weight: 500;
        }

        input[type="number"], select {
            width: 100%;
            padding: 12px 16px;
            border-radius: 8px;
            border: 1px solid #475569;
            background-color: #0f172a;
            color: white;
            font-size: 15px;
            transition: all 0.3s ease;
        }

        input:focus, select:focus {
            outline: none;
            border-color: var(--accent-color);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }

        button {
            width: 100%;
            padding: 14px;
            background-color: var(--accent-color);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s ease;
            margin-top: 10px;
        }

        button:hover {
            background-color: var(--accent-hover);
        }

        .result-box {
            margin-top: 25px;
            padding: 15px;
            border-radius: 8px;
            background-color: rgba(16, 185, 129, 0.1);
            border: 1px solid var(--success-color);
            text-align: center;
        }

        .result-box h3 {
            color: var(--success-color);
            font-size: 18px;
        }
    </style>
</head>
<body>

<div class="container">
    <h2>Model Predictor</h2>
    <p class="subtitle">Gaussian Naive Bayes Deployment</p>

    <form method="POST" action="/">
        <div class="form-group">
            <label for="gender">Gender</label>
            <select id="gender" name="gender" required>
                <option value="1">Male</option>
                <option value="0">Female</option>
            </select>
        </div>

        <div class="form-group">
            <label for="age">Age</label>
            <input type="number" id="age" name="age" placeholder="e.g. 35" min="0" max="120" required>
        </div>

        <div class="form-group">
            <label for="salary">Estimated Salary</label>
            <input type="number" id="salary" name="salary" placeholder="e.g. 50000" min="0" required>
        </div>

        <button type="submit">Predict Result</button>
    </form>

    {% if prediction is not none %}
    <div class="result-box">
        <h3>Prediction Output: {{ prediction }}</h3>
    </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            # Extract inputs from form
            gender = float(request.form.get("gender"))
            age = float(request.form.get("age"))
            salary = float(request.form.get("salary"))
            
            # Structure features as a 2D array for the scikit-learn model
            features = np.array([[gender, age, salary]])
            
            # Generate prediction
            pred_output = model.predict(features)[0]
            prediction = int(pred_output)
            
        except Exception as e:
            prediction = f"Error processing input: {str(e)}"

    return render_template_string(HTML_LAYOUT, prediction=prediction)

if __name__ == "__main__":
    # Bind to PORT for Render compatibility
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
