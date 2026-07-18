import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# FIX: Changed 'naive_model_pkl' to 'naive_model.pkl' to match your repository file name
MODEL_PATH = "naive_model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# HTML layout with a sophisticated Slate & Emerald color palette
HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Purchase Prediction Dashboard</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --accent-color: #6366f1;
            --accent-hover: #4f46e5;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --success-color: #10b981;
            --border-color: #334155;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
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

        .card {
            background-color: var(--card-bg);
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
            width: 100%;
            max-width: 450px;
            border: 1px solid var(--border-color);
        }

        h2 {
            text-align: center;
            margin-bottom: 6px;
            color: var(--text-main);
            font-weight: 600;
            letter-spacing: -0.5px;
        }

        .subtitle {
            text-align: center;
            color: var(--text-muted);
            font-size: 14px;
            margin-bottom: 28px;
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
            color: #fff;
            font-size: 15px;
            transition: all 0.2s ease-in-out;
        }

        input:focus, select:focus {
            outline: none;
            border-color: var(--accent-color);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
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
            padding: 16px;
            border-radius: 8px;
            background-color: rgba(16, 185, 129, 0.1);
            border: 1px solid var(--success-color);
            text-align: center;
            animation: fadeIn 0.4s ease;
        }

        .result-box h3 {
            color: var(--success-color);
            font-size: 18px;
            font-weight: 600;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(5px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

<div class="card">
    <h2>Model Inference</h2>
    <p class="subtitle">Gaussian Naive Bayes Predictor</p>

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
            <input type="number" id="age" name="age" placeholder="Enter age" min="0" max="120" required>
        </div>

        <div class="form-group">
            <label for="salary">Estimated Salary</label>
            <input type="number" id="salary" name="salary" placeholder="Enter estimated salary" min="0" required>
        </div>

        <button type="submit">Predict Purchase Intent</button>
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
            # Extract numerical features
            gender = float(request.form.get("gender"))
            age = float(request.form.get("age"))
            salary = float(request.form.get("salary"))
            
            # Format inputs for scikit-learn
            features = np.array([[gender, age, salary]])
            
            # Predict
            pred_output = model.predict(features)[0]
            prediction = int(pred_output)
            
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template_string(HTML_LAYOUT, prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
