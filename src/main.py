import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

# Import
from model.model import FakeNewsDetector
from flask import Flask, render_template, request

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")

# Initialize the detector
print("Initializing Fake News Detector for Flask app...")
detector = FakeNewsDetector()
if not detector.model:
    print("CRITICAL: Model could not be loaded for the Flask application. Predictions will fail.")

@app.route("/", methods=["GET", "POST"])
def home():
    
    result = None
    submitted_text = None
    
    if request.method == "POST":
        text_input = request.form.get("text_input", "").strip()
        submitted_text = text_input  
        
        if not text_input:
            result = {"label": "Error", "confidence": 0.0, "error": "Please enter some text to analyze."}
        elif not detector.model:
            result = {"label": "Error", "confidence": 0.0, "error": "Model is not available. Please check server logs."}
        else:
            try:
                prediction_label, confidence = detector.predict(text_input)
                result = {"label": prediction_label, "confidence": confidence}
            except Exception as e:
                print(f"Error during prediction: {e}")
                result = {"label": "Error", "confidence": 0.0, "error": f"An error occurred during analysis: {str(e)}"}
                
    return render_template("index.html", result=result, submitted_text=submitted_text)

# Create a simple HTML template if it doesn't exist
def create_template_if_missing():
   
    templates_dir = os.path.join(SCRIPT_DIR, "templates")
    os.makedirs(templates_dir, exist_ok=True)
    
    template_path = os.path.join(templates_dir, "index.html")
    if not os.path.exists(template_path):
        with open(template_path, "w") as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Fake News Detector</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .container {
            background-color: #f9f9f9;
            border-radius: 5px;
            padding: 20px;
            margin-top: 20px;
        }
        textarea {
            width: 100%;
            height: 150px;
            padding: 10px;
            margin-bottom: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            resize: vertical;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 4px;
        }
        .real {
            background-color: #dff0d8;
            border: 1px solid #d6e9c6;
            color: #3c763d;
        }
        .fake {
            background-color: #f2dede;
            border: 1px solid #ebccd1;
            color: #a94442;
        }
        .error {
            background-color: #fcf8e3;
            border: 1px solid #faebcc;
            color: #8a6d3b;
        }
    </style>
</head>
<body>
    <h1>Fake News Detector</h1>
    
    <div class="container">
        <form method="POST">
            <p>Enter a news article or text to analyze:</p>
            <textarea name="text_input" placeholder="Paste news article text here...">{{ submitted_text }}</textarea>
            <button type="submit">Analyze</button>
        </form>
        
        {% if result %}
        <div class="result {% if result.label == 'REAL' %}real{% elif result.label == 'FAKE' %}fake{% else %}error{% endif %}">
            {% if result.error %}
                <h3>Error</h3>
                <p>{{ result.error }}</p>
            {% else %}
                
                
                <h3>Prediction: 
                             {% if result.label == 'REAL' %} REAL
            {% elif result.label == 'FAKE' %} FAKE
            {% elif result.label == 'UNCERTAIN' %} UNCERTAIN (Confidence below threshold)
            {% elif result.label == 'Error' %} Error
            {% else %} Unknown
            {% endif %}
                </h3>

                
                <p>Confidence: {{ "%.2f"|format(result.confidence*100) }}%</p>
            {% endif %}
        </div>
        {% endif %}
    </div>
</body>
</html>""")
        print(f"Created template file: {template_path}")


if __name__ == "__main__":
    create_template_if_missing()
    
    print("Starting Flask development server...")
    print("Open your browser and navigate to http://localhost:5000")
    
    app.run(host="127.0.0.1", port=5000, debug=True)
