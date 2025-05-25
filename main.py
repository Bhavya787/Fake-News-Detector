#!/usr/bin/env python3
"""
Modified main.py for Render deployment with memory optimizations
- Updated paths to work with flattened directory structure
- Implemented lazy loading for model to reduce memory usage
- Added model quantization for further memory reduction
"""

import sys
import os
import gc
import threading

# Import Flask and related libraries
from flask import Flask, render_template, request

# Import the FakeNewsDetector class from model.py
from model.model import FakeNewsDetector

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")

# Global variables for lazy loading
detector = None
detector_lock = threading.Lock()

def get_detector():
    """
    Lazy loading function for the detector.
    Only loads the model when needed and ensures thread safety.
    """
    global detector
    
    # If detector is already loaded, return it
    if detector is not None and detector.model is not None:
        return detector
        
    # Thread-safe loading of the model
    with detector_lock:
        # Double-check in case another thread loaded it while we were waiting
        if detector is not None and detector.model is not None:
            return detector
            
        print("Lazy loading Fake News Detector model...")
        detector = FakeNewsDetector()
        
        # Apply quantization to reduce memory usage
        if detector.model is not None:
            try:
                import torch
                # Convert model to float16 for memory reduction
                detector.model = detector.model.half()  # Convert to float16
                print("Model quantized to float16 to reduce memory usage")
            except Exception as e:
                print(f"Warning: Could not quantize model: {e}")
                
        if not detector.model:
            print("CRITICAL: Model could not be loaded. Predictions will fail.")
            
        return detector

@app.route("/", methods=["GET", "POST"])
def home():
    """
    Main route for the web application.
    GET: Displays the form for text input
    POST: Processes the submitted text and returns prediction
    """
    result = None
    submitted_text = None
    
    if request.method == "POST":
        text_input = request.form.get("text_input", "").strip()
        submitted_text = text_input  # Store for display
        
        if not text_input:
            result = {"label": "Error", "confidence": 0.0, "error": "Please enter some text to analyze."}
        else:
            try:
                # Lazy load the detector only when needed
                current_detector = get_detector()
                
                if not current_detector.model:
                    result = {"label": "Error", "confidence": 0.0, "error": "Model is not available. Please check server logs."}
                else:
                    prediction_label, confidence = current_detector.predict(text_input)
                    result = {"label": prediction_label, "confidence": confidence}
                    
                    # Force garbage collection after prediction to free memory
                    gc.collect()
                    
            except Exception as e:
                print(f"Error during prediction: {e}")
                result = {"label": "Error", "confidence": 0.0, "error": f"An error occurred during analysis: {str(e)}"}
    
    return render_template("index.html", result=result, submitted_text=submitted_text)

# Create a simple HTML template if it doesn't exist
def create_template_if_missing():
    """Creates a basic HTML template for the web application if it doesn't exist."""
    templates_dir = os.path.join("templates")
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
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <h1>Fake News Detector</h1>
    
    <div class="container">
        <form method="POST">
            <p>Enter a news article or text to analyze:</p>
            <textarea name="text_input" placeholder="Paste news article text here...">{{ submitted_text }}</textarea>
            <button type="submit">Analyze</button>
        </form>
    </div>
    
    {% if result %}
    <div class="result {% if result.label == 'REAL' %}real{% elif result.label == 'FAKE' %}fake{% else %}error{% endif %}">
        {% if result.error %}
            <h3>Error</h3>
            <p>{{ result.error }}</p>
        {% else %}
            <h3>Prediction: {{ result.label }}</h3>
            <p>Confidence: {{ "%.2f"|format(result.confidence*100) }}%</p>
        {% endif %}
    </div>
    {% endif %}
</body>
</html>""")
        print(f"Created template file: {template_path}")

if __name__ == "__main__":
    # Create template if it doesn't exist
    create_template_if_missing()
    
    # Start the Flask development server
    print("Starting Flask development server...")
    print("Open your browser and navigate to http://localhost:5000")
    
    # For Windows, use host='127.0.0.1' instead of '0.0.0.0' to avoid firewall prompts
    app.run(host="0.0.0.0", port=5000, debug=True)
