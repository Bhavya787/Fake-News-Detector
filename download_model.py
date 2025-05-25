#!/usr/bin/env python3
"""
Download script for fake news detection model.
This script will download a pre-trained model from Hugging Face during the Render build process.
"""

import os
import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Define the model to use - using a valid, publicly available pre-trained model for fake news detection
MODEL_NAME = "Pulk17/Fake-News-Detection"
SAVE_PATH = "saved_models/fake_news_model"

def main():
    print("Downloading pre-trained fake news detection model...")
    
    # Create directories if they don't exist
    os.makedirs(os.path.join(SAVE_PATH, "model"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_PATH, "tokenizer"), exist_ok=True)
    
    try:
        # Download and save the model
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.save_pretrained(os.path.join(SAVE_PATH, "model"))
        print(f"Model saved to {os.path.join(SAVE_PATH, 'model')}")
        
        # Download and save the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.save_pretrained(os.path.join(SAVE_PATH, "tokenizer"))
        print(f"Tokenizer saved to {os.path.join(SAVE_PATH, 'tokenizer')}")
        
        print("Model and tokenizer download completed successfully!")
        return 0
    except Exception as e:
        print(f"Error downloading model: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
