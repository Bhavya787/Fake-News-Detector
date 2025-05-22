#!/usr/bin/env python3
# model.py - Fake News Detection Model for Windows
# This file contains the FakeNewsDetector class and related functions

import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# --- Configuration ---
MODEL_NAME = "distilbert-base-uncased"  # Base model for fine-tuning
MAX_LENGTH = 512  # Max sequence length for the transformer model
BATCH_SIZE = 8  # Batch size for training
EPOCHS = 3  # Number of training epochs
LEARNING_RATE = 2e-5  # Learning rate for optimizer
RANDOM_SEED = 42  # Random seed for reproducibility

# Directory structure - Windows compatible paths
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to get the project root (assuming model.py is in src/model/)



PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
# Define paths relative to project root
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "saved_models", "fake_news_model")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "model_results")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# Create necessary directories
for directory in [DATA_DIR, MODEL_DIR, OUTPUT_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# --- Dataset Paths ---
TRUE_NEWS_PATH = os.path.join(DATA_DIR, "True.csv")
FAKE_NEWS_PATH = os.path.join(DATA_DIR, "Fake.csv")

# --- PyTorch Dataset Class ---
class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# --- Fake News Detector Class ---
class FakeNewsDetector:
    def __init__(self, model_name_or_path=None):
        """
        Initializes the detector. 
        Tries to load a local model from MODEL_DIR if it exists, 
        otherwise downloads/loads the specified model_name_or_path (or MODEL_NAME).
        """
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        load_path = model_name_or_path
        if os.path.exists(MODEL_DIR) and os.path.isdir(MODEL_DIR) and len(os.listdir(MODEL_DIR)) > 0 and model_name_or_path is None:
            print(f"Loading model from local path: {MODEL_DIR}")
            load_path = MODEL_DIR
        elif model_name_or_path is None:
            print(f"Local model not found at {MODEL_DIR}. Using default: {MODEL_NAME}")
            load_path = MODEL_NAME
        else:
            print(f"Loading specified model: {load_path}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(load_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
            self.model.to(self.device)
            self.model.eval()  # Set model to evaluation mode
            print(f"Model '{load_path}' loaded successfully on {self.device}.")
        except Exception as e:
            print(f"Error loading model '{load_path}': {e}")
            print("Please ensure the model name is correct or the path contains a valid Hugging Face model.")
            # As a fallback for demonstration if everything else fails, try the default again explicitly
            if load_path != MODEL_NAME:
                try:
                    print(f"Attempting to load default model as fallback: {MODEL_NAME}")
                    self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                    self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
                    self.model.to(self.device)
                    self.model.eval()
                    print(f"Default model '{MODEL_NAME}' loaded successfully on {self.device}.")
                except Exception as fallback_e:
                    print(f"Error loading default model '{MODEL_NAME}': {fallback_e}")
                    self.model = None  # Ensure model is None if loading fails
                    self.tokenizer = None

    def predict(self, text):
        """
        Predicts if the given text is fake news.
        Returns a tuple of (prediction_label, confidence)
        """
        if not self.model or not self.tokenizer:
            return "Error: Model not loaded.", 0.0

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class_id = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0, predicted_class_id].item()

        # For fake news detection model, the labels should be:
        if hasattr(self.model.config, 'id2label'):
            prediction_label = self.model.config.id2label[predicted_class_id]
        else:
            # Fallback labels if not defined in config
            labels = {0: 'FAKE', 1: 'REAL'}  # Adjust if your model uses different label mapping
            prediction_label = labels.get(predicted_class_id, "Unknown")
            
        return prediction_label, confidence

    def save_model_locally(self, save_directory=None):
        """
        Saves the currently loaded model and tokenizer to the specified local directory.
        """
        if save_directory is None:
            save_directory = MODEL_DIR
            
        if self.model and self.tokenizer:
            os.makedirs(save_directory, exist_ok=True)
            self.model.save_pretrained(save_directory)
            self.tokenizer.save_pretrained(save_directory)
            print(f"Model and tokenizer saved to {save_directory}")
        else:
            print("Error: No model loaded to save.")

# --- Functions for Data Preparation ---
def create_dummy_data():
    """Creates dummy data for testing if real data is not available."""
    print("Creating dummy data for testing...")
    
    # Create dummy True.csv with 40 samples
    if not os.path.exists(TRUE_NEWS_PATH) or os.path.getsize(TRUE_NEWS_PATH) < 100:
        print("Creating dummy True.csv dataset with 40 samples")
        dummy_true_data = {
            "title": ["True News Title " + str(i) for i in range(1, 41)],
            "text": ["This is a legitimate news article with accurate information " + str(i) for i in range(1, 41)],
            "subject": ["politicsNews", "worldnews", "scienceNews", "business", "technology"] * 8,
            "date": ["January " + str(i % 30 + 1) + ", 2018" for i in range(1, 41)]
        }
        pd.DataFrame(dummy_true_data).to_csv(TRUE_NEWS_PATH, index=False)
    
    # Create dummy Fake.csv with 40 samples
    if not os.path.exists(FAKE_NEWS_PATH) or os.path.getsize(FAKE_NEWS_PATH) < 100:
        print("Creating dummy Fake.csv dataset with 40 samples")
        dummy_fake_data = {
            "title": ["Fake News Title " + str(i) for i in range(1, 41)],
            "text": ["This is a fabricated news story with false information " + str(i) for i in range(1, 41)],
            "subject": ["politics", "conspiracy", "entertainment", "sports", "health"] * 8,
            "date": ["February " + str(i % 28 + 1) + ", 2017" for i in range(1, 41)]
        }
        pd.DataFrame(dummy_fake_data).to_csv(FAKE_NEWS_PATH, index=False)

def load_and_prepare_data():
    """Loads and prepares the dataset for training."""
    try:
        df_true = pd.read_csv(TRUE_NEWS_PATH)
        df_fake = pd.read_csv(FAKE_NEWS_PATH)
        print(f"Successfully loaded True.csv ({df_true.shape[0]} rows) and Fake.csv ({df_fake.shape[0]} rows)")
    except Exception as e:
        print(f"ERROR: Could not load dataset files: {e}")
        print("Creating dummy data for demonstration...")
        create_dummy_data()
        df_true = pd.read_csv(TRUE_NEWS_PATH)
        df_fake = pd.read_csv(FAKE_NEWS_PATH)
        print(f"Successfully loaded dummy data: True.csv ({df_true.shape[0]} rows) and Fake.csv ({df_fake.shape[0]} rows)")

    # Add labels before merging (1 for True news, 0 for Fake news)
    df_true["label"] = 1
    df_fake["label"] = 0

    # Concatenate the dataframes
    df = pd.concat([df_true, df_fake], ignore_index=True)

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Create full_text column combining title and text
    df["full_text"] = df["title"] + " " + df["text"]
    
    return df

def split_data(df):
    """Splits the data into training, validation, and test sets."""
    texts = df["full_text"].tolist()
    labels = df["label"].tolist()
    
    total_samples = len(texts)
    
    # Determine appropriate split sizes based on dataset size
    if total_samples < 20:
        TEST_SIZE = 0.2  # 20% for test
        VALIDATION_SIZE = 0.25  # 25% of remaining for validation
    elif total_samples < 100:
        TEST_SIZE = 0.15  # 15% for test
        VALIDATION_SIZE = 0.15  # 15% of remaining for validation
    else:
        TEST_SIZE = 0.15  # Standard split
        VALIDATION_SIZE = 0.15  # Standard split
    
    print(f"Using TEST_SIZE={TEST_SIZE:.2f}, VALIDATION_SIZE={VALIDATION_SIZE:.2f}")
    
    try:
        # First, split into training+validation and test
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=labels
        )
        
        # Then, split training+validation into training and validation
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels, test_size=VALIDATION_SIZE, 
            random_state=RANDOM_SEED, stratify=train_labels
        )
        
        print("Successfully split dataset using stratified sampling")
    except ValueError as e:
        print(f"Warning: Stratified split failed with error: {e}")
        print("Falling back to simple random splitting")
        
        # Simple fixed-size splitting for very small datasets
        n_samples = len(texts)
        n_test = max(2, int(n_samples * 0.2))  # At least 2 samples for test
        n_val = max(2, int((n_samples - n_test) * 0.25))  # At least 2 samples for validation
        n_train = n_samples - n_test - n_val  # Remainder for training
        
        if n_train < 2:
            raise ValueError("Dataset too small for meaningful splitting. Need at least 6 samples.")
        
        # Shuffle indices
        indices = list(range(n_samples))
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(indices)
        
        # Split indices
        test_indices = indices[:n_test]
        val_indices = indices[n_test:n_test+n_val]
        train_indices = indices[n_test+n_val:]
        
        # Create splits
        test_texts = [texts[i] for i in test_indices]
        test_labels = [labels[i] for i in test_indices]
        val_texts = [texts[i] for i in val_indices]
        val_labels = [labels[i] for i in val_indices]
        train_texts = [texts[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
    
    print(f"Dataset split sizes:")
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    print(f"Test samples: {len(test_texts)}")
    
    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels

def prepare_datasets(train_texts, val_texts, test_texts, train_labels, val_labels, test_labels):
    """Prepares PyTorch datasets for training, validation, and testing."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Tokenizer loaded successfully.")
    
    print("Tokenizing texts...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    print("Texts tokenized.")
    
    train_dataset = NewsDataset(train_encodings, train_labels)
    val_dataset = NewsDataset(val_encodings, val_labels)
    test_dataset = NewsDataset(test_encodings, test_labels)
    
    print("PyTorch Datasets created.")
    
    return tokenizer, train_dataset, val_dataset, test_dataset

# --- Functions for Model Training ---
def compute_metrics(pred):
    """Computes evaluation metrics for the model."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

def train_model(train_dataset, val_dataset):
    """Trains the model on the provided datasets."""
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    print(f"Model '{MODEL_NAME}' loaded successfully.")
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        save_total_limit=1,  # Only keep the best model checkpoint
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train the model
    print("Training the model...")
    train_result = trainer.train()
    print("Model training completed.")
    
    # Save training metrics
    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    
    # Evaluate on validation set
    print("Evaluating the model on validation set...")
    val_metrics = trainer.evaluate()
    trainer.log_metrics("eval", val_metrics)
    trainer.save_metrics("eval", val_metrics)
    
    return trainer, model, train_metrics, val_metrics

def evaluate_model(trainer, test_dataset, test_labels):
    """Evaluates the model on the test dataset."""
    print("Evaluating the model on test set...")
    test_results = trainer.predict(test_dataset)
    test_metrics = test_results.metrics
    
    # Save test metrics
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)
    
    # Get predictions
    test_preds = test_results.predictions.argmax(-1)
    
    # Calculate detailed metrics
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average="binary"
    )
    test_accuracy = accuracy_score(test_labels, test_preds)
    
    # Generate confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    
    # Generate classification report
    class_report = classification_report(test_labels, test_preds, target_names=["Fake", "Real"])
    print("\nClassification Report:")
    print(class_report)
    
    return test_metrics, test_accuracy, test_precision, test_recall, test_f1, cm, class_report

def save_trained_model(trainer, tokenizer):
    """Saves the trained model and tokenizer."""
    print(f"Saving the model to {MODEL_DIR}...")
    trainer.save_model(MODEL_DIR)
    print(f"Saving the tokenizer to {MODEL_DIR}...")
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Model and tokenizer saved successfully to {MODEL_DIR}")

# --- Main Function for Training ---
def train_and_save_model():
    """Main function to train and save the model."""
    print("\n--- Starting Fake News Detection Model Training ---\n")
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Split data
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = split_data(df)
    
    # Prepare datasets
    tokenizer, train_dataset, val_dataset, test_dataset = prepare_datasets(
        train_texts, val_texts, test_texts, train_labels, val_labels, test_labels
    )
    
    # Train model
    trainer, model, train_metrics, val_metrics = train_model(train_dataset, val_dataset)
    
    # Evaluate model
    test_metrics, test_accuracy, test_precision, test_recall, test_f1, cm, class_report = evaluate_model(
        trainer, test_dataset, test_labels
    )
    
    # Save model
    save_trained_model(trainer, tokenizer)
    
    print("\n--- Model Training and Evaluation Completed ---")
    print(f"Model saved to {MODEL_DIR}")
    
    # Return metrics for reporting
    return {
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1
    }

def test_model_predictions():
    """Tests the model with example texts."""
    print("\n--- Testing Model Predictions ---\n")
    
    # Initialize detector
    detector = FakeNewsDetector()
    
    if detector.model:
        # Test examples
        test_examples = [
            "Scientists discover new method to generate clean energy. The findings were published in a peer-reviewed journal today.",
            "BREAKING: Aliens confirmed to be living among us, says anonymous source with blurry photo!",
            "New study shows regular exercise improves heart health. Researchers from multiple universities collaborated on the long-term study.",
            "SHOCKING: Government secretly putting mind-control chemicals in the water supply, according to unnamed insider!"
        ]
        
        for i, text in enumerate(test_examples):
            prediction, confidence = detector.predict(text)
            print(f"\nExample {i+1}:")
            print(f"Text: {text[:100]}...")
            print(f"Prediction: {prediction}")
            print(f"Confidence: {confidence:.4f}")
    else:
        print("Model could not be loaded. Please train the model first or check the model path.")

# --- Main Execution ---
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fake News Detection Model")
    parser.add_argument("--train", action="store_true", help="Train and save the model")
    parser.add_argument("--test", action="store_true", help="Test the model with example texts")
    parser.add_argument("--predict", type=str, help="Predict a single text input")
    
    args = parser.parse_args()
    
    if args.train:
        metrics = train_and_save_model()
        print("\nTraining Results Summary:")
        print(f"Training Accuracy: {metrics['train_metrics'].get('train_accuracy', 'N/A')}")
        print(f"Validation Accuracy: {metrics['val_metrics'].get('eval_accuracy', 'N/A')}")
        print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"Test F1 Score: {metrics['test_f1']:.4f}")
    
    if args.test:
        test_model_predictions()
    
    if args.predict:
        detector = FakeNewsDetector()
        if detector.model:
            prediction, confidence = detector.predict(args.predict)
            print(f"\nPrediction for input text:")
            print(f"Text: {args.predict[:100]}...")
            print(f"Prediction: {prediction}")
            print(f"Confidence: {confidence:.4f}")
        else:
            print("Model could not be loaded. Please train the model first or check the model path.")
    
    # If no arguments provided, show help
    if not (args.train or args.test or args.predict):
        parser.print_help()
