#!/usr/bin/env python3
"""
Fake News Detection - Transformer-based Model Implementation
Enhanced version for Google Colab with Drive integration and model download.
"""

# === Installation (Run this cell first in Colab) ===
# !pip install -q transformers datasets torch pandas scikit-learn

# === Imports ===
import os
import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import logging
from datetime import datetime
import shutil # For zipping

# === Google Colab Specific Imports ===
try:
    from google.colab import drive, files
    COLAB_ENV = True
except ImportError:
    COLAB_ENV = False
    print("Not running in Google Colab environment. Drive mounting and file download will be skipped.")

# === Setup Logging ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# === Configuration ===
CONFIG = {
    "MODEL_NAME": "distilbert-base-uncased", # Base model from Hugging Face
    "MAX_LENGTH": 256,                     # Max sequence length for tokenizer
    "BATCH_SIZE": 16,                      # Batch size (adjust based on Colab GPU memory)
    "EPOCHS": 3,                           # Number of training epochs
    "LEARNING_RATE": 2e-5,                 # Learning rate
    "WEIGHT_DECAY": 0.01,                  # Weight decay for regularization
    "WARMUP_STEPS": 500,                   # Number of warmup steps for learning rate scheduler
    "RANDOM_SEED": 42,                     # Seed for reproducibility
    "EARLY_STOPPING_PATIENCE": 3,          # Patience for early stopping
    "TEST_SIZE": 0.2,                      # Proportion of data for the test set
    "VAL_SIZE": 0.1,                       # Proportion of data for the validation set (from total)
}

# === Set Random Seeds ===
torch.manual_seed(CONFIG["RANDOM_SEED"])
np.random.seed(CONFIG["RANDOM_SEED"])

# === Google Drive Mounting (Colab Only) ===
if COLAB_ENV:
    logger.info("Mounting Google Drive...")
    drive.mount("/content/drive")
    # Define project root within Google Drive
    # IMPORTANT: Create this folder structure in your Google Drive beforehand!
    # MyDrive > fake_news_detector > data
    # MyDrive > fake_news_detector > saved_models
    # MyDrive > fake_news_detector > training_output

    PROJECT_ROOT = "/content/drive/MyDrive/fake news detector"


# === Directory Setup ===
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "saved_models", "fake_news_detector")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "training_output")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

# Create directories if they don't exist
for directory in [MODEL_SAVE_DIR, OUTPUT_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Ensured directory exists: {directory}")

# === Dataset Class ===
class NewsDataset(torch.utils.data.Dataset):
    """Custom Dataset class for news data"""
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# === Fake News Detector Class ===
class FakeNewsDetector:
    """Main class for fake news detection pipeline"""
    def __init__(self, model_name_or_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.model = None
        self.tokenizer = None
        self.trainer = None
        if model_name_or_path:
            self.load_model(model_name_or_path)

    def load_model(self, model_path):
        """Load a pre-trained model and tokenizer"""
        try:
            logger.info(f"Loading model from: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise

    def create_dummy_data(self, num_samples=100):
        """Creates a small dummy dataset for testing purposes if real data fails to load."""
        logger.warning("Creating dummy dataset for testing...")
        texts = [f"This is sample text number {i}." for i in range(num_samples)]
        labels = np.random.randint(0, 2, size=num_samples).tolist()
        df = pd.DataFrame({"text": texts, "label": labels})
        logger.info(f"Created dummy dataset with {len(df)} samples.")
        return df

    def load_data(self, true_csv_path, fake_csv_path):
        """Loads data from True.csv and Fake.csv files."""
        try:
            if not os.path.exists(true_csv_path) or not os.path.exists(fake_csv_path):
                logger.error(f"Dataset files not found at: {true_csv_path} and {fake_csv_path}")
                return self.create_dummy_data() # Fallback to dummy data

            logger.info(f"Loading data from {true_csv_path} and {fake_csv_path}...")
            true_df = pd.read_csv(true_csv_path, low_memory=False)
            fake_df = pd.read_csv(fake_csv_path, low_memory=False)

            true_df["label"] = 1
            fake_df["label"] = 0

            # Combine title and text, handle missing values
            def combine_text(df):
                df["title"] = df["title"].fillna("")
                df["text"] = df["text"].fillna("")
                df["full_text"] = df["title"] + " " + df["text"]
                return df[["full_text", "label"]].rename(columns={"full_text": "text"})

            true_df = combine_text(true_df)
            fake_df = combine_text(fake_df)

            df = pd.concat([true_df, fake_df], ignore_index=True)
            df = df.sample(frac=1, random_state=CONFIG["RANDOM_SEED"]).reset_index(drop=True)

            # Basic cleaning
            df = df.dropna(subset=["text"])
            df["text"] = df["text"].str.strip()
            df = df[df["text"].str.len() > 10] # Remove very short texts

            logger.info(f"Loaded and combined {len(df)} samples.")
            logger.info(f"Class distribution: REAL={sum(df['label']==1)}, FAKE={sum(df['label']==0)}")
            return df

        except Exception as e:
            logger.error(f"Error loading or processing data: {e}")
            return self.create_dummy_data() # Fallback

    def prepare_data(self, df, test_size, val_size):
        """Splits data into train, validation, and test sets."""
        texts = df["text"].tolist()
        labels = df["label"].tolist()

        if not labels or len(set(labels)) < 2:
             logger.error("Not enough data or labels for stratified split. Check data loading.")
             # Handle error appropriately, maybe raise exception or return None
             raise ValueError("Insufficient data for splitting.")

        # Split test set first
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=CONFIG["RANDOM_SEED"], stratify=labels
        )

        # Split train and validation sets from the remaining data
        val_size_adjusted = val_size / (1 - test_size)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts, train_val_labels, test_size=val_size_adjusted, random_state=CONFIG["RANDOM_SEED"], stratify=train_val_labels
        )

        logger.info(f"Data split: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")
        return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels

    def compute_metrics(self, eval_pred):
        """Computes evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

    def train(self, train_texts, train_labels, val_texts, val_labels):
        """Trains the sequence classification model."""
        logger.info("Initializing tokenizer and model for training...")
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG["MODEL_NAME"])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            CONFIG["MODEL_NAME"],
            num_labels=2,
            id2label={0: "FAKE", 1: "REAL"},
            label2id={"FAKE": 0, "REAL": 1}
        )
        self.model.to(self.device)

        logger.info("Creating datasets...")
        train_dataset = NewsDataset(train_texts, train_labels, self.tokenizer, CONFIG["MAX_LENGTH"])
        val_dataset = NewsDataset(val_texts, val_labels, self.tokenizer, CONFIG["MAX_LENGTH"])

        logger.info("Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=CONFIG["EPOCHS"],
            per_device_train_batch_size=CONFIG["BATCH_SIZE"],
            per_device_eval_batch_size=CONFIG["BATCH_SIZE"],
            learning_rate=CONFIG["LEARNING_RATE"],
            weight_decay=CONFIG["WEIGHT_DECAY"],
            warmup_steps=CONFIG["WARMUP_STEPS"],
            logging_dir=LOGS_DIR,
            logging_strategy="epoch",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to="none", # Disable wandb/tensorboard reporting unless configured
            save_total_limit=2, # Save only the best and the last checkpoint
            dataloader_pin_memory=False, # Set to False if issues arise
            gradient_accumulation_steps=1, # Can increase if facing memory limits
        )

        callbacks = [EarlyStoppingCallback(early_stopping_patience=CONFIG["EARLY_STOPPING_PATIENCE"])]

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks,
        )

        logger.info("Starting training...")
        train_result = self.trainer.train()
        logger.info("Training finished.")
        logger.info(f"Training Metrics: {train_result.metrics}")

        # Save the final best model explicitly after training
        self.save_model(MODEL_SAVE_DIR)

        return train_result

    def evaluate(self, test_texts, test_labels):
        """Evaluates the model on the test set."""
        if not self.trainer:
            logger.error("Trainer not initialized. Cannot evaluate.")
            return None, None

        logger.info("Evaluating model on the test set...")
        test_dataset = NewsDataset(test_texts, test_labels, self.tokenizer, CONFIG["MAX_LENGTH"])

        eval_results = self.trainer.evaluate(eval_dataset=test_dataset)
        logger.info(f"Test Evaluation Results: {eval_results}")

        # Generate classification report
        predictions = self.trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        report = classification_report(test_labels, y_pred, target_names=["FAKE", "REAL"])
        logger.info("\nDetailed Classification Report (Test Set):\n" + report)

        return eval_results, report

    def save_model(self, save_path):
        """Saves the trained model and tokenizer."""
        if not self.model or not self.tokenizer:
            logger.error("Model or tokenizer not available for saving.")
            return
        logger.info(f"Saving model and tokenizer to {save_path}...")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info("Model and tokenizer saved successfully.")

    def predict(self, text):
        """Predicts the label for a single piece of text."""
        if not self.model or not self.tokenizer:
            logger.error("Model not loaded. Train or load a model first.")
            return "Error", 0.0

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=CONFIG["MAX_LENGTH"],
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()

        label = self.model.config.id2label[predicted_class]
        return label, confidence

# === Main Execution ===
def main():
    """Main function to run the fake news detection pipeline."""
    print("=" * 50)
    print("=== Fake News Detection Fine-Tuning Pipeline (Colab) ===")
    print("=" * 50 + "\n")

    # --- 1. Initialize Detector ---
    # We initialize without a path to train a new model
    detector = FakeNewsDetector()

    # --- 2. Load Data ---

    true_csv = os.path.join(PROJECT_ROOT, "True.csv")
    fake_csv = os.path.join(PROJECT_ROOT, "Fake.csv")
    df = detector.load_data(true_csv, fake_csv)

    if df is None or len(df) < 50:
        logger.error("Failed to load sufficient data. Exiting.")
        return

    # --- 3. Prepare Data Splits ---
    try:
        train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = detector.prepare_data(
            df, test_size=CONFIG["TEST_SIZE"], val_size=CONFIG["VAL_SIZE"]
        )
    except ValueError as e:
        logger.error(f"Error during data splitting: {e}. Exiting.")
        return

    # --- 4. Train Model ---
    try:
        train_result = detector.train(train_texts, train_labels, val_texts, val_labels)
        if not detector.trainer or not detector.model:
             logger.error("Training did not complete successfully. Exiting.")
             return
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        return # Exit if training fails

    # --- 5. Evaluate Model ---
    logger.info("\n--- Evaluating Final Model on Test Set ---")
    eval_results, report = detector.evaluate(test_texts, test_labels)

    # --- 6. Test Predictions (Optional) ---
    print("\n" + "=" * 50)
    print("=== Testing Predictions on Examples ===")
    test_examples = [
        "Scientists discover new planet in nearby galaxy.",
        "BREAKING: Eating chocolate cures all diseases, says study funded by candy company."
    ]
    for text in test_examples:
        prediction, confidence = detector.predict(text)
        print(f"\nText:       {text[:100]}...")
        print(f"Prediction: {prediction} (Confidence: {confidence:.4f})")

    # --- 7. Zip and Download Model (Colab Only) ---
    if COLAB_ENV:
        print("\n" + "=" * 50)
        print("=== Zipping and Downloading Model ===")
        try:
            zip_filename = f"{os.path.basename(MODEL_SAVE_DIR)}.zip"
            zip_filepath = os.path.join(PROJECT_ROOT, "saved_models", zip_filename)
            logger.info(f"Zipping model directory {MODEL_SAVE_DIR} to {zip_filepath}...")
            shutil.make_archive(zip_filepath.replace(".zip", ""), 'zip', MODEL_SAVE_DIR)
            logger.info(f"Model zipped successfully to {zip_filepath}")

            print(f"\nInitiating download for {zip_filename}...")
            print("Check your browser for the download prompt.")
            files.download(zip_filepath)
            print("Download initiated.")

        except Exception as e:
            logger.error(f"Error zipping or downloading model: {e}")
            print(f"Could not automatically download. You can find the saved model in your Google Drive at: {MODEL_SAVE_DIR}")
            print(f"And the zipped file (if created) at: {zip_filepath}")

    else:
        print("\n" + "=" * 50)
        print("=== Model Saving Information (Non-Colab) ===")
        print(f"Model saved to: {MODEL_SAVE_DIR}")
        print("Zipping and automatic download are skipped outside Colab.")

    print("\n" + "=" * 50)
    print("=== Pipeline Complete ===")
    print("=" * 50)

if __name__ == "__main__":
    main()

