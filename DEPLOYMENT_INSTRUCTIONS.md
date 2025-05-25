# Deployment Instructions for Fake News Detector

This document provides updated step-by-step instructions for deploying the Fake News Detector project on Render's free tier.

## Overview

The Fake News Detector is a Flask web application that uses a machine learning model to detect fake news. The original model file (model.safetensors) is quite large (255MB) and couldn't be included in the GitHub repository. Our deployment solution automatically downloads a pre-trained model from Hugging Face during the build process.

## Deployment Files Structure

The deployment package has been restructured to work properly with Render:

```
/
├── main.py                  # Main Flask application
├── requirements.txt         # Python dependencies
├── download_model.py        # Script to download the model
├── render.yaml              # Render configuration
├── model/                   # Model code
├── static/                  # CSS and static assets
├── templates/               # HTML templates
└── saved_models/           # Will be created during build
```

## Deployment Steps

### 1. Create a Render Account

If you don't already have one, sign up for a free account at [render.com](https://render.com).

### 2. Create a New Web Service

1. Log in to your Render dashboard
2. Click on "New" and select "Web Service"
3. Connect your GitHub repository or use the deployment files we've prepared

### 3. Configure the Web Service

Use the following settings:
- **Name**: fake-news-detector (or any name you prefer)
- **Environment**: Python
- **Region**: Choose the region closest to your users
- **Branch**: main (or your preferred branch)
- **Build Command**: `pip install -r requirements.txt && python download_model.py`
- **Start Command**: `gunicorn main:app`
- **Plan**: Free

### 4. Deploy

Click "Create Web Service" to start the deployment process. The initial build may take several minutes as it needs to:
1. Install all dependencies
2. Download the pre-trained model from Hugging Face

### 5. Access Your Application

Once deployment is complete, Render will provide a URL to access your application (typically in the format `https://fake-news-detector-xxxx.onrender.com`).

## Important Notes

1. **Free Tier Limitations**: Render's free tier includes:
   - 512MB RAM
   - Shared CPU
   - The service will spin down after 15 minutes of inactivity
   - Limited to 750 hours per month

2. **Model Download**: The deployment automatically downloads a pre-trained fake news detection model from Hugging Face. If you want to use your specific model instead, you would need to:
   - Upload your model to a storage service (Google Drive, Dropbox, etc.)
   - Modify the download_model.py script to fetch your model instead

3. **Cold Starts**: Since the free tier spins down after inactivity, the first request after inactivity may take longer to process as the service needs to start up again.

## Troubleshooting

If you encounter issues during deployment:

1. **Build Failures**: 
   - Check that all files are in the correct locations
   - Ensure requirements.txt is at the root of your repository
   - Verify that the build command can find and execute download_model.py

2. **Model Download Issues**: 
   - Ensure the Hugging Face model is accessible
   - Check for any network or permission issues

3. **Memory Limitations**: 
   - If the application crashes, it might be due to the 512MB RAM limit of the free tier
   - Consider using a smaller model or optimizing the application

For persistent issues, consider upgrading to a paid tier or exploring alternative deployment options.
