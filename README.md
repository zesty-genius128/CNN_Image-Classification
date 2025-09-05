# CNN Image Classification MERN Stack Project

Complete solution for the three-part machine learning project with CNN image classification, hyperparameter tuning, and MERN stack UI.

## Prerequisites

- Python 3.8+
- Node.js 16+
- MongoDB
- Git

## Project Structure

```
cnn-image-classification/
├── backend/
│   ├── server.js
│   ├── predict_images.py
│   ├── package.json
│   └── uploads/
├── frontend/
│   ├── src/
│   │   ├── App.js
│   │   ├── App.css
│   │   └── index.js
│   └── package.json
├── ml_models/
│   ├── problem1_cnn_classification.py
│   ├── problem2_hyperparameter_tuning.py
│   ├── best_cnn_model.h5
│   └── requirements.txt
└── README.md
```

## Installation Guide

### 1. Python Environment Setup

```bash
# Create virtual environment
python -m venv ml_env
source ml_env/bin/activate  # On Windows: ml_env\Scripts\activate

# Install Python dependencies
cd ml_models
pip install -r requirements.txt
```

### 2. Backend Setup

```bash
cd backend
npm install
```

### 3. Frontend Setup

```bash
cd frontend
npm install
```

### 4. MongoDB Setup

```bash
# Install MongoDB Community Edition
# Start MongoDB service
mongod --dbpath /your/db/path

# Or use MongoDB Atlas (cloud)
# Update connection string in backend/server.js
```

## Running the Application

### 1. Train the CNN Model (Problem 1)

```bash
cd ml_models
python problem1_cnn_classification.py
```

### 2. Run Hyperparameter Tuning (Problem 2)

```bash
python problem2_hyperparameter_tuning.py
```

### 3. Start Backend Server

```bash
cd backend
npm run dev
```

Server will run on http://localhost:5000

### 4. Start Frontend

```bash
cd frontend
npm start
```

Frontend will run on http://localhost:3000

## Features

### Problem 1: CNN Classification
- Robust CNN architecture with batch normalization and dropout
- Data augmentation for handling scale, rotation, illumination
- Comprehensive evaluation with accuracy, precision, recall, F1-score
- Confusion matrix visualization
- CIFAR-10 dataset integration

### Problem 2: Hyperparameter Tuning
- Grid search over 36 parameter combinations
- MongoDB integration for result storage
- Performance metrics tracking
- Best model selection and saving

### Problem 3: MERN Stack UI
- File upload functionality (up to 100 images)
- Image preview with grid layout
- Real-time classification using best model
- Confidence scores and probability distributions
- Responsive design with modern UI
- Progress tracking and error handling

## API Endpoints

- `POST /api/upload` - Upload multiple images
- `GET /api/images` - Get all uploaded images
- `POST /api/predict` - Predict image classes
- `GET /api/predictions` - Get prediction history
- `DELETE /api/clear` - Clear all data

## Usage Instructions

1. Upload Images: Click "Choose Images" and select up to 100 images
2. Upload to Server: Click "Upload Images" to send files to backend
3. Classify Images: Click "Classify X Images" to run predictions
4. View Results: See predicted classes with confidence scores
5. Clear Data: Use "Clear All" to reset everything

