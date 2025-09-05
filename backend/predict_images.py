# predict_images.py - Python script for image classification
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import os

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def load_best_model():
    """Load the best trained model"""
    try:
        # Try to load the newly trained model first
        model_path = os.path.join(os.path.dirname(__file__), '..', 'ml_models', 'cifar10_cnn_model.h5')
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)
            return model
        
        # Fallback to best_cnn_model.h5
        model_path = os.path.join(os.path.dirname(__file__), '..', 'ml_models', 'best_cnn_model.h5')
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)
            return model
        else:
            # If no saved model, create a default model (this should not happen in production)
            print("Warning: No saved model found. Using default model.")
            return create_default_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        return create_default_model()

def create_default_model():
    """Create a default CNN model if saved model is not available"""
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    try:
        # Load and resize image
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to 32x32 (CIFAR-10 size)
        image = image.resize((32, 32))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Normalize pixel values to [0, 1]
        image_array = image_array.astype('float32') / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def predict_single_image(model, image_path):
    """Predict class for a single image"""
    try:
        # Preprocess image
        processed_image = preprocess_image(image_path)
        
        if processed_image is None:
            return {
                'error': f'Failed to preprocess image: {image_path}',
                'predictedClass': 'unknown',
                'confidence': 0.0
            }
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = CIFAR10_CLASSES[predicted_class_idx]
        
        return {
            'predictedClass': predicted_class,
            'confidence': confidence,
            'allProbabilities': {
                CIFAR10_CLASSES[i]: float(predictions[0][i]) 
                for i in range(len(CIFAR10_CLASSES))
            }
        }
    except Exception as e:
        return {
            'error': f'Prediction failed for {image_path}: {str(e)}',
            'predictedClass': 'unknown',
            'confidence': 0.0
        }

def main():
    """Main function to process image predictions"""
    try:
        # Get image list from command line argument
        if len(sys.argv) < 2:
            print(json.dumps({'error': 'No image list provided'}))
            return
        
        images_json = sys.argv[1]
        images = json.loads(images_json)
        
        # Load the model
        model = load_best_model()
        
        predictions = []
        
        for image_info in images:
            # Construct the correct path from the backend directory
            backend_dir = os.path.dirname(os.path.abspath(__file__))
            image_path = os.path.join(backend_dir, 'uploads', image_info['filename'])
            
            if not os.path.exists(image_path):
                predictions.append({
                    'filename': image_info['filename'],
                    'originalName': image_info.get('originalName', image_info['filename']),
                    'predictedClass': 'unknown',
                    'confidence': 0.0,
                    'error': f'Image file not found: {image_path}'
                })
                continue
            
            # Predict image class
            prediction_result = predict_single_image(model, image_path)
            
            # Add file information
            prediction_result.update({
                'filename': image_info['filename'],
                'originalName': image_info.get('originalName', image_info['filename'])
            })
            
            predictions.append(prediction_result)
        
        # Output results as JSON
        print(json.dumps(predictions))
        
    except Exception as e:
        error_result = {
            'error': f'Main prediction error: {str(e)}',
            'predictions': []
        }
        print(json.dumps(error_result))

if __name__ == '__main__':
    main()