# Problem 1: CNN Image Classification using CIFAR-10

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import ssl
import urllib.request

# Fix SSL certificate issue on macOS
ssl._create_default_https_context = ssl._create_unverified_context

# Load and preprocess CIFAR-10 dataset
def load_and_preprocess_data():
    """Load CIFAR-10 dataset and preprocess it"""
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Convert labels to categorical
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    return (x_train, y_train), (x_test, y_test)

# Define CNN architecture
def create_cnn_model(input_shape=(32, 32, 3), num_classes=10, activation='relu'):
    """Create a robust CNN model for CIFAR-10 classification"""
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation=activation, input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation=activation, padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation=activation, padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation=activation, padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation=activation, padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation=activation, padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fully Connected Layers
        layers.Flatten(),
        layers.Dense(512, activation=activation),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation=activation),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Training function
def train_model(model, train_data, validation_data, optimizer='adam', learning_rate=0.001, 
                batch_size=32, epochs=50):
    """Train the CNN model with specified parameters"""
    
    # Configure optimizer
    if optimizer == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    
    # Compile model
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks for better training
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7),
        keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
    
    # Data augmentation to handle challenges like rotation, translation, illumination
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )
    
    x_train, y_train = train_data
    x_val, y_val = validation_data
    
    # Train with data augmentation
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history

# Evaluation function
def evaluate_model(model, test_data):
    """Evaluate the model and generate comprehensive metrics"""
    x_test, y_test = test_data
    
    # Predictions
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes, 
                              target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Calculate metrics
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    return {
        'accuracy': test_accuracy,
        'loss': test_loss,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_true_classes, y_pred_classes, 
                                                     target_names=class_names, output_dict=True)
    }

# Main execution
def main():
    """Main function to execute the complete pipeline"""
    print("Loading and preprocessing CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    print("Creating CNN model...")
    model = create_cnn_model()
    
    print("Model Architecture:")
    model.summary()
    
    print("Training the model...")
    history = train_model(model, (x_train, y_train), (x_test, y_test))
    
    print("Evaluating the model...")
    metrics = evaluate_model(model, (x_test, y_test))
    
    print(f"Final Test Accuracy: {metrics['accuracy']:.4f}")
    
    # Save the model
    model.save('cifar10_cnn_model.h5')
    print("Model saved as 'cifar10_cnn_model.h5'")
    
    return model, history, metrics

if __name__ == "__main__":
    model, history, metrics = main()