# Problem 2: Hyperparameter Tuning for CNN with MongoDB Integration

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pymongo
from datetime import datetime
import itertools
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import ssl

# Fix SSL certificate issue on macOS
ssl._create_default_https_context = ssl._create_unverified_context

# Import the model creation function from Problem 1
# (Assuming the previous code is in a module called cnn_model)

class HyperparameterTuner:
    def __init__(self, mongo_uri="mongodb://localhost:27017/", db_name="ml_experiments"):
        """Initialize hyperparameter tuner with MongoDB connection"""
        try:
            self.client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            # Test the connection
            self.client.server_info()
            self.db = self.client[db_name]
            self.collection = self.db["CNN_HyperParameter_Tuning"]
            self.mongo_available = True
            print("✅ Connected to MongoDB for experiment tracking")
        except (pymongo.errors.ServerSelectionTimeoutError, pymongo.errors.ConnectionFailure):
            print("⚠️  MongoDB not available - results will be stored in memory only")
            print("   Install & start MongoDB to persist hyperparameter results")
            self.mongo_available = False
            self.results_memory = []
        
        # Hyperparameter grid
        self.param_grid = {
            'learning_rate': [0.01, 0.001, 0.0001],
            'batch_size': [32, 64, 128],
            'optimizer': ['adam', 'sgd'],
            'activation_function': ['relu', 'tanh']
        }
    
    def create_cnn_model(self, activation='relu'):
        """Create CNN model with specified activation function"""
        model = keras.Sequential([
            # First Convolutional Block
            keras.layers.Conv2D(32, (3, 3), activation=activation, input_shape=(32, 32, 3), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(32, (3, 3), activation=activation, padding='same'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),
            
            # Second Convolutional Block
            keras.layers.Conv2D(64, (3, 3), activation=activation, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(64, (3, 3), activation=activation, padding='same'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),
            
            # Third Convolutional Block
            keras.layers.Conv2D(128, (3, 3), activation=activation, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(128, (3, 3), activation=activation, padding='same'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),
            
            # Fully Connected Layers
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation=activation),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(256, activation=activation),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation='softmax')
        ])
        
        return model
    
    def load_data(self):
        """Load and preprocess CIFAR-10 data"""
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        
        # Normalize
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Convert to categorical
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        return (x_train, y_train), (x_test, y_test)
    
    def train_and_evaluate(self, params, train_data, test_data):
        """Train model with given parameters and return metrics"""
        x_train, y_train = train_data
        x_test, y_test = test_data
        
        print(f"Training with params: {params}")
        
        # Create model
        model = self.create_cnn_model(activation=params['activation_function'])
        
        # Configure optimizer
        if params['optimizer'] == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=params['learning_rate'])
        else:
            optimizer = keras.optimizers.SGD(learning_rate=params['learning_rate'], momentum=0.9)
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7)
        ]
        
        # Train model (reduced epochs for faster tuning)
        history = model.fit(
            x_train, y_train,
            batch_size=params['batch_size'],
            epochs=20,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        
        # Get predictions for additional metrics
        y_pred = model.predict(x_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
        recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
        
        return {
            'accuracy': float(test_accuracy),
            'loss': float(test_loss),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'val_accuracy': float(max(history.history['val_accuracy'])),
            'val_loss': float(min(history.history['val_loss']))
        }
    
    def generate_param_combinations(self):
        """Generate all possible parameter combinations"""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combinations = list(itertools.product(*values))
        
        param_combinations = []
        for combination in combinations:
            param_dict = dict(zip(keys, combination))
            param_combinations.append(param_dict)
        
        return param_combinations
    
    def save_to_mongodb(self, params, metrics, experiment_id):
        """Save experiment results to MongoDB or memory"""
        document = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now(),
            'hyperparameters': params,
            'metrics': metrics,
            'status': 'completed'
        }
        
        if self.mongo_available:
            self.collection.insert_one(document)
            print(f"✅ Saved experiment {experiment_id} to MongoDB")
        else:
            self.results_memory.append(document)
            print(f"💾 Saved experiment {experiment_id} to memory")
    
    def run_hyperparameter_tuning(self):
        """Run complete hyperparameter tuning process"""
        print("Starting hyperparameter tuning...")
        
        # Load data
        train_data, test_data = self.load_data()
        
        # Generate parameter combinations
        param_combinations = self.generate_param_combinations()
        print(f"Total combinations to test: {len(param_combinations)}")
        
        best_accuracy = 0
        best_params = None
        results = []
        
        for i, params in enumerate(param_combinations, 1):
            try:
                print(f"\nExperiment {i}/{len(param_combinations)}")
                
                # Train and evaluate
                metrics = self.train_and_evaluate(params, train_data, test_data)
                
                # Save to MongoDB
                self.save_to_mongodb(params, metrics, i)
                
                # Track best performance
                if metrics['accuracy'] > best_accuracy:
                    best_accuracy = metrics['accuracy']
                    best_params = params.copy()
                
                results.append({
                    'experiment_id': i,
                    'params': params,
                    'metrics': metrics
                })
                
                print(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
                
            except Exception as e:
                print(f"Error in experiment {i}: {str(e)}")
                continue
        
        return results, best_params, best_accuracy
    
    def get_best_hyperparameters(self):
        """Retrieve best hyperparameters from MongoDB or memory"""
        if self.mongo_available:
            # Find document with highest accuracy
            best_result = self.collection.find().sort("metrics.accuracy", -1).limit(1)
            
            for result in best_result:
                return result['hyperparameters'], result['metrics']
            
            return None, None
        else:
            # Find best from memory
            if not self.results_memory:
                return None, None
            
            best_result = max(self.results_memory, key=lambda x: x['metrics']['accuracy'])
            return best_result['hyperparameters'], best_result['metrics']
    
    def display_results_summary(self):
        """Display summary of all experiments"""
        if self.mongo_available:
            results = list(self.collection.find().sort("metrics.accuracy", -1))
        else:
            # Sort memory results by accuracy
            results = sorted(self.results_memory, key=lambda x: x['metrics']['accuracy'], reverse=True)
        
        print("\n" + "="*80)
        print("HYPERPARAMETER TUNING RESULTS SUMMARY")
        print("="*80)
        
        for i, result in enumerate(results[:10], 1):  # Top 10 results
            params = result['hyperparameters']
            metrics = result['metrics']
            
            print(f"\nRank {i}:")
            print(f"  Experiment ID: {result['experiment_id']}")
            print(f"  Learning Rate: {params['learning_rate']}")
            print(f"  Batch Size: {params['batch_size']}")
            print(f"  Optimizer: {params['optimizer']}")
            print(f"  Activation: {params['activation_function']}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")

def main():
    """Main function to run hyperparameter tuning"""
    # Initialize tuner
    tuner = HyperparameterTuner()
    
    # Clear previous results (optional)
    # tuner.collection.delete_many({})
    
    # Run hyperparameter tuning
    results, best_params, best_accuracy = tuner.run_hyperparameter_tuning()
    
    print(f"\n{'='*60}")
    print("HYPERPARAMETER TUNING COMPLETED")
    print(f"{'='*60}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"Best Parameters: {best_params}")
    
    # Display results summary
    tuner.display_results_summary()
    
    # Get best hyperparameters from database
    best_params_db, best_metrics_db = tuner.get_best_hyperparameters()
    
    print(f"\nBest hyperparameters from database:")
    print(f"Parameters: {best_params_db}")
    print(f"Metrics: {best_metrics_db}")
    
    return tuner, results, best_params

if __name__ == "__main__":
    tuner, results, best_params = main()

# Example of how to use the best parameters for Problem 3
def get_best_model_for_ui():
    """Function to get the best trained model for UI integration"""
    tuner = HyperparameterTuner()
    best_params, best_metrics = tuner.get_best_hyperparameters()
    
    if best_params is None:
        print("No hyperparameter tuning results found. Please run tuning first.")
        return None
    
    # Load data
    (x_train, y_train), (x_test, y_test) = tuner.load_data()
    
    # Create and train model with best parameters
    model = tuner.create_cnn_model(activation=best_params['activation_function'])
    
    # Configure optimizer
    if best_params['optimizer'] == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
    else:
        optimizer = keras.optimizers.SGD(learning_rate=best_params['learning_rate'], momentum=0.9)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train final model
    model.fit(
        x_train, y_train,
        batch_size=best_params['batch_size'],
        epochs=50,
        validation_split=0.2,
        verbose=1
    )
    
    # Save the best model
    model.save('best_cnn_model.h5')
    print("Best model saved as 'best_cnn_model.h5'")
    
    return model, best_params