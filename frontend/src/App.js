// React Frontend - App.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE_URL = 'http://localhost:8000/api';

function App() {
    const [images, setImages] = useState([]);
    const [selectedFiles, setSelectedFiles] = useState([]);
    const [predictions, setPredictions] = useState({});
    const [loading, setLoading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [predictionProgress, setPredictionProgress] = useState(0);

    // CIFAR-10 class names for reference
    const CIFAR10_CLASSES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ];

    // Load images on component mount
    useEffect(() => {
        loadImages();
    }, []);

    const loadImages = async () => {
        try {
            const response = await axios.get(`${API_BASE_URL}/images`);
            setImages(response.data.images);
        } catch (error) {
            console.error('Error loading images:', error);
            alert('Failed to load images');
        }
    };

    const handleFileSelection = (event) => {
        const files = Array.from(event.target.files);
        setSelectedFiles(files);
    };

    const uploadImages = async () => {
        if (selectedFiles.length === 0) {
            alert('Please select images to upload');
            return;
        }

        if (selectedFiles.length > 100) {
            alert('Maximum 100 images allowed');
            return;
        }

        setLoading(true);
        setUploadProgress(0);

        const formData = new FormData();
        selectedFiles.forEach((file) => {
            formData.append('images', file);
        });

        try {
            const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
                onUploadProgress: (progressEvent) => {
                    const progress = Math.round(
                        (progressEvent.loaded * 100) / progressEvent.total
                    );
                    setUploadProgress(progress);
                },
            });

            alert(response.data.message);
            setSelectedFiles([]);
            setUploadProgress(0);
            await loadImages();
        } catch (error) {
            console.error('Upload error:', error);
            alert('Upload failed');
        } finally {
            setLoading(false);
        }
    };

    const predictImages = async () => {
        if (images.length === 0) {
            alert('No images to predict');
            return;
        }

        setLoading(true);
        setPredictionProgress(0);

        try {
            const imageData = images.map(img => ({
                filename: img.filename,
                originalName: img.filename
            }));

            const response = await axios.post(`${API_BASE_URL}/predict`, {
                images: imageData
            });

            // Convert predictions array to object for easy lookup
            const predictionsObj = {};
            response.data.predictions.forEach(pred => {
                predictionsObj[pred.filename] = pred;
            });

            setPredictions(predictionsObj);
            alert('Predictions completed!');
        } catch (error) {
            console.error('Prediction error:', error);
            alert('Prediction failed');
        } finally {
            setLoading(false);
            setPredictionProgress(0);
        }
    };

    const clearAll = async () => {
        if (window.confirm('Are you sure you want to clear all images and predictions?')) {
            try {
                await axios.delete(`${API_BASE_URL}/clear`);
                setImages([]);
                setPredictions({});
                setSelectedFiles([]);
                alert('All data cleared successfully');
            } catch (error) {
                console.error('Clear error:', error);
                alert('Failed to clear data');
            }
        }
    };

    const getConfidenceColor = (confidence) => {
        if (confidence >= 0.8) return '#4CAF50'; // Green
        if (confidence >= 0.6) return '#FF9800'; // Orange
        return '#F44336'; // Red
    };

    return (
        <div className="App">
            <header className="App-header">
                <h1>🖼️ CNN Image Classification System</h1>
                <p>Upload images and let our AI classify them using CIFAR-10 categories</p>
            </header>

            <main className="main-content">
                {/* Upload Section */}
                <section className="upload-section">
                    <h2>📤 Upload Images</h2>
                    <div className="upload-controls">
                        <input
                            type="file"
                            multiple
                            accept="image/*"
                            onChange={handleFileSelection}
                            className="file-input"
                            id="file-input"
                        />
                        <label htmlFor="file-input" className="file-input-label">
                            Choose Images (Max 100)
                        </label>
                        
                        {selectedFiles.length > 0 && (
                            <div className="selected-files-info">
                                <p>{selectedFiles.length} files selected</p>
                            </div>
                        )}
                        
                        <button 
                            onClick={uploadImages} 
                            disabled={loading || selectedFiles.length === 0}
                            className="btn btn-primary"
                        >
                            {loading ? 'Uploading...' : 'Upload Images'}
                        </button>

                        {uploadProgress > 0 && (
                            <div className="progress-bar">
                                <div 
                                    className="progress-fill" 
                                    style={{ width: `${uploadProgress}%` }}
                                ></div>
                                <span className="progress-text">{uploadProgress}%</span>
                            </div>
                        )}
                    </div>
                </section>

                {/* Control Buttons */}
                <section className="controls-section">
                    <h2>🎯 Classification Controls</h2>
                    <div className="controls">
                        <button 
                            onClick={predictImages}
                            disabled={loading || images.length === 0}
                            className="btn btn-success"
                        >
                            {loading ? 'Predicting...' : `Classify ${images.length} Images`}
                        </button>
                        
                        <button 
                            onClick={loadImages}
                            disabled={loading}
                            className="btn btn-secondary"
                        >
                            Refresh Images
                        </button>
                        
                        <button 
                            onClick={clearAll}
                            disabled={loading}
                            className="btn btn-danger"
                        >
                            Clear All
                        </button>
                    </div>
                </section>

                {/* Images Display Section */}
                <section className="images-section">
                    <h2>🖼️ Image Classification Results</h2>
                    
                    {images.length === 0 ? (
                        <div className="no-images">
                            <p>No images uploaded yet. Upload some images to get started!</p>
                        </div>
                    ) : (
                        <div className="images-grid">
                            {images.map((image, index) => {
                                const prediction = predictions[image.filename];
                                return (
                                    <div key={image.filename} className="image-card">
                                        <div className="image-container">
                                            <img 
                                                src={image.url} 
                                                alt={`Upload ${index + 1}`}
                                                className="image-preview"
                                            />
                                        </div>
                                        
                                        <div className="image-info">
                                            <h4 className="image-title">
                                                Image {index + 1}
                                            </h4>
                                            <p className="image-filename">
                                                {image.filename}
                                            </p>
                                            
                                            <div className="prediction-info">
                                                {prediction ? (
                                                    prediction.error ? (
                                                        <div className="prediction-error">
                                                            <span className="class-label error">Error</span>
                                                            <p className="error-message">{prediction.error}</p>
                                                        </div>
                                                    ) : (
                                                        <div className="prediction-success">
                                                            <span 
                                                                className="class-label"
                                                                style={{ 
                                                                    backgroundColor: getConfidenceColor(prediction.confidence) 
                                                                }}
                                                            >
                                                                {prediction.predictedClass}
                                                            </span>
                                                            <div className="confidence-info">
                                                                <span className="confidence-label">Confidence:</span>
                                                                <span className="confidence-value">
                                                                    {(prediction.confidence * 100).toFixed(1)}%
                                                                </span>
                                                            </div>
                                                            
                                                            {prediction.allProbabilities && (
                                                                <details className="all-probabilities">
                                                                    <summary>View All Probabilities</summary>
                                                                    <div className="probabilities-list">
                                                                        {Object.entries(prediction.allProbabilities)
                                                                            .sort(([,a], [,b]) => b - a)
                                                                            .map(([className, prob]) => (
                                                                                <div key={className} className="probability-item">
                                                                                    <span className="class-name">{className}:</span>
                                                                                    <span className="probability-value">
                                                                                        {(prob * 100).toFixed(1)}%
                                                                                    </span>
                                                                                </div>
                                                                            ))
                                                                        }
                                                                    </div>
                                                                </details>
                                                            )}
                                                        </div>
                                                    )
                                                ) : (
                                                    <span className="class-label pending">
                                                        Not classified yet
                                                    </span>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </section>

                {/* Class Reference */}
                <section className="reference-section">
                    <h2>📚 CIFAR-10 Classes Reference</h2>
                    <div className="classes-grid">
                        {CIFAR10_CLASSES.map((className, index) => (
                            <span key={className} className="class-reference">
                                {className}
                            </span>
                        ))}
                    </div>
                </section>
            </main>
        </div>
    );
}

export default App;