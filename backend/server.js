// Backend Server (server.js) - Node.js/Express
const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const mongoose = require('mongoose');

const app = express();
const PORT = process.env.PORT || 8000;

// Middleware
app.use(cors());
app.use(express.json());
app.use('/uploads', express.static('uploads'));

// MongoDB connection (with fallback)
mongoose.connect('mongodb://localhost:27017/image_classification', {
    useNewUrlParser: true,
    useUnifiedTopology: true,
    serverSelectionTimeoutMS: 5000 // Timeout after 5s instead of 30s
}).catch(err => {
    console.log('MongoDB not available, using in-memory storage for predictions');
    console.log('Install MongoDB to persist data: brew install mongodb-community');
    mongoAvailable = false;
});

// Schema for storing image predictions
const ImagePredictionSchema = new mongoose.Schema({
    filename: String,
    originalName: String,
    predictedClass: String,
    confidence: Number,
    timestamp: { type: Date, default: Date.now }
});

const ImagePrediction = mongoose.model('ImagePrediction', ImagePredictionSchema);

// In-memory storage fallback
let inMemoryPredictions = [];
let mongoAvailable = true;

// Multer configuration for file upload
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        const uploadDir = 'uploads/';
        if (!fs.existsSync(uploadDir)) {
            fs.mkdirSync(uploadDir, { recursive: true });
        }
        cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
    }
});

const upload = multer({ 
    storage: storage,
    fileFilter: (req, file, cb) => {
        const allowedTypes = /jpeg|jpg|png|gif/;
        const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
        const mimetype = allowedTypes.test(file.mimetype);
        
        if (mimetype && extname) {
            return cb(null, true);
        } else {
            cb('Error: Images Only!');
        }
    }
});

// CIFAR-10 class names
const CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
];

// Routes

// Upload multiple images
app.post('/api/upload', upload.array('images', 100), async (req, res) => {
    try {
        if (!req.files || req.files.length === 0) {
            return res.status(400).json({ error: 'No files uploaded' });
        }

        const uploadedFiles = req.files.map(file => ({
            filename: file.filename,
            originalName: file.originalname,
            path: file.path,
            size: file.size
        }));

        res.json({
            message: `${uploadedFiles.length} files uploaded successfully`,
            files: uploadedFiles
        });
    } catch (error) {
        console.error('Upload error:', error);
        res.status(500).json({ error: 'Upload failed' });
    }
});

// Get all uploaded images
app.get('/api/images', async (req, res) => {
    try {
        const uploadsDir = 'uploads/';
        
        if (!fs.existsSync(uploadsDir)) {
            return res.json({ images: [] });
        }

        const files = fs.readdirSync(uploadsDir);
        const imageFiles = files.filter(file => {
            const ext = path.extname(file).toLowerCase();
            return ['.jpg', '.jpeg', '.png', '.gif'].includes(ext);
        });

        const images = imageFiles.map(file => ({
            filename: file,
            path: `/uploads/${file}`,
            url: `http://localhost:${PORT}/uploads/${file}`
        }));

        res.json({ images });
    } catch (error) {
        console.error('Error getting images:', error);
        res.status(500).json({ error: 'Failed to get images' });
    }
});

// Predict image classes using the best model
app.post('/api/predict', async (req, res) => {
    try {
        const { images } = req.body;
        
        if (!images || images.length === 0) {
            return res.status(400).json({ error: 'No images provided for prediction' });
        }

        // Create a Python script call to use the trained model
        const predictions = await predictImages(images);
        
        // Save predictions to database or memory
        if (mongoAvailable) {
            const savedPredictions = await Promise.all(
                predictions.map(async (pred) => {
                    const imagePred = new ImagePrediction({
                        filename: pred.filename,
                        originalName: pred.originalName,
                        predictedClass: pred.predictedClass,
                        confidence: pred.confidence
                    });
                    return await imagePred.save();
                })
            );
        } else {
            // Store in memory
            predictions.forEach(pred => {
                inMemoryPredictions.push({
                    filename: pred.filename,
                    originalName: pred.originalName,
                    predictedClass: pred.predictedClass,
                    confidence: pred.confidence,
                    timestamp: new Date()
                });
            });
        }

        res.json({
            message: 'Predictions completed',
            predictions: predictions
        });
    } catch (error) {
        console.error('Prediction error:', error);
        res.status(500).json({ error: 'Prediction failed' });
    }
});

// Helper function to call Python prediction script
function predictImages(images) {
    return new Promise((resolve, reject) => {
        const shellScript = path.join(__dirname, 'run_prediction.sh');
        const imageList = JSON.stringify(images);
        
        const python = spawn('bash', [shellScript, imageList]);
        
        let dataString = '';
        
        python.stdout.on('data', (data) => {
            dataString += data.toString();
        });
        
        python.stderr.on('data', (data) => {
            console.error(`Python error: ${data}`);
        });
        
        python.on('close', (code) => {
            if (code === 0) {
                try {
                    const predictions = JSON.parse(dataString);
                    resolve(predictions);
                } catch (error) {
                    reject(new Error('Failed to parse prediction results'));
                }
            } else {
                reject(new Error(`Python script exited with code ${code}`));
            }
        });
    });
}

// Get prediction history
app.get('/api/predictions', async (req, res) => {
    try {
        if (mongoAvailable) {
            const predictions = await ImagePrediction.find().sort({ timestamp: -1 });
            res.json({ predictions });
        } else {
            // Return from memory
            const sortedPredictions = inMemoryPredictions.sort((a, b) => b.timestamp - a.timestamp);
            res.json({ predictions: sortedPredictions });
        }
    } catch (error) {
        console.error('Error getting predictions:', error);
        res.status(500).json({ error: 'Failed to get predictions' });
    }
});

// Delete all uploaded images and predictions
app.delete('/api/clear', async (req, res) => {
    try {
        // Clear uploads directory
        const uploadsDir = 'uploads/';
        if (fs.existsSync(uploadsDir)) {
            const files = fs.readdirSync(uploadsDir);
            files.forEach(file => {
                fs.unlinkSync(path.join(uploadsDir, file));
            });
        }
        
        // Clear database or memory
        if (mongoAvailable) {
            await ImagePrediction.deleteMany({});
        } else {
            inMemoryPredictions = [];
        }
        
        res.json({ message: 'All data cleared successfully' });
    } catch (error) {
        console.error('Clear error:', error);
        res.status(500).json({ error: 'Failed to clear data' });
    }
});

// Start server
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
    console.log(`Upload endpoint: http://localhost:${PORT}/api/upload`);
    console.log(`Images endpoint: http://localhost:${PORT}/api/images`);
});

module.exports = app;