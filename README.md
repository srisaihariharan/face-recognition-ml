# Face Recognition ML

A standalone, lightweight Face Recognition ML module extracted from the Face-Recognition-Attendance-System. This library implements LBPH (Local Binary Pattern Histograms) based face detection and recognition using OpenCV.

## Features

- **Face Detection**: Using Haar Cascade Classifier for fast and reliable face detection
- **Face Recognition**: LBPH (Local Binary Pattern Histogram) algorithm for accurate face matching
- **Real-time Processing**: Process video streams with live face recognition
- **Easy Integration**: Simple Python API for integration into any application
- **Lightweight**: Minimal dependencies, runs efficiently on standard hardware

## Technologies

- **Language**: Python 3.7+
- **Face Detection**: Haar Cascade Classifier
- **Face Recognition Algorithm**: LBPH (Local Binary Pattern Histogram)
- **Key Libraries**:
  - OpenCV (opencv-contrib-python >= 4.0.1)
  - NumPy
  - Pandas
  - Pillow

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/srisaihariharan/face-recognition-ml.git
cd face-recognition-ml
```

2. Create virtual environment (recommended):
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
face-recognition-ml/
├── recognize.py          # Main face recognition module
├── train.py             # Model training script
├── capture.py           # Face image capture utility
├── requirements.txt      # Project dependencies
└── README.md            # This file
```

## Core Modules

### recognize.py
Main module for real-time face recognition. Features:
- Real-time video stream processing
- Face detection and recognition
- Confidence score calculation
- Optional CSV logging of recognition results

### train.py
Training module for face recognition model. Features:
- Train LBPH recognizer from labeled face images
- Save trained model for later use
- Support for multiple faces per person

### capture.py
Utility for capturing face images for training. Features:
- Real-time face capture from webcam
- Automatic image preprocessing
- Organized storage of training data

## Usage

### Face Recognition
```python
from recognize import recognize_faces

# Run face recognition on video stream
recognize_faces(
    model_path='models/trained_model.yml',
    cascade_path='haarcascade_frontalface_default.xml',
    student_csv='student_data.csv'
)
```

### Training a Model
```python
from train import train_model

# Train model on face images
train_model(
    training_images_path='training_images/',
    output_model_path='models/trained_model.yml'
)
```

## Algorithm Details

### Haar Cascade Classifier
- Pre-trained cascade classifier for frontal face detection
- Fast multi-scale face detection
- Configurable sensitivity and scale factors

### LBPH (Local Binary Pattern Histogram)
- Local texture-based face recognition
- Divides face image into small regions
- Extracts LBP histograms from each region
- Compares histograms for face matching
- Provides confidence score for each match

## Configuration

### Video Capture Settings
- Resolution: 640x480 (configurable)
- Minimum face size: 10% of frame dimensions
- Detection scale factor: 1.2
- Detection neighbors: 5

### Recognition Confidence Thresholds
- **Strong match** (> 67% confidence): Green label
- **Medium match** (50-67% confidence): Yellow label  
- **Weak match** (< 50% confidence): Red label

## Requirements

See `requirements.txt` for complete list:
- opencv-contrib-python >= 4.0.1
- numpy >= 1.15.4
- Pillow >= 5.4.1
- pandas >= 0.24.0
- yagmail >= 0.11.224 (optional, for email integration)

## Performance

- Face Detection: Real-time on standard webcams
- Recognition Speed: ~50-100ms per frame (depends on hardware)
- Accuracy: ~95% with well-trained model on controlled lighting

## Known Limitations

- Works best with frontal face images
- Lighting conditions affect recognition accuracy
- Requires sufficient training samples per person (10-30+ images)
- Single face per frame for optimal performance

## Future Enhancements

- [ ] Support for deep learning models (CNN, FaceNet)
- [ ] Profile face detection
- [ ] Expression and emotion detection
- [ ] GPU acceleration support
- [ ] REST API endpoint
- [ ] Mobile-friendly deployment

## Troubleshooting

### No faces detected
- Ensure adequate lighting
- Check camera resolution and orientation
- Verify Haar cascade file path
- Adjust detection sensitivity parameters

### Low recognition accuracy
- Increase training samples
- Improve training image quality and lighting consistency
- Check if faces are similar (twins, siblings)
- Fine-tune confidence thresholds

### Performance issues
- Reduce video resolution
- Increase detection scale factor
- Skip frames for processing
- Consider hardware acceleration

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

MIT License - See LICENSE file for details

## Original Source

Extracted from: [kmhmubin/Face-Recognition-Attendance-System](https://github.com/kmhmubin/Face-Recognition-Attendance-System)

## Disclaimer

This project is for educational and research purposes. Ensure compliance with local laws and regulations regarding facial recognition technology and privacy.
