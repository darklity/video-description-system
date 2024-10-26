# Video Description System

A real-time video analysis system that provides automatic descriptions of scenes without audio. This system uses computer vision and machine learning to detect objects, analyze scenes, and generate natural language descriptions of what it sees.

## 🎯 Features

- Real-time object detection using YOLOv5
- Natural language scene descriptions
- Spatial relationship analysis
- Object counting and tracking
- Timestamp overlay
- Comprehensive logging system
- Visual bounding boxes with confidence scores
- Description updates at configurable intervals

## 🔧 Requirements

### Hardware Requirements
- Camera or video input device
- CUDA-capable GPU (recommended for better performance)
- Minimum 8GB RAM
- x86_64 processor

### Software Requirements
- Python 3.8 or higher
- CUDA Toolkit 11.0+ (for GPU support)
- Git

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/video-description-system.git
cd video-description-system
```

2. Create and activate a virtual environment (recommended):
```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install the required packages:
```bash
# Install PyTorch (CPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch (GPU version - recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # for CUDA 11.8

# Install other dependencies
pip install -r requirements.txt
```

### Requirements.txt
```txt
opencv-python==4.8.1.78
numpy>=1.21.0
pandas>=1.3.0
transformers>=4.30.0
scikit-image>=0.19.0
torch>=2.0.0
torchvision>=0.15.0
```

## 🚀 Usage

1. Basic usage with default settings:
```bash
python video_description.py
```

2. To use a specific video file instead of webcam:
```python
from video_description import VideoDescriptionSystem

video_system = VideoDescriptionSystem()
video_system.run("path/to/your/video.mp4")
```

3. To customize configuration:
```python
video_system = VideoDescriptionSystem()
video_system.config.update({
    "frame_width": 1280,
    "frame_height": 720,
    "confidence_threshold": 0.6,
    "description_update_interval": 1
})
video_system.run()
```

## ⌨️ Controls

- Press 'q' to quit the application
- The system will automatically update descriptions based on the configured interval

## 📁 Project Structure

```
video-description-system/
│
├── video_description.py     # Main application file
├── requirements.txt         # Project dependencies
├── README.md               # Project documentation
└── logs/                   # Log files directory
    └── video_description.log
```

## 🔍 Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| video_source | 0 | Camera index or video file path |
| frame_width | 640 | Width of processed frames |
| frame_height | 480 | Height of processed frames |
| confidence_threshold | 0.5 | Minimum confidence for object detection |
| fps | 30 | Target frames per second |
| description_update_interval | 2 | Seconds between description updates |
| max_objects_per_description | 5 | Maximum objects to include in description |

## 🐛 Troubleshooting

1. If you encounter CUDA out of memory errors:
   - Reduce frame_width and frame_height in the configuration
   - Increase description_update_interval

2. If the video feed is not opening:
   - Check if the correct video_source is specified
   - Verify camera permissions
   - Ensure the video file path is correct

3. For performance issues:
   - Enable GPU support by installing the CUDA version of PyTorch
   - Adjust the frame size and processing intervals
   - Close other GPU-intensive applications

## 📝 Logging

The system creates detailed logs in both the console and a log file:
- Location: `logs/video_description.log`
- Log Level: INFO
- Format: `timestamp - level - message`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YOLOv5 by Ultralytics
- Hugging Face Transformers
- OpenCV community

## 📞 Support

For support, please:
1. Check the troubleshooting section
2. Review existing issues on GitHub
3. Create a new issue with:
   - Detailed description of the problem
   - Steps to reproduce
   - System information
   - Log files

---
**Note**: This project is for educational and research purposes. Performance may vary based on hardware capabilities and input sources.
