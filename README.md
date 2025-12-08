# Lung Segmentation Application

Medical image segmentation application using YOLOv8-seg for automated lung and body structure detection from chest X-ray images.

## 🎯 Features

- **Real-time Segmentation**: Fast and accurate segmentation using YOLOv8n-seg
- **Multi-class Detection**: Segments 4 classes:
  - Body
  - Cord (Spinal Cord)
  - Right Lung (Paru Kanan)
  - Left Lung (Paru Kiri)
- **Interactive Web Interface**: User-friendly Streamlit application
- **Metrics Dashboard**: Comprehensive detection metrics and statistics
- **Export Functionality**: Download results and metrics

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster inference

### Installation

1. **Clone or download this repository**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Place your trained model**:
   - Put your `best.pt` model file in the `models/` directory
   - Or train a new model using the provided notebook

### Running the Application

```bash
streamlit run src/app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📁 Project Structure

```
lung-segmentation-app/
├── src/
│   ├── app.py              # Main Streamlit application
│   ├── inference.py        # Model inference logic
│   └── utils.py            # Utility functions
├── models/
│   ├── best.pt             # Trained model weights
│   └── metadata.json       # Model metadata
├── assets/
│   └── sample_images/      # Sample images for testing
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🎮 Usage

1. **Upload Image**: Click "Choose an X-ray image..." and select a chest X-ray image
2. **Configure Parameters**: Adjust confidence and IoU thresholds in the sidebar
3. **View Results**: See segmentation results in multiple tabs:
   - **Visualization**: Segmented image with masks and labels
   - **Metrics**: Detection statistics and per-class information
   - **Export**: Download results and metrics

## ⚙️ Configuration

Edit `config.yaml` to customize:
- Model parameters (confidence, IoU thresholds)
- Visualization settings (colors, labels)
- Performance options (caching, device)

## 🔧 Model Training

To train your own model, use the provided Jupyter notebook:

```bash
jupyter notebook main.ipynb
```

Follow the notebook sections:
1. Data preparation
2. COCO to YOLO format conversion
3. Model training
4. Evaluation
5. Export for deployment

## 📊 Model Performance

- **Model**: YOLOv8n-seg (nano)
- **mAP@0.5**: ~75%
- **Input Size**: 640x640
- **Inference Speed**: ~30 FPS (GPU) / ~5 FPS (CPU)
- **Model Size**: ~6 MB

## 🛠️ Troubleshooting

### Model not found error
- Ensure `best.pt` is in the `models/` directory
- Check file path in `config.yaml`

### CUDA out of memory
- Reduce batch size in config
- Use CPU instead: set `device: "cpu"` in config

### Slow inference
- Use GPU if available
- Reduce image resolution
- Use lighter model (YOLOv8n-seg is already the lightest)

## 📝 Notes

- This application is for research and educational purposes only
- Not intended for clinical diagnosis
- Results should be verified by medical professionals

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 📧 Contact

For questions or issues, please open an issue on the repository.

---

**Developed with ❤️ using YOLOv8-seg and Streamlit**
