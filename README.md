# Deep Learning Image Processing Web App

## Introduction
This project is a **Deep Learning Image Processing Web Application** that allows users to upload images and process them using different deep learning models. The application supports:
- **Object Detection**: Identifies objects in the image and marks them with bounding boxes.
- **Segmentation**: Classifies each pixel in the image to separate different objects.
- **Classification**: Predicts the category of the uploaded image and provides confidence scores for the top predictions.

The web app is built using **Flask** as the backend, integrates **PyTorch** for deep learning model inference, and leverages **OpenCV** for image processing.

---

## Features
- **User-Friendly Interface**: Upload images and select the model type via a simple web form.
- **Real-Time Image Processing**: Process images dynamically and display results instantly.
- **Flexible Model Support**: Supports classification, object detection, and segmentation.
- **Results Display**:
  - Shows the original image alongside the processed output.
  - For classification, lists the top-5 predicted labels with probabilities.

---

## Installation
### Prerequisites
Ensure you have **Python 3.8+** installed.

### Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/deep-learning-image-processing.git
   cd deep-learning-image-processing
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask app:**
   ```bash
   python app.py
   ```

5. **Access the web interface:**
   Open your browser and go to:
   ```
   http://127.0.0.1:5000/
   ```

---

## Usage
1. **Upload an Image**: Click the "Choose File" button and select an image.
2. **Select a Model Type**: Choose from Classification, Detection, or Segmentation.
3. **Process the Image**: Click the "Process Image" button.
4. **View Results**:
   - For **Object Detection & Segmentation**, the processed image is displayed with overlays.
   - For **Classification**, a list of top predictions with probabilities is displayed next to the original image.

---

## Project Structure
```
├── static/          # Stores uploaded images and processed results
├── templates/       # HTML templates for the web interface
│   ├── index.html   # Main UI page
├── app.py           # Flask backend logic
├── labels_map.txt   # Lables of classification Dataset
├── requirements.txt # Required dependencies
└── README.md        # Project documentation
```

---

## Future Enhancements
- Add more deep learning models (e.g., GANs, Style Transfer, Super Resolution).
- Improve UI with Bootstrap or React.
- Deploy to cloud services (AWS, Google Cloud, or Heroku).

---

## Contributors
- **Sathish Kumar Prabaharan** - Developer

Feel free to fork, contribute, and enhance this project!

---

## Contact
For any issues or improvements, feel free to open an **Issue** or **Pull Request** on GitHub.

