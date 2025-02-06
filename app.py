from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
import cv2
import os
import numpy as np
import json
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models import resnet152, ResNet152_Weights
import matplotlib.pyplot as plt

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
# COCO class labels
COCO_CLASSES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 
    43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 
    49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 
    59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 
    64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 
    72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 
    77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 
    81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 
    87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
}
# Load pre-trained models
models = {
    "Detection": maskrcnn_resnet50_fpn_v2(weights="DEFAULT", pretrained=True),
    "Segmentation": torch.hub.load("pytorch/vision:v0.10.0", "deeplabv3_resnet101", pretrained=True),
    "Classification": resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
}

# Preprocessing function
def preprocess_image(image, model_type):
    # image = Image.open(image).convert("RGB")
    if model_type == "Detection":
        return transforms.functional.to_tensor(image).unsqueeze(0)
    elif model_type == "Segmentation":
        transform = transforms.Compose([
        transforms.Resize((520, 520)),  # Resize to 520x520
        transforms.ToTensor(),  # Convert to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    elif model_type == "Classification":
        transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

    return transform(image).unsqueeze(0)

# Function to process image
def process_image(model_type, image):
    model = models[model_type]
    model.eval()

    img_tensor = preprocess_image(image, model_type)

    with torch.no_grad():
        output = model(img_tensor)

    if model_type == "Detection":
        return output[0]  # Return bounding boxes
    elif model_type == "Segmentation":
        return output["out"]  # Return segmentation masks
    elif model_type == "Classification":
        class_id = torch.argmax(output).item()
        return f"Predicted Class ID: {class_id}"

def draw_predictions(image, outputs, threshold=0.5):
    """Draws bounding boxes, masks, and labels on the image."""
    image_np = np.array(image)
    
    for i, score in enumerate(outputs[0]['scores'].cpu().numpy()):
        if score > threshold:
            # Get bounding box
            bbox = outputs[0]['boxes'][i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = bbox

            # Get class label
            class_id = int(outputs[0]['labels'][i].cpu().numpy())
            label = COCO_CLASSES.get(class_id, "Unknown")

            # Draw bounding box
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 1)

            # Draw label
            cv2.putText(image_np, f"{label} ({score:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return image_np

# Overlay mask on image
def overlay_mask(image, mask, alpha=0.5):
    image = np.array(image)
    image = cv2.resize(image, (mask.shape[1], mask.shape[0]))

    # Assign random colors
    num_classes = mask.max() + 1
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    colored_mask = colors[mask]

    # Blend image and mask
    overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    return overlay

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        model_type = request.form.get("model_type")
        image_file = request.files["image"]
        img_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        image_file.save(img_path)
        if image_file:
            image = Image.open(image_file)
            model = models[model_type]
            model.eval()

            img_tensor = preprocess_image(image, model_type)
            # Save processed image
            output_path = os.path.join(PROCESSED_FOLDER, "output.png")

            with torch.no_grad():
                output = model(img_tensor)

            if model_type == "Detection":
                result = draw_predictions(image, output)  # Return bounding boxes
                cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            elif model_type == "Segmentation":
                output = output["out"][0]  # Return segmentation masks
                result = output.argmax(0).byte().cpu().numpy()
                overlay = overlay_mask(image, result)
                cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            elif model_type == "Classification":
                preds = torch.topk(output, k=5).indices.squeeze(0).tolist()
                labels_map = json.load(open('labels_map.txt'))
                labels_map = [labels_map[str(i)] for i in range(1000)]
                output_res ={}
                for idx in preds:
                    label = labels_map[idx]
                    prob = torch.softmax(output, dim=1)[0, idx].item()
                    output_res[label] = f"({prob * 100:.2f}%)"
                output_path = output_res

            return render_template("index.html", result=output_path, image=img_path, model_type=model_type)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
