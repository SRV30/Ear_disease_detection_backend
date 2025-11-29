from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import models, transforms
from PIL import Image
import os
from werkzeug.utils import secure_filename
import torch.nn.functional as F
import numpy as np

app = Flask(__name__)
CORS(app)  # React se requests allow

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
device = torch.device('cpu')

classes_13 = ['000', '001', '002', '003', '004', '005', '006', '007',
              '008', '009', '010', '011', '012']


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --------- Model Load ----------
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 13)
state_dict = torch.load(
    'saved_models/resnet18_finetuned.pth',
    map_location=device
)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# --------- Grad-CAM ----------
def generate_and_save_gradcam(model, img_tensor, img_for_overlay, target_indices, save_path):
    conv_features = []
    gradients = []

    def forward_hook(m, i, o):
        conv_features.append(o)

    def backward_hook(m, gi, go):
        gradients.append(go[0])

    h1 = model.layer4.register_forward_hook(forward_hook)
    h2 = model.layer4.register_backward_hook(backward_hook)

    img_tensor.requires_grad_(True)
    outputs = model(img_tensor)

    idx = torch.tensor(target_indices, device=device)
    target = outputs[0, idx].sum()

    model.zero_grad()
    target.backward()

    h1.remove()
    h2.remove()

    f = conv_features[0]
    g = gradients[0]

    w = g.mean(dim=(2, 3))

    cam = torch.zeros(f.shape[2:], device=device)
    for i in range(w.shape[1]):
        cam += w[0, i] * f[0, i]

    cam = torch.relu(cam)
    cam -= cam.min()
    cam /= cam.max() if cam.max() != 0 else 1

    cam = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(img_for_overlay.size[1], img_for_overlay.size[0]),
        mode='bilinear',
        align_corners=False
    )[0, 0].detach().cpu().numpy().copy()

    img_np = np.array(img_for_overlay).astype(np.float32) / 255.0
    heatmap = np.zeros_like(img_np)
    heatmap[..., 0] = cam
    heatmap[..., 1] = cam * 0.5

    overlay = np.clip(heatmap * 0.5 + img_np * 0.7, 0, 1)
    Image.fromarray((overlay * 255).astype(np.uint8)).save(save_path)


# --------- API ----------
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")

    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    image_url = '/' + file_path.replace("\\", "/")

    img = Image.open(file_path).convert("RGB")
    img_resized = img.resize((224, 224))
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        p = F.softmax(outputs, dim=1)[0]

    prob_normal = p[0:8].sum().item()
    prob_wax = p[8:11].sum().item()
    prob_infection = p[11:13].sum().item()
    macro_probs = [prob_normal, prob_wax, prob_infection]

    labels = ["Normal", "Wax", "Infection"]
    idx = int(torch.tensor(macro_probs).argmax())
    pred_label = labels[idx]
    confidence = macro_probs[idx] * 100
    probs_percent = [round(v * 100, 2) for v in macro_probs]

    if pred_label == "Normal":
        target = list(range(0, 8))
    elif pred_label == "Wax":
        target = list(range(8, 11))
    else:
        target = list(range(11, 13))

    heatmap_filename = f"heatmap_{os.path.splitext(filename)[0]}_{np.random.randint(10000)}.jpg"
    heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)

    generate_and_save_gradcam(model, img_t, img_resized, target, heatmap_path)

    heatmap_url = '/' + heatmap_path.replace("\\", "/")

    return jsonify({
        "label": pred_label,
        "confidence": round(confidence, 2),
        "probabilities": probs_percent,
        "image_url": image_url,
        "heatmap_url": heatmap_url
    })


@app.route("/ping")
def ping():
    return "pong"


if __name__ == "__main__":
    app.run(debug=True)
