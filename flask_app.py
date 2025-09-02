from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load your trained model
model = load_model("best_model.keras")
class_names = ["NORMAL", "PNEUMONIA"]

def predict_xray(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)   # no /255 (already in model)
    prediction = model.predict(img_array)
    prob_normal, prob_pneumonia = prediction[0]
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class, float(prob_pneumonia)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join("uploads", file.filename)
            file.save(filepath)
            pred_class, prob = predict_xray(filepath)
            return f"Prediction: {pred_class}, Pneumonia probability: {prob*100:.4f}"
    return '''
    <h1>Chest X-Ray Classifier</h1>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit" value="Upload">
    </form>
    '''

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
