import argparse
import io
import os
from PIL import Image
import torch
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return "No file selected"
        
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model([img])

        results.render()
        
        int_directory = 0
        directory = f"./static/output{int_directory}"
        
        while os.path.exists(directory):
            int_directory += 1
            directory = f"./static/output{int_directory}"
            
        results.save(save_dir=directory)
        return redirect(f"{directory}/image0.jpg")
        
    return render_template("index.html")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  
    model.eval()
    app.run(host="0.0.0.0", port=args.port) 
