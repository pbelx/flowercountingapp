import os
from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, jwt_required, create_access_token
from werkzeug.security import check_password_hash, generate_password_hash
from PIL import Image
import io
import torch
from torchvision.transforms import functional as F
from transformers import DetrImageProcessor, DetrForObjectDetection
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins



#app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'your-secret-key')  # Change this!
jwt = JWTManager(app)

# Mock user database
users = {
    "admin": generate_password_hash("password")
}

# Mock job database
jobs = {}

# Load the model and image processor
model_name = "smutuvi/flower_count_model"  # Replace with the actual model name
image_processor = DetrImageProcessor.from_pretrained(model_name)
model = DetrForObjectDetection.from_pretrained(model_name)
model.eval()  # Set the model to evaluation mode

def process_image(image_bytes):
    # Open the image
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Prepare the image
    inputs = image_processor(images=image, return_tensors="pt")
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process the outputs
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]
    
    # Count the number of detected flowers
    flower_count = len(results["scores"])
    
    return flower_count

@app.route('/auth/token', methods=['POST'])
def login():
    username = request.json.get('username', None)
    password = request.json.get('password', None)
    
    if username not in users or not check_password_hash(users[username], password):
        return jsonify({"message": "Bad username or password"}), 401

    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token)

@app.route('/images/batch', methods=['POST'])
@jwt_required()
def batch_process():
    if 'images' not in request.files:
        return jsonify({"message": "No images in request"}), 400

    results = []
    for image in request.files.getlist('images'):
        flower_count = process_image(image.read())
        results.append({
            "image_id": image.filename,
            "flower_count": flower_count
        })

    return jsonify({"results": results})

@app.route('/images/<image_id>', methods=['GET'])
@jwt_required()
def get_image_count(image_id):
    # In a real application, you would fetch this from a database
    # Here, we're just returning a mock result
    return jsonify({
        "image_id": image_id,
        "flower_count": 5  # Mock count
    })

@app.route('/dataset/process', methods=['POST'])
@jwt_required()
def process_dataset():
    # In a real application, you would start a background job here
    # For this example, we'll just return a mock job ID
    job_id = "123456"
    jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "results": []
    }
    return jsonify({"job_id": job_id}), 202

@app.route('/dataset/status/<job_id>', methods=['GET'])
@jwt_required()
def get_job_status(job_id):
    if job_id not in jobs:
        return jsonify({"message": "Job not found"}), 404

    return jsonify(jobs[job_id])

if __name__ == '__main__':
    app.run(debug=True)
 
