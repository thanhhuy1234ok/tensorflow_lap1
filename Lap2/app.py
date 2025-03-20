from flask import Flask, request, jsonify
from google.cloud import vision
import os
from werkzeug.utils import secure_filename

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] ="AIzaSyBRTZOxiA-KtDYkg7i3qCAeXMtYvOG6uuE"
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)



@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    with open(filepath, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    
    if not texts:
        return jsonify({'text': ''})
    
    detected_text = texts[0].description
    
    return jsonify({'text': detected_text})

if __name__ == '__main__':
    app.run(debug=True)