from flask import Flask, request, jsonify
from flask import Flask, render_template
from werkzeug.utils import secure_filename
import os
from runPrediction import makePrediction
from keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg'}

model = load_model("Brain_MRI_scanner.h5")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return open('web/index.html').read()

@app.route('/process_image', methods=['POST'])
def process_image_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'})

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected image'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(filepath)
        prediction = makePrediction(model, filepath)
        return jsonify({'prediction': prediction})
    else:
        return jsonify({'error': 'Invalid file'})

if __name__ == '__main__':
    app.run(debug=True)
