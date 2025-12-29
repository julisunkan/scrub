import os
import cv2
import numpy as np
import pytesseract
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
import zipfile
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

def remove_text_from_image(image_path, output_path):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    # Convert to grayscale for OCR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Get text data from OCR
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    
    # Create a mask for inpainting
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        if int(data['conf'][i]) > 0 and data['text'][i].strip():  # Only boxes with text and confidence
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            # Draw a white rectangle on the mask where text is found
            # Use smaller padding to be more precise
            cv2.rectangle(mask, (x-1, y-1), (x + w + 1, y + h + 1), 255, -1)
    
    # Inpaint the original image using the mask
    # TELEA algorithm is generally better for thin structures like text
    result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    
    # Save the result
    cv2.imwrite(output_path, result)
    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'images' not in request.files:
        return jsonify({'error': 'No images uploaded'}), 400
    
    files = request.files.getlist('images')
    processed_files = []
    
    for file in files:
        if file is None or file.filename == '' or file.filename is None:
            continue
        
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        
        processed_filename = 'processed_' + filename
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        
        if remove_text_from_image(upload_path, processed_path):
            processed_files.append({
                'original': filename,
                'processed': processed_filename,
                'url': f'/static/processed/{processed_filename}'
            })
            
    return jsonify({'files': processed_files})

@app.route('/download-zip', methods=['POST'])
def download_zip():
    filenames = request.json.get('filenames', [])
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        for filename in filenames:
            file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
            if os.path.exists(file_path):
                zf.write(file_path, filename)
    
    memory_file.seek(0)
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name='processed_images.zip'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
