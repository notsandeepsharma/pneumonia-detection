import os
import base64
import io
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__, template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to read image file. Please ensure the file is a valid image.")
    
    img = cv2.resize(img, (512, 512))
    img_blurred = cv2.GaussianBlur(img, (5, 5), 0)
    img_equalized = cv2.equalizeHist(img_blurred)
    
    return img, img_blurred, img_equalized


def approximate_lung_region(img_equalized):
    h, w = img_equalized.shape
    crop_margin = int(w * 0.1)
    img_cropped = img_equalized[crop_margin:h-crop_margin, crop_margin:w-crop_margin]
    
    edges = cv2.Canny(img_cropped, 50, 150)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    lung_mask = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel_large)
    
    full_mask = np.zeros_like(img_equalized)
    full_mask[crop_margin:h-crop_margin, crop_margin:w-crop_margin] = lung_mask
    
    return full_mask


def calculate_opacity_score(img_equalized, lung_mask):
    lung_region = cv2.bitwise_and(img_equalized, img_equalized, mask=lung_mask)
    
    threshold = 180
    bright_pixels = np.sum(lung_region > threshold)
    
    total_lung_pixels = np.sum(lung_mask > 0)
    
    if total_lung_pixels == 0:
        return 0.0
    
    opacity_score = bright_pixels / total_lung_pixels
    return float(opacity_score)


def calculate_contrast_ratio(img_equalized, lung_mask):
    lung_region = cv2.bitwise_and(img_equalized, img_equalized, mask=lung_mask)
    
    lung_pixels = lung_region[lung_mask > 0]
    
    if len(lung_pixels) == 0:
        return 0.0
    
    contrast_ratio = float(np.std(lung_pixels) / 255.0)
    return contrast_ratio


def detect_regions(img_equalized, lung_mask):
    lung_region = cv2.bitwise_and(img_equalized, img_equalized, mask=lung_mask)
    
    laplacian = cv2.Laplacian(lung_region, cv2.CV_64F)
    laplacian_abs = np.abs(laplacian)
    
    laplacian_normalized = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    _, suspicious = cv2.threshold(laplacian_normalized, 50, 255, cv2.THRESH_BINARY)
    
    suspicious_pixels = np.sum(suspicious > 0)
    total_lung_pixels = np.sum(lung_mask > 0)
    
    if total_lung_pixels == 0:
        region_flag = 0.0
    else:
        region_flag = suspicious_pixels / total_lung_pixels
    
    return float(region_flag), suspicious


def generate_heatmap(img_original, suspicious_regions, lung_mask):
    heatmap = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
    
    heatmap[suspicious_regions > 0] = [0, 0, 255]
    
    overlay = heatmap.copy()
    heatmap = cv2.addWeighted(img_original[:, :, np.newaxis].repeat(3, axis=2), 0.7, overlay, 0.3, 0)
    
    return heatmap


def classify_risk(opacity_score, contrast_ratio, region_flag):
    score = (opacity_score * 0.4) + (contrast_ratio * 0.3) + (region_flag * 0.3)
    
    if score < 0.15:
        risk_level = "Low"
        explanation = "Minimal indicators of pneumonia patterns detected."
    elif score < 0.30:
        risk_level = "Moderate"
        explanation = "Some indicators of potential pneumonia patterns detected. Further evaluation recommended."
    else:
        risk_level = "High"
        explanation = "Multiple indicators of pneumonia patterns detected. Professional medical evaluation strongly recommended."
    
    return risk_level, explanation, float(score)


def analyze_xray(image_path):
    try:
        if not os.path.exists(image_path):
            raise ValueError("Image file not found")
        
        img_original, img_blurred, img_equalized = preprocess_image(image_path)
        
        lung_mask = approximate_lung_region(img_equalized)
        
        opacity_score = calculate_opacity_score(img_equalized, lung_mask)
        contrast_ratio = calculate_contrast_ratio(img_equalized, lung_mask)
        region_flag, suspicious_regions = detect_regions(img_equalized, lung_mask)
        
        risk_level, explanation, overall_score = classify_risk(opacity_score, contrast_ratio, region_flag)
        
        heatmap = generate_heatmap(img_original, suspicious_regions, lung_mask)
        
        _, buffer = cv2.imencode('.png', heatmap)
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            'success': True,
            'risk_level': risk_level,
            'opacity_score': round(opacity_score * 100, 2),
            'contrast_ratio': round(contrast_ratio * 100, 2),
            'region_flag': round(region_flag * 100, 2),
            'overall_score': round(overall_score * 100, 2),
            'explanation': explanation,
            'heatmap_image': f'data:image/png;base64,{heatmap_base64}'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/test_xray.png')
def serve_test_image():
    import os
    test_image_path = os.path.join(os.path.dirname(__file__), 'test_xray.png')
    if os.path.exists(test_image_path):
        from flask import send_file
        return send_file(test_image_path, mimetype='image/png')
    return 'Test image not found', 404


@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'File type not allowed. Use JPG, PNG, GIF, or BMP'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = analyze_xray(filepath)
        
        os.remove(filepath)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
