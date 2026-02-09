from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from datetime import datetime
import logging
import traceback
import io

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

API_KEY = "W2MJN5J0ruYBJsrAOLG6"
MODEL_ID = "pancreaticcancer-fnrsn/3"
URL = f"https://detect.roboflow.com/{MODEL_ID}?api_key={API_KEY}"

# Clinical classification for pancreatic cancer screening
CLINICAL_CLASSIFICATION = {
    "Suspicious_Mass": {
        "display_name": "Suspicious Solid Mass",
        "risk_level": "HIGH",
        "recommendation": "Urgent oncology referral. Consider biopsy and contrast-enhanced CT/MRI.",
        "color": "danger",
        "icon": "⚠️",
        "next_steps": ["Biopsy", "CT scan", "Oncology consult"]
    },
    "Cystic_Lesion": {
        "display_name": "Cystic Lesion",
        "risk_level": "MODERATE",
        "recommendation": "Monitor with follow-up imaging in 3-6 months. Consider EUS evaluation.",
        "color": "warning",
        "icon": "🟡",
        "next_steps": ["Follow-up imaging", "EUS", "Tumor markers"]
    },
    "Duct_Dilation": {
        "display_name": "Pancreatic Duct Dilation",
        "risk_level": "MODERATE",
        "recommendation": "Further evaluation with MRCP. Rule out intraductal papillary mucinous neoplasm (IPMN).",
        "color": "warning",
        "icon": "🟡",
        "next_steps": ["MRCP", "ERCP", "CA19-9 test"]
    },
    "Atrophy": {
        "display_name": "Pancreatic Atrophy",
        "risk_level": "LOW",
        "recommendation": "Monitor for symptoms. Consider chronic pancreatitis or aging changes.",
        "color": "info",
        "icon": "🔵",
        "next_steps": ["Clinical follow-up", "Enzyme tests", "Nutritional assessment"]
    },
    "Pancreatitis": {
        "display_name": "Pancreatitis",
        "risk_level": "MEDICAL",
        "recommendation": "Gastroenterology referral. Symptomatic management and identify etiology.",
        "color": "info",
        "icon": "🏥",
        "next_steps": ["Lipase test", "Alcohol cessation", "Pain management"]
    },
    "Calcifications": {
        "display_name": "Pancreatic Calcifications",
        "risk_level": "MODERATE",
        "recommendation": "Associated with chronic pancreatitis. Monitor for endocrine/exocrine insufficiency.",
        "color": "warning",
        "icon": "🟡",
        "next_steps": ["Diabetes screening", "Fecal elastase", "Vitamin levels"]
    },
    "Normal_Pancreas": {
        "display_name": "Normal Pancreas",
        "risk_level": "NONE",
        "recommendation": "No immediate action needed. Routine screening as per risk factors.",
        "color": "success",
        "icon": "✅",
        "next_steps": ["Continue regular screening", "Lifestyle counseling"]
    }
}

@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "service": "GastroVisionAI Pancreatic Cancer Detection",
        "version": "1.0.0",
        "endpoints": {
            "detection": "POST /predict",
            "health": "GET /health"
        }
    })

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_ID
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and pancreatic cancer screening"""
    start_time = datetime.now()
    
    try:
        logger.info(f"Received predict request. Files: {list(request.files.keys())}")
        
        # Check if image file exists
        if 'image' not in request.files:
            logger.warning("No image file in request")
            return jsonify({
                "success": False,
                "error": "No image uploaded",
                "message": "Please select an image file for analysis"
            }), 400

        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            logger.warning("Empty filename")
            return jsonify({
                "success": False,
                "error": "No file selected",
                "message": "Please select an image file"
            }), 400

        # Validate file type
        allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}
        filename_lower = file.filename.lower()
        if not any(filename_lower.endswith(ext) for ext in allowed_extensions):
            logger.warning(f"Invalid file type: {file.filename}")
            return jsonify({
                "success": False,
                "error": "Invalid file type",
                "message": f"Supported formats: PNG, JPG, JPEG, GIF, BMP"
            }), 400

        # Read file content
        file_content = file.read()
        file_size = len(file_content)
        
        if file_size == 0:
            logger.warning("Empty file uploaded")
            return jsonify({
                "success": False,
                "error": "Empty file",
                "message": "The uploaded file is empty"
            }), 400
        
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            logger.warning(f"File too large: {file_size} bytes")
            return jsonify({
                "success": False,
                "error": "File too large",
                "message": "File size must be less than 10MB"
            }), 400
        
        logger.info(f"Sending image to Roboflow: {file.filename}, Size: {file_size} bytes")
        
        # Reset file pointer
        file.seek(0)
        
        # Send to Roboflow
        files = {
            'file': (file.filename, file_content, file.mimetype or 'image/png')
        }
        
        headers = {
            'User-Agent': 'GastroVisionAI/1.0'
        }
        
        response = requests.post(URL, files=files, headers=headers, timeout=30)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Roboflow response status: {response.status_code}, Time: {processing_time:.2f}s")
        
        if response.status_code == 200:
            try:
                roboflow_data = response.json()
                logger.info(f"Roboflow response received. Predictions: {len(roboflow_data.get('predictions', []))}")
                
                # Process Roboflow response
                processed_result = process_roboflow_response(roboflow_data)
                
                return jsonify({
                    "success": True,
                    "processing_time": f"{processing_time:.2f}s",
                    "model_used": MODEL_ID,
                    "predictions": processed_result["predictions"],
                    "clinical_assessment": processed_result["clinical_assessment"]
                })
                
            except ValueError as e:
                logger.error(f"Failed to parse Roboflow JSON: {e}")
                logger.error(f"Response text: {response.text[:500]}")
                return jsonify({
                    "success": False,
                    "error": "Invalid response from AI model",
                    "message": "The AI model returned an invalid response.",
                    "details": "Could not parse JSON response"
                }), 500
            
        else:
            logger.error(f"Roboflow API error: {response.status_code}")
            logger.error(f"Response: {response.text[:500]}")
            
            error_msg = "Failed to analyze image"
            if response.status_code == 401:
                error_msg = "Invalid API key"
            elif response.status_code == 404:
                error_msg = "Model not found"
            elif response.status_code == 429:
                error_msg = "API rate limit exceeded"
            elif response.status_code >= 500:
                error_msg = "AI service temporarily unavailable"
            
            return jsonify({
                "success": False,
                "error": f"AI Model Error (Code: {response.status_code})",
                "message": error_msg,
                "details": response.text[:200] if response.text else "No details available"
            }), 500

    except requests.exceptions.Timeout:
        logger.error("Request timeout to Roboflow")
        return jsonify({
            "success": False,
            "error": "Request Timeout",
            "message": "AI analysis took too long. Please try again with a smaller image."
        }), 504
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Network Error",
            "message": "Cannot connect to AI service. Please check your internet connection."
        }), 503
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())  # This will print the full traceback
        return jsonify({
            "success": False,
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again.",
            "debug_info": str(e)
        }), 500

def process_roboflow_response(roboflow_data):
    """Process and enhance Roboflow response with clinical context"""
    
    # If no predictions, return normal pancreas
    if not roboflow_data.get('predictions') or len(roboflow_data['predictions']) == 0:
        normal_info = CLINICAL_CLASSIFICATION["Normal_Pancreas"]
        return {
            "predictions": [{
                "class": "Normal_Pancreas",
                "display_name": normal_info["display_name"],
                "confidence": 0.95,  # Default high confidence for normal
                "risk_level": normal_info["risk_level"],
                "recommendation": normal_info["recommendation"],
                "icon": normal_info["icon"],
                "next_steps": normal_info["next_steps"]
            }],
            "clinical_assessment": {
                "overall_risk": "NONE",
                "summary": "No abnormalities detected in pancreatic imaging.",
                "follow_up": "Routine screening recommended based on risk factors."
            }
        }
    
    processed_predictions = []
    
    for pred in roboflow_data['predictions']:
        class_name = pred.get('class', 'Unknown')
        
        # Get clinical context or default to unknown
        clinical_info = CLINICAL_CLASSIFICATION.get(
            class_name, 
            {
                "display_name": class_name.replace('_', ' ').title(),
                "risk_level": "UNKNOWN",
                "recommendation": "Further evaluation needed. Please consult a specialist.",
                "color": "unknown",
                "icon": "❓",
                "next_steps": ["Consult gastroenterology specialist"]
            }
        )
        
        enhanced_pred = {
            "class": class_name,
            "display_name": clinical_info["display_name"],
            "confidence": float(pred.get('confidence', 0)),
            "risk_level": clinical_info["risk_level"],
            "recommendation": clinical_info["recommendation"],
            "icon": clinical_info["icon"],
            "next_steps": clinical_info["next_steps"],
            "bounding_box": {
                "x": pred.get('x'),
                "y": pred.get('y'),
                "width": pred.get('width'),
                "height": pred.get('height')
            } if all(k in pred for k in ['x', 'y', 'width', 'height']) else None
        }
        
        processed_predictions.append(enhanced_pred)
    
    # Determine overall risk
    risk_levels = {"HIGH": 4, "MODERATE": 3, "LOW": 2, "MEDICAL": 3, "NONE": 1, "UNKNOWN": 2}
    overall_risk_value = 0
    for p in processed_predictions:
        risk_value = risk_levels.get(p["risk_level"], 2)
        if risk_value > overall_risk_value:
            overall_risk_value = risk_value
    
    risk_mapping_reverse = {4: "HIGH", 3: "MODERATE", 2: "LOW", 1: "NONE"}
    overall_risk = risk_mapping_reverse.get(overall_risk_value, "UNKNOWN")
    
    clinical_assessment = {
        "overall_risk": overall_risk,
        "findings_count": len(processed_predictions),
        "summary": generate_clinical_summary(processed_predictions),
        "follow_up": generate_follow_up_plan(processed_predictions)
    }
    
    return {
        "predictions": processed_predictions,
        "clinical_assessment": clinical_assessment
    }

def generate_clinical_summary(predictions):
    """Generate a clinical summary based on findings"""
    if not predictions:
        return "No pancreatic abnormalities detected."
    
    if len(predictions) == 1:
        pred = predictions[0]
        return f"Detected {pred['display_name'].lower()} with {pred['confidence']:.1%} confidence."
    else:
        findings = [f"{p['display_name']} ({p['confidence']:.1%})" for p in predictions]
        return f"Detected multiple findings: {', '.join(findings)}."

def generate_follow_up_plan(predictions):
    """Generate follow-up plan based on findings"""
    if not predictions:
        return "Continue with age-appropriate cancer screening."
    
    # Check if all predictions are normal
    if all(p['class'] == 'Normal_Pancreas' for p in predictions):
        return "No immediate action needed. Continue regular health screenings."
    
    # Get unique next steps from all predictions
    all_steps = []
    for pred in predictions:
        if pred['class'] != 'Normal_Pancreas':  # Only get steps for abnormal findings
            all_steps.extend(pred.get('next_steps', []))
    
    # Remove duplicates while preserving order
    unique_steps = []
    for step in all_steps:
        if step not in unique_steps:
            unique_steps.append(step)
    
    if unique_steps:
        return f"Recommended: {', '.join(unique_steps[:3])}."
    else:
        return "Consult with a healthcare provider for appropriate follow-up."

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 GastroVisionAI Pancreatic Cancer Detection")
    print("=" * 60)
    print(f"📡 Server: http://127.0.0.1:5000")
    print(f"🤖 Model: {MODEL_ID}")
    print(f"🔑 API Key: {'*' * 10}{API_KEY[-4:] if API_KEY else 'None'}")
    print(f"🏥 Classes: {', '.join(CLINICAL_CLASSIFICATION.keys())}")
    print("=" * 60)
    print("Starting server...")
    
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)