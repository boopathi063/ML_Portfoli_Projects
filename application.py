from flask import Flask, render_template, request, jsonify
import logging
from datetime import datetime
import os
from src.utils import predict_subscription

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

application = Flask(__name__)
app = application
@app.route("/", methods=["GET"])
def index():
    """Home page with prediction form"""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests"""
    try:
        # Collect inputs from form
        input_data = {}
        for key in request.form.keys():
            value = request.form.get(key)
            if value != "":
                input_data[key] = value

        # Convert numeric fields
        numeric_fields = [
            "age", "balance", "duration", "campaign", "previous", "pdays",
            "euribor3m", "cons.conf.idx", "emp.var.rate", "nr.employed"
        ]
        for field in numeric_fields:
            if field in input_data:
                input_data[field] = float(input_data[field])

        # Get prediction
        pred, proba = predict_subscription(input_data)

        # Risk level
        if proba >= 0.7:
            risk_level = "HIGH"
            recommendation = "🎯 Strong candidate - Prioritize for contact"
            css_class = "high-prob"
        elif proba >= 0.4:
            risk_level = "MEDIUM"
            recommendation = "⚡ Moderate candidate - Include in campaign"
            css_class = "medium-prob"
        else:
            risk_level = "LOW"
            recommendation = "📧 Low priority - Consider digital marketing only"
            css_class = "low-prob"

        result_data = {
            "prediction": "Will Subscribe" if pred == 1 else "Will Not Subscribe",
            "probability": f"{proba:.1%}",
            "probability_raw": proba,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "css_class": css_class,
            "input_data": input_data,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return render_template("result.html", **result_data)

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return render_template("index.html",
                               error="❌ Something went wrong. Please try again.",
                               input_data=request.form.to_dict())

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        pred, proba = predict_subscription(data)

        return jsonify({
            "success": True,
            "prediction": int(pred),
            "probability": float(proba),
            "message": "Will Subscribe" if pred == 1 else "Will Not Subscribe"
        })

    except Exception as e:
        logger.error(f"API prediction failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    app.run(host="0.0.0.0")
