# Enhanced src/utils.py
import numpy as np 
import pandas as pd
import joblib
import logging
import os
from typing import Tuple, Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

# Model configuration
MODEL_PATH = "models/final_model_lightgbm_optimized.joblib"  # Update with your best model path
BACKUP_MODEL_PATH = "models/xgb_pipeline.pkl"  # Fallback model

def load_model():
    """Load the trained model with fallback options"""
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded from {MODEL_PATH}")
            return model
        elif os.path.exists(BACKUP_MODEL_PATH):
            model = joblib.load(BACKUP_MODEL_PATH)
            logger.info(f"Backup model loaded from {BACKUP_MODEL_PATH}")
            return model
        else:
            raise FileNotFoundError("No model file found")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

# Load model once at startup
try:
    model = load_model()
except Exception as e:
    logger.error(f"Critical: Could not load model: {e}")
    model = None

def validate_input_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean input data"""
    # Set default values for missing fields
    defaults = {
        "balance": 0,
        "duration": 0,
        "campaign": 1,
        "previous": 0,
        "pdays": -1,
        "euribor3m": 1.0,
        "cons.conf.idx": -40.0,
        "emp.var.rate": 1.1,
        "nr.employed": 5000,
        "default": "no",
        "housing": "no", 
        "loan": "no",
        "contact": "cellular",
        "poutcome": "nonexistent"
    }
    
    # Apply defaults for missing values
    for key, default_value in defaults.items():
        if key not in data or data[key] == '' or data[key] is None:
            data[key] = default_value
    
    # Validate ranges
    validations = {
        "age": (18, 100, "Age must be between 18 and 100"),
        "balance": (-10000, 100000, "Balance seems unrealistic"),
        "duration": (0, 5000, "Duration must be between 0 and 5000 seconds"),
        "campaign": (1, 50, "Campaign contacts must be between 1 and 50"),
        "previous": (0, 20, "Previous contacts must be between 0 and 20"),
        "pdays": (-1, 999, "Pdays must be -1 or between 0 and 999")
    }
    
    for field, (min_val, max_val, error_msg) in validations.items():
        if field in data:
            try:
                value = float(data[field])
                if not (min_val <= value <= max_val):
                    logger.warning(f"Value out of range: {field}={value}. {error_msg}")
            except (ValueError, TypeError):
                logger.warning(f"Invalid value for {field}: {data[field]}")
    
    return data

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced feature engineering with error handling"""
    try:
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Duration transformations
        df["duration_log"] = np.log1p(df["duration"].clip(0))
        df["duration_sq"] = df["duration"] ** 2
        
        # Campaign binning with more granular categories
        df["campaign_bin"] = pd.cut(
            df["campaign"].clip(1, 20),
            bins=[0, 1, 2, 4, 8, 999],
            labels=["1", "2", "3-4", "5-8", "9+"]
        )
        
        # Enhanced pdays features
        df["pdays_recent"] = df["pdays"].replace(-1, np.nan)
        df["pdays_not_contacted"] = (df["pdays"] == -1).astype(int)
        df["pdays_very_recent"] = ((df["pdays"] > 0) & (df["pdays"] <= 7)).astype(int)
        
        # Previous campaign success rate approximation
        df["prev_success"] = (df["poutcome"] == "success").astype(int)
        
        # Month cyclical encoding
        month_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }
        df["month_num"] = df["month"].map(month_map).fillna(6)  # Default to June
        df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
        
        # Day of week cyclical encoding
        dow_map = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4}
        df["day_of_week_num"] = df["day_of_week"].map(dow_map).fillna(2)  # Default to Wed
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week_num"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week_num"] / 7)
        
        # Enhanced job grouping
        job_mapping = {
            "admin.": "white_collar", "management": "white_collar", "technician": "skilled",
            "services": "skilled", "blue-collar": "manual", "housemaid": "manual",
            "retired": "retired", "student": "student", "unemployed": "unemployed",
            "entrepreneur": "self_employed", "self-employed": "self_employed", 
            "unknown": "unknown"
        }
        df["job_group"] = df["job"].map(job_mapping).fillna("other")
        
        # Education grouping
        education_mapping = {
            "primary": "basic", "secondary": "secondary", 
            "tertiary": "tertiary", "unknown": "unknown"
        }
        df["education_group"] = df["education"].map(education_mapping).fillna("unknown")
        
        # Age binning
        df["age_group"] = pd.cut(df["age"], 
                               bins=[0, 25, 35, 50, 65, 100], 
                               labels=["young", "adult", "middle", "senior", "elderly"])
        
        # Balance features
        df["balance_positive"] = (df["balance"] > 0).astype(int)
        df["balance_log"] = np.sign(df["balance"]) * np.log1p(np.abs(df["balance"]))
        
        # Interaction features
        df["duration_balance"] = df["duration"] * df["balance_positive"]
        df["age_job"] = df["age"].astype(str) + "_" + df["job_group"]
        
        # Economic indicators normalization
        df["euribor3m_norm"] = (df["euribor3m"] - 1.0) / 4.0  # Rough normalization
        df["emp_var_rate_norm"] = (df["emp.var.rate"] + 2.0) / 4.0  # Rough normalization
        
        logger.info("Feature engineering completed successfully")
        return df
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
        # Return original dataframe if feature engineering fails
        return df

def predict_subscription(input_data: Dict[str, Any]) -> Tuple[int, float]:
    """
    Enhanced prediction function with comprehensive error handling
    
    Args:
        input_data: Dictionary containing customer information
        
    Returns:
        Tuple of (prediction, probability)
    """
    try:
        if model is None:
            raise ValueError("Model not loaded")
        
        # Validate and clean input data
        cleaned_data = validate_input_data(input_data)
        
        # Create DataFrame
        df = pd.DataFrame([cleaned_data])
        
        # Apply feature engineering
        df_processed = feature_engineering(df)
        
        # Make predictions
        proba = model.predict_proba(df_processed)[0, 1]
        pred = model.predict(df_processed)[0]
        
        # Clip probability to reasonable range
        proba = np.clip(proba, 0.001, 0.999)
        
        logger.info(f"Prediction successful: pred={pred}, proba={proba:.4f}")
        
        return int(pred), float(proba)
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        # Return conservative prediction in case of error
        return 0, 0.1

def get_model_info() -> Dict[str, Any]:
    """Get information about the loaded model"""
    try:
        if model is None:
            return {"status": "error", "message": "Model not loaded"}
        
        return {
            "status": "loaded",
            "model_type": type(model).__name__,
            "model_path": MODEL_PATH if os.path.exists(MODEL_PATH) else BACKUP_MODEL_PATH,
            "features": getattr(model, 'feature_names_in_', 'Unknown')
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Export key functions
__all__ = ['predict_subscription', 'get_model_info']
