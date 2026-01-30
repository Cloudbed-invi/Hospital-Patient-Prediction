import os
import datetime
import logging

# Configure logging for Firebase service
logger = logging.getLogger('firebase_service')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Global State for Firebase App
_db = None
_is_initialized = False

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    _HAS_FIREBASE_SDK = True
except ImportError:
    logger.warning("Firebase Admin SDK not installed. Firebase persistence will be disabled.")
    _HAS_FIREBASE_SDK = False

def initialize_firebase():
    """
    Initializes Firebase Admin SDK.
    Safe to call multiple times.
    """
    global _db, _is_initialized
    
    if not _HAS_FIREBASE_SDK:
        return False

    if _is_initialized:
        return True

    try:
        # Check for credentials in environment variables or default location
        # For this implementation, we look for GOOGLE_APPLICATION_CREDENTIALS
        # or a specific 'firebase_credentials.json' in the root if not set.
        cred_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 'firebase_credentials.json')
        
        if os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            # Check if app is already initialized to avoid error
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred)
            _db = firestore.client()
            _is_initialized = True
            logger.info("Firebase initialized successfully.")
            return True
        else:
            logger.warning(f"Firebase credentials not found at {cred_path}. Persistence disabled.")
            return False

    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {e}")
        return False

def save_prediction(date_str, prediction, cluster_label=None, alert_level=None):
    """
    Saves a prediction to the 'predictions' collection.
    Non-blocking / Fail-safe.
    """
    if not _is_initialized:
        return

    try:
        doc_ref = _db.collection('predictions').document(date_str)
        data = {
            'date': date_str,
            'predicted_value': prediction,
            'cluster_label': cluster_label if cluster_label is not None else "Unknown",
            'alert_level': alert_level if alert_level is not None else "Unknown",
            'created_at': firestore.SERVER_TIMESTAMP
        }
        doc_ref.set(data, merge=True)
        logger.info(f"Saved prediction for {date_str} to Firebase.")
    except Exception as e:
        logger.error(f"Error saving prediction to Firebase: {e}")

def save_feedback(date_str, actual_value, submitted_at=None):
    """
    Saves user feedback (corrected actuals) to the 'feedback' collection.
    Non-blocking / Fail-safe.
    """
    if not _is_initialized:
        return

    try:
        doc_ref = _db.collection('feedback').document(date_str)
        data = {
            'date': date_str,
            'user_entered_actual': actual_value,
            'submitted_at': submitted_at if submitted_at else firestore.SERVER_TIMESTAMP
        }
        doc_ref.set(data, merge=True)
        logger.info(f"Saved feedback for {date_str} to Firebase.")
    except Exception as e:
        logger.error(f"Error saving feedback to Firebase: {e}")
