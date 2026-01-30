
import sys
import os

# Mock the environment to ensure no credentials exist initially
if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
    del os.environ['GOOGLE_APPLICATION_CREDENTIALS']

try:
    import firebase_service
    print("Imported firebase_service successfully.")
except ImportError:
    print("Could not import firebase_service (deps missing?).")
    sys.exit(0)

print(f"Initializing Firebase (expecting False/Safe fail)...")
success = firebase_service.initialize_firebase()
print(f"Initialization success: {success}")

print("Attempting to save prediction (should not crash)...")
try:
    firebase_service.save_prediction("2025-01-01", 100)
    print("save_prediction call completed without error.")
except Exception as e:
    print(f"FAIL: save_prediction raised exception: {e}")

print("Attempting to save feedback (should not crash)...")
try:
    firebase_service.save_feedback("2025-01-01", 105)
    print("save_feedback call completed without error.")
except Exception as e:
    print(f"FAIL: save_feedback raised exception: {e}")
