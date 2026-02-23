"""
Configuration file for Attendance System
All configurable parameters are defined here.
"""

# Face Detection & Recognition Settings
FACE_DETECTION_CONFIDENCE = 0.5  # Minimum confidence for face detection
RECOGNITION_THRESHOLD = 0.75  # Minimum similarity score for recognition
FACE_DETECTOR_PROTO = "deploy.prototxt"
FACE_DETECTOR_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"

# Attendance Settings
COOLDOWN_SECONDS = 30  # Seconds between attendance marks for same person
MIN_IMAGES_FOR_REGISTRATION = 3  # Minimum images required
RECOMMENDED_IMAGES = 10  # Recommended number of images

# File Paths
DATASET_DIR = "dataset"
EMBEDDINGS_FILE = "face_embeddings.pkl"
SERVICE_ACCOUNT_KEY = "serviceAccountKey.json"

# Model Settings
EMBEDDING_MODEL_INPUT_SIZE = (224, 224)  # Input size for MobileNetV2
FACE_DETECTOR_INPUT_SIZE = (300, 300)  # Input size for face detector

# Display Settings
DISPLAY_FONT_SCALE = 0.8
DISPLAY_THICKNESS = 2

