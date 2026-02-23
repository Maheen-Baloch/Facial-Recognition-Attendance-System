
import os
import cv2
import pickle
import numpy as np
from datetime import date, datetime

import firebase_admin
from firebase_admin import credentials, firestore

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity


DATASET_DIR = "dataset"
EMBEDDINGS_FILE = "face_embeddings.pkl"

ATTENDANCE_DB = "firestore"
FACE_DETECTOR_PROTO = "deploy.prototxt"
FACE_DETECTOR_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"


_firestore_client = None


def init_firestore(service_account_path='serviceAccountKey.json'):
    """Initializes Firestore client using a service account JSON.
    Call this once at program start (helpers will lazily initialize if needed).
    """
    global _firestore_client
    if _firestore_client is not None:
        return _firestore_client

    if not os.path.exists(service_account_path):
        raise FileNotFoundError(f"Firestore service account file not found: {service_account_path}")

    cred = credentials.Certificate(service_account_path)
    try:
        firebase_admin.initialize_app(cred)
    except ValueError:
        
        pass

    _firestore_client = firestore.client()
    return _firestore_client


def ensure_dirs():
    os.makedirs(DATASET_DIR, exist_ok=True)



def load_face_detector(proto_path=FACE_DETECTOR_PROTO, model_path=FACE_DETECTOR_MODEL):
    if not os.path.exists(proto_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Face detector Caffe model files missing. Please add deploy.prototxt and the .caffemodel file.")
    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    return net


def load_embedding_model():
    
    base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    model = Model(inputs=base_model.input, outputs=base_model.output)
    return model


def preprocess_face(face_image):
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (224, 224))
    arr = img_to_array(face_resized)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


def save_embeddings(embeddings, names, path=EMBEDDINGS_FILE):
    data = {'embeddings': embeddings, 'names': names}
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved {len(names)} embeddings to {path}")


def load_embeddings(path=EMBEDDINGS_FILE):
    if not os.path.exists(path):
        print(f"Warning: Embeddings file '{path}' not found. Database is empty.")
        return np.array([]), np.array([])
    with open(path, 'rb') as f:
        data = pickle.load(f)
    embeddings = np.array(data.get('embeddings', []), dtype=np.float32)
    names = np.array(data.get('names', []), dtype=str)
    print(f"Loaded {len(names)} embeddings from database.")
    return embeddings, names



def find_best_match(query_embedding, known_embs, known_names, threshold=0.70):
    if len(known_embs) == 0:
        return "Unknown (DB Empty)", 0.0

    query_embedding = query_embedding.reshape(1, -1)
    similarities = cosine_similarity(query_embedding, known_embs)
    best_match_index = np.argmax(similarities)
    best_score = similarities[0, best_match_index]
    best_name = known_names[best_match_index]
    print(f"🔍 Top match: {best_name} (Score: {best_score:.3f}).")

    if best_score >= threshold:
        return best_name, float(best_score)
    else:
        return "Unknown", float(best_score)



def _ensure_firestore():
    global _firestore_client
    if _firestore_client is None:
        init_firestore()
    return _firestore_client


def add_student_to_firestore(name, reg_no):
    """Adds a new student document to `students` collection and marks them absent for today in `attendance` collection."""
    db = _ensure_firestore()
    doc_ref = db.collection('students').document(reg_no)

    doc_ref.set({
        'name': name,
        'reg_no': reg_no,
        'created_at': datetime.utcnow().isoformat()
    })

    today = date.today().isoformat()
    att_doc_ref = db.collection('attendance').document(today)
    att = att_doc_ref.get().to_dict() or {}
    students_map = att.get('students', {})

    if reg_no not in students_map:
        students_map[reg_no] = {'name': name, 'status': 'absent', 'time': None}
        att_doc_ref.set({'students': students_map}, merge=True)

    print(f"Added student {name} ({reg_no}) to Firestore.")
    return True


def mark_attendance_firestore(folder_name):
    """
    Accepts the folder_name format produced by registration: e.g. 'Full_Name_12345'
    Parses reg_no and name, then marks that student's status as 'present' for today's date in attendance collection.
    Returns (ok:bool, message:str)
    """
    db = _ensure_firestore()

    if '_' in folder_name:

        student_name, reg_no = folder_name.rsplit('_', 1)
    else:
        
        student_name = folder_name
        reg_no = folder_name

    today = date.today().isoformat()
    att_doc_ref = db.collection('attendance').document(today)

    student_doc = db.collection('students').document(reg_no)
    if not student_doc.get().exists:

        student_doc.set({'name': student_name, 'reg_no': reg_no, 'created_at': datetime.utcnow().isoformat()})

    att_snapshot = att_doc_ref.get()
    att_data = att_snapshot.to_dict() or {}
    students_map = att_data.get('students', {})

    current_time = datetime.now().strftime('%H:%M:%S')

    if reg_no in students_map and students_map[reg_no].get('status') == 'present':
        return False, f"⚠️ {student_name} ({reg_no}) already marked present today."

    students_map[reg_no] = {'name': student_name, 'status': 'present', 'time': current_time}
    att_doc_ref.set({'students': students_map}, merge=True)

    return True, f"✅ {student_name} ({reg_no}) marked present at {current_time}."


def get_attendance_by_date_firestore(day=None):
    """Returns a list of (reg_no, name, time) for students marked present on the given day (YYYY-MM-DD)."""
    db = _ensure_firestore()
    if day is None:
        day = date.today().isoformat()
    att_doc_ref = db.collection('attendance').document(day)
    att_snapshot = att_doc_ref.get()
    if not att_snapshot.exists:
        return []
    att_data = att_snapshot.to_dict() or {}
    students_map = att_data.get('students', {})
    result = []
    for reg_no, info in students_map.items():
        if info.get('status') == 'present':
            result.append((reg_no, info.get('name'), info.get('time')))
    
    result.sort(key=lambda x: x[0])
    return result


def get_all_students_firestore():
    """Returns a list of (reg_no, name) of all registered students."""
    db = _ensure_firestore()
    students = db.collection('students').stream()
    result = []
    for s in students:
        d = s.to_dict()
        result.append((d.get('reg_no', s.id), d.get('name', '')))
    result.sort(key=lambda x: x[0])
    return result


def mark_attendance(folder_name):
    """Wrapper to match the original main.py call signature. Uses Firestore underneath."""
    try:
        return mark_attendance_firestore(folder_name)
    except Exception as e:
        return False, f"Error marking attendance: {e}"
