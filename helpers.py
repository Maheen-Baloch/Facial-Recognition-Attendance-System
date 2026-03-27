import os
import cv2
import pickle
import numpy as np
from datetime import date, datetime

import firebase_admin
from firebase_admin import credentials, firestore

from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "face_embeddings.pkl")

ATTENDANCE_DB = "firestore"
FACE_DETECTOR_PROTO = os.path.join(BASE_DIR, "deploy.prototxt")
FACE_DETECTOR_MODEL = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
FACENET_INPUT_SIZE = (160, 160)


_firestore_client = None


class FaceNetEmbedder:
    """Adapter to keep a Keras-like predict() API used across the project."""

    def __init__(self):
        self._model = FaceNet()

    def predict(self, batch, verbose=0):
        """
        batch: numpy array of shape (N, H, W, 3) — uint8 RGB.
        Returns: numpy array of shape (N, 512) — L2-normalised embeddings.
        keras-facenet handles resizing and prewhitening internally.
        L2-normalisation makes cosine similarity == dot product for stable thresholding.
        """
        if batch is None or len(batch) == 0:
            return np.array([])
        images = [batch[i] for i in range(len(batch))]
        embs = self._model.embeddings(images)
        embs = np.asarray(embs, dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        return embs / norms


def init_firestore(service_account_path=None):
    """
    Initializes Firestore client.
    Priority:
      1. FIREBASE_KEY_JSON env var (Hugging Face Secrets / any cloud deployment)
      2. serviceAccountKey.json file on disk (local development)
    """
    global _firestore_client
    if _firestore_client is not None:
        return _firestore_client

    import json as _json

    key_json = os.environ.get("FIREBASE_KEY_JSON")
    if key_json:
        # Cloud / HF Spaces: secret injected as environment variable
        key_dict = _json.loads(key_json)
        cred = credentials.Certificate(key_dict)
    else:
        # Local dev: read from file
        if service_account_path is None:
            service_account_path = os.path.join(BASE_DIR, 'serviceAccountKey.json')
        if not os.path.exists(service_account_path):
            raise FileNotFoundError(
                "Firestore credentials not found. "
                "Set FIREBASE_KEY_JSON env var or place serviceAccountKey.json "
                f"at: {service_account_path}"
            )
        cred = credentials.Certificate(service_account_path)

    try:
        firebase_admin.initialize_app(cred)
    except ValueError:
        pass  # Already initialized

    _firestore_client = firestore.client()
    return _firestore_client


def ensure_dirs():
    os.makedirs(DATASET_DIR, exist_ok=True)


def load_face_detector(proto_path=FACE_DETECTOR_PROTO, model_path=FACE_DETECTOR_MODEL):
    if not os.path.exists(proto_path) or not os.path.exists(model_path):
        raise FileNotFoundError(
            "Face detector model files missing. "
            "Add deploy.prototxt and res10_300x300_ssd_iter_140000.caffemodel."
        )
    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    return net


def load_embedding_model():
    return FaceNetEmbedder()


def preprocess_face(face_image):
    """
    Resize and convert face crop to RGB uint8 for keras-facenet.
    keras-facenet does its own internal standardization (prewhitening),
    so we must NOT apply it here.
    """
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, FACENET_INPUT_SIZE)
    arr = np.expand_dims(face_resized, axis=0)  # (1, 160, 160, 3)
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


def find_best_match(
    query_embedding,
    known_embs,
    known_names,
    threshold=0.65,
    margin=0.05,
):
    """
    Match a query face embedding against the database.
    Returns (name, score). Name starts with "Unknown" if no match found.
    Both embeddings must be L2-normalised (FaceNetEmbedder does this automatically).
    Same person  → cosine sim ~0.65–0.99
    Diff person  → cosine sim ~0.10–0.50
    """
    if len(known_embs) == 0:
        return "Unknown (empty database)", 0.0

    q = query_embedding.reshape(1, -1).astype(np.float32)
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm

    similarities = cosine_similarity(q, known_embs.astype(np.float32))[0]

    unique_names = np.unique(known_names)
    identity_scores = []
    for person in unique_names:
        mask = known_names == person
        person_sims = similarities[mask]
        top_k = min(3, len(person_sims))
        top_mean = float(np.mean(np.sort(person_sims)[-top_k:]))
        identity_scores.append((person, top_mean))

    identity_scores.sort(key=lambda x: x[1], reverse=True)
    best_name, best_score = identity_scores[0]

    print(f"  Best match: {best_name}  score={best_score:.3f}")

    if len(identity_scores) == 1:
        return (best_name, best_score) if best_score >= threshold else ("Unknown", best_score)

    second_score = identity_scores[1][1]
    if best_score >= threshold and (best_score - second_score) >= margin:
        return best_name, best_score

    return "Unknown", best_score


def parse_folder_name(folder_name):
    """Parse 'First_Last_RegNo' → ('First Last', 'RegNo')."""
    parts = folder_name.rsplit('_', 1)
    if len(parts) == 2:
        return parts[0].replace('_', ' '), parts[1]
    return folder_name, folder_name


# ---------------------------------------------------------------------------
# Firestore helpers
# ---------------------------------------------------------------------------

def _ensure_firestore():
    global _firestore_client
    if _firestore_client is None:
        init_firestore()
    return _firestore_client


def add_student_to_firestore(name, reg_no):
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
    db = _ensure_firestore()
    student_name, reg_no = parse_folder_name(folder_name)
    today = date.today().isoformat()
    att_doc_ref = db.collection('attendance').document(today)

    student_doc = db.collection('students').document(reg_no)
    if not student_doc.get().exists:
        student_doc.set({
            'name': student_name,
            'reg_no': reg_no,
            'created_at': datetime.utcnow().isoformat()
        })

    att_snapshot = att_doc_ref.get()
    att_data = att_snapshot.to_dict() or {}
    students_map = att_data.get('students', {})
    current_time = datetime.now().strftime('%H:%M:%S')

    if reg_no in students_map and students_map[reg_no].get('status') == 'present':
        return False, f"⚠️  {student_name} ({reg_no}) already marked present today."

    students_map[reg_no] = {'name': student_name, 'status': 'present', 'time': current_time}
    att_doc_ref.set({'students': students_map}, merge=True)
    return True, f"✅  {student_name} ({reg_no}) marked present at {current_time}."


def get_attendance_by_date_firestore(day=None):
    db = _ensure_firestore()
    if day is None:
        day = date.today().isoformat()
    att_doc_ref = db.collection('attendance').document(day)
    att_snapshot = att_doc_ref.get()
    if not att_snapshot.exists:
        return []
    att_data = att_snapshot.to_dict() or {}
    students_map = att_data.get('students', {})
    result = [
        (reg_no, info.get('name'), info.get('time'))
        for reg_no, info in students_map.items()
        if info.get('status') == 'present'
    ]
    result.sort(key=lambda x: x[0])
    return result


def get_all_students_firestore():
    db = _ensure_firestore()
    students_iter = db.collection('students').stream()
    result = [
        (d.get('reg_no', s.id), d.get('name', ''))
        for s in students_iter
        for d in [s.to_dict()]
    ]
    result.sort(key=lambda x: x[0])
    return result


def mark_attendance(folder_name):
    try:
        return mark_attendance_firestore(folder_name)
    except Exception as e:
        return False, f"Error marking attendance: {e}"
