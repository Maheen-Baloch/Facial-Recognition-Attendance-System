from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from typing import List, Dict, Optional
import os
import cv2
import numpy as np
from datetime import datetime

from helpers import (
    ensure_dirs,
    DATASET_DIR,
    EMBEDDINGS_FILE,
    load_face_detector,
    load_embedding_model,
    load_embeddings,
    save_embeddings,
    preprocess_face,
    find_best_match,
    mark_attendance,
    get_attendance_by_date_firestore,
    add_student_to_firestore,
    get_all_students_firestore,
)


app = FastAPI(
    title="Facial Recognition Attendance API",
    description="API endpoints for student registration and attendance using face recognition + Firestore.",
    version="1.0.0",
)


# Globals for models and embeddings (lazy-initialized)
FACE_DETECTOR = None
EMBEDDING_MODEL = None
KNOWN_EMBS: Optional[np.ndarray] = None
KNOWN_NAMES: Optional[np.ndarray] = None


def init_models():
    """
    Lazy initialization of face detector, embedding model, and embeddings.
    Mirrors the logic in attendence_system.init_models(), but for API usage.
    """
    global FACE_DETECTOR, EMBEDDING_MODEL, KNOWN_EMBS, KNOWN_NAMES

    ensure_dirs()

    if FACE_DETECTOR is None:
        FACE_DETECTOR = load_face_detector()
    if EMBEDDING_MODEL is None:
        EMBEDDING_MODEL = load_embedding_model()

    KNOWN_EMBS, KNOWN_NAMES = load_embeddings()


def detect_faces(frame: np.ndarray, confidence_threshold: float = 0.5):
    """
    Detect faces in a frame using the DNN face detector.

    Returns a list of (x1, y1, x2, y2) boxes.
    """
    if FACE_DETECTOR is None:
        raise RuntimeError("Face detector is not initialized.")

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
    )
    FACE_DETECTOR.setInput(blob)
    detections = FACE_DETECTOR.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            faces.append((x1, y1, x2, y2))

    return faces


def update_embeddings():
    """
    Regenerate embeddings from the entire dataset directory.
    This mirrors attendence_system.update_embeddings() but without prints to console.
    """
    global KNOWN_EMBS, KNOWN_NAMES

    init_models()

    new_embs = []
    new_names = []

    for person_folder in os.listdir(DATASET_DIR):
        pf = os.path.join(DATASET_DIR, person_folder)
        if not os.path.isdir(pf):
            continue
        for img_name in os.listdir(pf):
            img_path = os.path.join(pf, img_name)
            if not img_path.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(img_path)
            if img is None or img.size == 0:
                continue
            try:
                processed_arr = preprocess_face(img)
                emb = EMBEDDING_MODEL.predict(processed_arr, verbose=0)[0]
                new_embs.append(emb)
                new_names.append(person_folder)
            except Exception:
                continue

    if new_embs:
        new_embs_arr = np.array(new_embs)
    else:
        new_embs_arr = np.array([])

    save_embeddings(new_embs_arr, np.array(new_names))
    KNOWN_EMBS = new_embs_arr
    KNOWN_NAMES = np.array(new_names)


def _validate_date(date_str: str) -> str:
    """Validate date format YYYY-MM-DD or raise HTTPException."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return date_str
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format '{date_str}'. Use YYYY-MM-DD.",
        )


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@app.get("/students")
def students() -> List[Dict[str, str]]:
    """
    Get list of all registered students from Firestore.
    """
    data = get_all_students_firestore()
    # data is list of (reg_no, name)
    return [{"reg_no": reg_no, "name": name} for reg_no, name in data]


@app.get("/attendance")
def attendance(date: Optional[str] = Query(None, description="YYYY-MM-DD; defaults to today")):
    """
    Get attendance for a specific date (or today if not provided).
    """
    day = None
    if date:
        day = _validate_date(date)

    entries = get_attendance_by_date_firestore(day)
    # entries is list of (reg_no, name, time)
    return [
        {"reg_no": reg_no, "name": name, "time": time_str}
        for reg_no, name, time_str in entries
    ]


@app.post("/mark-attendance")
async def mark(file: UploadFile = File(...)):
    """
    Mark attendance using an uploaded face image.
    - Detects the largest face in the image
    - Computes embedding and finds best match
    - If recognized, marks attendance in Firestore
    """
    init_models()

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    faces = detect_faces(img, confidence_threshold=0.5)
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected in the image.")

    # Choose the largest face
    best_face = None
    max_area = 0
    for x1, y1, x2, y2 in faces:
        face = img[y1:y2, x1:x2]
        if face.size == 0:
            continue
        area = face.shape[0] * face.shape[1]
        if area > max_area:
            max_area = area
            best_face = face

    if best_face is None:
        raise HTTPException(
            status_code=400, detail="Could not crop a valid face region from the image."
        )

    # Embed and recognize
    processed_arr = preprocess_face(best_face)
    emb = EMBEDDING_MODEL.predict(processed_arr, verbose=0)[0]
    name, score = find_best_match(emb, KNOWN_EMBS, KNOWN_NAMES, threshold=0.75)

    if name.startswith("Unknown"):
        return {
            "recognized": False,
            "name": None,
            "score": score,
            "message": "Face not recognized.",
        }

    ok, msg = mark_attendance(name)
    return {
        "recognized": ok,
        "name": name,
        "score": score,
        "message": msg,
    }


@app.post("/register-student")
async def register(name: str, reg_no: str, file: UploadFile = File(...)):
    """
    Register a new student using an uploaded face image.
    - Saves cropped face image to dataset
    - Regenerates embeddings
    - Adds student to Firestore

    Params:
    - name: Full name of the student
    - reg_no: Registration / roll number
    - file: Face image (jpg/jpeg/png)
    """
    if not name.strip():
        raise HTTPException(status_code=400, detail="Name cannot be empty.")
    if not reg_no.strip():
        raise HTTPException(status_code=400, detail="Registration number cannot be empty.")

    init_models()
    ensure_dirs()

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    faces = detect_faces(img, confidence_threshold=0.5)
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected in the image.")

    # Pick the largest face
    best_face = None
    max_area = 0
    for x1, y1, x2, y2 in faces:
        face = img[y1:y2, x1:x2]
        if face.size == 0:
            continue
        area = face.shape[0] * face.shape[1]
        if area > max_area:
            max_area = area
            best_face = face

    if best_face is None:
        raise HTTPException(
            status_code=400, detail="Could not crop a valid face region from the image."
        )

    # Save face to dataset
    folder_name = f"{name.replace(' ', '_')}_{reg_no}"
    save_path = os.path.join(DATASET_DIR, folder_name)
    os.makedirs(save_path, exist_ok=True)

    img_name = f"{folder_name}_api_0.jpg"
    img_path = os.path.join(save_path, img_name)
    cv2.imwrite(img_path, best_face)

    # Regenerate embeddings
    update_embeddings()

    # Add student to Firestore
    try:
        add_student_to_firestore(name, reg_no)
    except Exception as e:
        # Don't fail registration if Firestore write fails; just report error
        return {
            "registered": True,
            "name": name,
            "reg_no": reg_no,
            "dataset_image": img_path,
            "firestore_error": str(e),
        }

    return {
        "registered": True,
        "name": name,
        "reg_no": reg_no,
        "dataset_image": img_path,
    }


