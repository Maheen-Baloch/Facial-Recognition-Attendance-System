from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List, Dict, Optional
import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

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
    description="Register students and mark attendance via face recognition backed by Firestore.",
    version="1.1.0",
)

# Allow all origins (works with ngrok)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend from static/ folder beside app.py
_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(_STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

@app.get("/", include_in_schema=False)
def serve_frontend():
    index = os.path.join(_STATIC_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "Place index.html in the static/ folder."}

# Globals — lazy-initialized on first request
FACE_DETECTOR = None
EMBEDDING_MODEL = None
KNOWN_EMBS: Optional[np.ndarray] = None
KNOWN_NAMES: Optional[np.ndarray] = None


def init_models():
    global FACE_DETECTOR, EMBEDDING_MODEL, KNOWN_EMBS, KNOWN_NAMES
    ensure_dirs()
    if FACE_DETECTOR is None:
        FACE_DETECTOR = load_face_detector()
    if EMBEDDING_MODEL is None:
        EMBEDDING_MODEL = load_embedding_model()
    KNOWN_EMBS, KNOWN_NAMES = load_embeddings()


def _detect_faces(frame: np.ndarray, confidence_threshold: float = 0.5):
    if FACE_DETECTOR is None:
        raise RuntimeError("Face detector not initialized.")
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    FACE_DETECTOR.setInput(blob)
    detections = FACE_DETECTOR.forward()
    faces = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            faces.append((max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)))
    return faces


def _crop_largest_face(img: np.ndarray) -> np.ndarray:
    faces = _detect_faces(img, confidence_threshold=0.5)
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected in the image.")
    best, max_area = None, 0
    for x1, y1, x2, y2 in faces:
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        area = crop.shape[0] * crop.shape[1]
        if area > max_area:
            max_area = area
            best = crop
    if best is None:
        raise HTTPException(status_code=400, detail="Could not crop a valid face region.")
    return best


def _next_image_path(save_path: str, folder_name: str) -> str:
    existing = []
    for f in os.listdir(save_path):
        stem = Path(f).stem
        prefix = f"{folder_name}_api_"
        if stem.startswith(prefix) and stem[len(prefix):].isdigit():
            existing.append(int(stem[len(prefix):]))
    idx = (max(existing) + 1) if existing else 0
    return os.path.join(save_path, f"{folder_name}_api_{idx}.jpg")


def _rebuild_embeddings():
    global KNOWN_EMBS, KNOWN_NAMES
    init_models()
    new_embs, new_names = [], []
    for person_folder in os.listdir(DATASET_DIR):
        pf = os.path.join(DATASET_DIR, person_folder)
        if not os.path.isdir(pf):
            continue
        for img_name in os.listdir(pf):
            img_path = os.path.join(pf, img_name)
            if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img = cv2.imread(img_path)
            if img is None or img.size == 0:
                continue
            try:
                emb = EMBEDDING_MODEL.predict(preprocess_face(img), verbose=0)[0]
                new_embs.append(emb)
                new_names.append(person_folder)
            except Exception:
                continue
    KNOWN_EMBS = np.array(new_embs) if new_embs else np.array([])
    KNOWN_NAMES = np.array(new_names)
    save_embeddings(KNOWN_EMBS, KNOWN_NAMES)


def _validate_date(date_str: str) -> str:
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return date_str
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format '{date_str}'. Use YYYY-MM-DD.",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/students")
def get_students() -> List[Dict[str, str]]:
    return [{"reg_no": r, "name": n} for r, n in get_all_students_firestore()]


@app.get("/attendance")
def get_attendance(
    date: Optional[str] = Query(None, description="YYYY-MM-DD; defaults to today")
):
    day = _validate_date(date) if date else None
    return [
        {"reg_no": r, "name": n, "time": t}
        for r, n, t in get_attendance_by_date_firestore(day)
    ]


@app.post("/mark-attendance")
async def mark(file: UploadFile = File(...)):
    init_models()
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    faces = _detect_faces(img)
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected in the image.")
    best, max_area = None, 0
    for x1, y1, x2, y2 in faces:
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        area = crop.shape[0] * crop.shape[1]
        if area > max_area:
            max_area = area
            best = crop
    if best is None:
        raise HTTPException(status_code=400, detail="Could not crop a valid face region.")
    emb = EMBEDDING_MODEL.predict(preprocess_face(best), verbose=0)[0]
    name, score = find_best_match(emb, KNOWN_EMBS, KNOWN_NAMES)
    if name.startswith("Unknown"):
        return {"recognized": False, "name": None, "score": round(score, 4),
                "message": "Face not recognized."}
    ok, msg = mark_attendance(name)
    return {"recognized": True, "name": name, "score": round(score, 4), "message": msg}


@app.post("/register-student")
async def register_student(name: str, reg_no: str, file: UploadFile = File(...)):
    if not name.strip():
        raise HTTPException(status_code=400, detail="Name cannot be empty.")
    if not reg_no.strip():
        raise HTTPException(status_code=400, detail="Registration number cannot be empty.")
    init_models()
    ensure_dirs()
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    best_face = _crop_largest_face(img)
    folder_name = f"{name.replace(' ', '_')}_{reg_no}"
    save_path = os.path.join(DATASET_DIR, folder_name)
    os.makedirs(save_path, exist_ok=True)
    img_path = _next_image_path(save_path, folder_name)
    cv2.imwrite(img_path, best_face)
    _rebuild_embeddings()
    firestore_error = None
    try:
        add_student_to_firestore(name, reg_no)
    except Exception as e:
        firestore_error = str(e)
    return {
        "registered": True,
        "name": name,
        "reg_no": reg_no,
        "dataset_image": img_path,
        **({"firestore_error": firestore_error} if firestore_error else {}),
    }


@app.post("/register-student-batch")
async def register_student_batch(
    name: str, reg_no: str, files: List[UploadFile] = File(...)
):
    if not name.strip():
        raise HTTPException(status_code=400, detail="Name cannot be empty.")
    if not reg_no.strip():
        raise HTTPException(status_code=400, detail="Registration number cannot be empty.")
    if not files:
        raise HTTPException(status_code=400, detail="At least one image file is required.")
    init_models()
    ensure_dirs()
    folder_name = f"{name.replace(' ', '_')}_{reg_no}"
    save_path = os.path.join(DATASET_DIR, folder_name)
    os.makedirs(save_path, exist_ok=True)
    saved_paths, failed_files = [], []
    for file in files:
        contents = await file.read()
        if not contents:
            failed_files.append({"filename": file.filename, "reason": "empty file"})
            continue
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            failed_files.append({"filename": file.filename, "reason": "decode failed"})
            continue
        try:
            best_face = _crop_largest_face(img)
        except HTTPException as e:
            failed_files.append({"filename": file.filename, "reason": e.detail})
            continue
        img_path = _next_image_path(save_path, folder_name)
        if cv2.imwrite(img_path, best_face):
            saved_paths.append(img_path)
        else:
            failed_files.append({"filename": file.filename, "reason": "write failed"})
    if not saved_paths:
        raise HTTPException(status_code=400, detail="No valid images could be processed.")
    _rebuild_embeddings()
    firestore_error = None
    try:
        add_student_to_firestore(name, reg_no)
    except Exception as e:
        firestore_error = str(e)
    return {
        "registered": True,
        "name": name,
        "reg_no": reg_no,
        "saved_count": len(saved_paths),
        "saved_images": saved_paths,
        "failed_files": failed_files,
        **({"firestore_error": firestore_error} if firestore_error else {}),
    }
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
