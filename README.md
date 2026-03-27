---
title: FaceAttend
emoji: 📸
colorFrom: green
colorTo: purple
sdk: docker
pinned: false
---

# Facial Recognition Attendance System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![FaceNet](https://img.shields.io/badge/FaceNet-Embeddings-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Detection-red)
![Firebase](https://img.shields.io/badge/Firebase-Firestore-yellow)
![License](https://img.shields.io/badge/License-MIT-purple)

An AI-powered attendance system that automates student attendance using deep learning and computer vision. Faces are detected in uploaded images, matched against registered students using facial embeddings, and attendance is recorded automatically in Firebase Firestore.

---

## Overview

Manual attendance systems are time-consuming, error-prone, and vulnerable to proxy attendance. This project solves these problems using facial recognition powered by FaceNet and a web-based interface accessible from any device.

---

## Features

- Face detection using OpenCV DNN (SSD ResNet)
- Facial embedding generation using FaceNet (512-d, L2-normalised)
- Cosine similarity matching with configurable threshold
- Student registration with single or batch photo upload
- Live webcam capture for both registration and attendance
- Attendance records stored per-date in Firebase Firestore
- CSV export of attendance records
- Web frontend — works on any browser, no app install needed
- REST API built with FastAPI
- Docker-ready for cloud deployment (Hugging Face Spaces)

---

## System Architecture

```
Browser (HTML / CSS / JavaScript)
        ↓
FastAPI Backend (Python)
        ↓
Face Detection — OpenCV DNN (SSD ResNet10)
        ↓
Embedding Generation — FaceNet (keras-facenet)
        ↓
Cosine Similarity Matching
        ↓
Firebase Firestore (Cloud Database)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, Uvicorn |
| Face Detection | OpenCV DNN — SSD ResNet10 |
| Face Embeddings | FaceNet via keras-facenet (512-d) |
| Similarity Matching | Cosine Similarity (scikit-learn) |
| Frontend | HTML, CSS, JavaScript |
| Database | Firebase Firestore |
| Deployment | Docker, Hugging Face Spaces |

---

## Project Structure

```
├── app.py                          # FastAPI application & endpoints
├── helpers.py                      # Models, embeddings, Firestore logic
├── Dockerfile                      # Container config for HF Spaces
├── requirements.txt
├── deploy.prototxt                 # Face detector config
├── res10_300x300_ssd_iter_140000.caffemodel  # Face detector weights
└── static/
    └── index.html                  # Web frontend
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serves the web frontend |
| GET | `/students` | List all registered students |
| GET | `/attendance?date=YYYY-MM-DD` | Get attendance for a date |
| POST | `/mark-attendance` | Mark attendance from uploaded image |
| POST | `/register-student` | Register student with single image |
| POST | `/register-student-batch` | Register student with multiple images |

---

## Recognition Workflow

1. Upload or capture a face image
2. Detect face region using OpenCV SSD detector
3. Crop and resize face to 160×160
4. Generate 512-d embedding using FaceNet
5. L2-normalise the embedding vector
6. Compute cosine similarity against all stored embeddings
7. If best match score ≥ 0.65 and ahead of runner-up by ≥ 0.05 → mark present
8. Record attendance with timestamp in Firestore

---

## Local Setup

```bash
# Clone the repo
git clone https://github.com/Maheen-Baloch/Facial-Recognition-Attendance-System
cd Facial-Recognition-Attendance-System

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add your Firebase service account key
# Place serviceAccountKey.json in the project root (never commit this)

# Run the server
uvicorn app:app --reload --port 8000
```

Open `http://localhost:8000` in your browser.

---

## Deployment (Hugging Face Spaces)

1. Create a new Space with **Docker** SDK
2. Add your Firebase credentials as a Secret:
   - Name: `FIREBASE_KEY_JSON`
   - Value: contents of `serviceAccountKey.json`
3. Push all files except `serviceAccountKey.json`
4. Upload `.caffemodel` using Git LFS

---

## Results

| Metric | Value |
|---|---|
| Recognition threshold | 0.65 cosine similarity |
| Embedding dimensions | 512-d (L2-normalised) |
| Face detector | SSD ResNet10 @ 300×300 |
| Recommended images per student | 10+ |
| Database | Firebase Firestore (real-time) |

---

## Author

**Maheen Baloch**
BS Artificial Intelligence — COMSATS University Islamabad
[GitHub](https://github.com/Maheen-Baloch) · [LinkedIn](https://linkedin.com/in/maheenbaloch133)
