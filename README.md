# Facial Recognition Attendance System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Swift](https://img.shields.io/badge/Swift-5-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![SwiftUI](https://img.shields.io/badge/SwiftUI-iOS-blue)
![Firebase](https://img.shields.io/badge/Firebase-Firestore-yellow)
![License](https://img.shields.io/badge/License-MIT-purple)

An AI-powered **Facial Recognition Attendance System** that automates attendance using deep learning, computer vision, and cloud database integration.

The system detects faces in real time, generates facial embeddings using **MobileNetV2**, and marks attendance automatically by matching with stored student records.

---

# Overview

Manual attendance systems are:

- Time-consuming  
- Prone to human error  
- Vulnerable to proxy attendance  

This project solves these problems using **AI and facial recognition**.

The system integrates:

- OpenCV for face detection  
- MobileNetV2 for embedding generation  
- FastAPI backend for processing  
- Firebase Firestore for cloud storage  
- SwiftUI iOS frontend for camera capture  

---

# Features

- Real-time face detection
- Automatic attendance marking
- Deep learning facial recognition
- Mobile iOS application
- Cloud-based attendance storage
- Fast and lightweight model
- Secure backend API

---

# System Architecture

- iOS App (SwiftUI)
      ↓
FastAPI Backend (Python)
      ↓
Face Detection (OpenCV)
      ↓
Embedding Generation (MobileNetV2)
      ↓
Embedding Matching (Cosine Similarity)
      ↓
Firebase Firestore (Database)
---

# Dataset Preparation

- 20–30 images per student
- Images are:
  - Cropped
  - Resized
  - Normalized
- Embeddings stored in:

```bash
face_embeddings.pkl
```
# Recognition Workflow

1. Capture image from camera
2. Detect face using OpenCV
3. Generate embedding using MobileNetV2
4. Compare embedding with stored embeddings
5. Calculate cosine similarity
6. If similarity > threshold:
      Mark attendance
   Else:
      Mark as unknown

Tech Stack

Backend
	•	Python
	•	FastAPI
	•	OpenCV
	•	TensorFlow / Keras
	•	NumPy
	•	Uvicorn

Frontend
	•	Swift
	•	SwiftUI
	•	AVFoundation

Database
	•	Firebase Firestore

AI Model
	•	MobileNetV2
	•	Transfer Learning
	•	Cosine Similarity Matching

⸻

# Results
Accuracy  : ~95%
Recognition Time  : < 0.5 sec
Images per person:  20–30
Cloud Sync:  Successful
