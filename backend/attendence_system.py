

import os
import cv2
import numpy as np
from datetime import datetime, date
from helpers import (
    ensure_dirs, DATASET_DIR, EMBEDDINGS_FILE, ATTENDANCE_DB,
    load_face_detector, load_embedding_model,
    load_embeddings, save_embeddings,
    preprocess_face, find_best_match,
    mark_attendance, get_attendance_by_date_firestore, add_student_to_firestore, get_all_students_firestore
)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



ensure_dirs()
FACE_DETECTOR = None
EMBEDDING_MODEL = None
KNOWN_EMBS = None
KNOWN_NAMES = None


def init_models():
    global FACE_DETECTOR, EMBEDDING_MODEL, KNOWN_EMBS, KNOWN_NAMES
    try:
        if FACE_DETECTOR is None:
            FACE_DETECTOR = load_face_detector()
        if EMBEDDING_MODEL is None:
            EMBEDDING_MODEL = load_embedding_model()
        KNOWN_EMBS, KNOWN_NAMES = load_embeddings()
        print(f"System ready: Loaded {len(KNOWN_NAMES)} known faces.")
    except Exception as e:
        print(f"FATAL ERROR during initialization: {e}")
        print("Please ensure you have the required model files and a valid serviceAccountKey.json for Firestore.")
        exit(1)


def register_new_student():
    global FACE_DETECTOR
    print("\n==============================")
    print("🎓 Register New Student")
    print("==============================")

    name = input("Enter student full name: ").strip()
    if not name:
        print("❌ Error: Name cannot be empty.")
        return
    
    reg_no = input("Enter registration / roll number: ").strip()
    if not reg_no:
        print("❌ Error: Registration number cannot be empty.")
        return

    folder_name = f"{name.replace(' ','_')}_{reg_no}"
    save_path = os.path.join(DATASET_DIR, folder_name)
    os.makedirs(save_path, exist_ok=True)

    print(f"Folder created: {save_path}")
    print("Starting webcam. Center your face. Press SPACE to capture a frame, ESC to finish.")

    if FACE_DETECTOR is None:
        try:
            FACE_DETECTOR = load_face_detector()
        except FileNotFoundError:
            print("Cannot register: Face detector files are missing.")
            return

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open webcam.")
        return

    img_counter = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        FACE_DETECTOR.setInput(blob)
        detections = FACE_DETECTOR.forward()

        face_detected = False
        faces = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence > 0.5:
                face_detected = True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype('int')
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                faces.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        instruction_color = (0, 255, 0) if face_detected else (0, 0, 255)
        instruction_text = "Ready to Capture (SPACE)" if face_detected else "No Face Detected"
        cv2.putText(frame, instruction_text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, instruction_color, 2)

        cv2.imshow(f'Register: {name} - Press SPACE to save, ESC to exit', frame)
        k = cv2.waitKey(1)

        if k%256 == 27:
            print('Escape hit, closing...')
            break
        elif k%256 == 32 and face_detected:
            best_face = None
            max_area = 0
            pad = 10
            for x1, y1, x2, y2 in faces:
                # Add padding and ensure bounds
                x1_pad = max(0, x1 + pad)
                y1_pad = max(0, y1 + pad)
                x2_pad = min(w - pad, x2 - pad)
                y2_pad = min(h - pad, y2 - pad)
                
                current_face = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                current_area = current_face.shape[0] * current_face.shape[1]
                if current_area > max_area and current_face.size > 0:
                    max_area = current_area
                    best_face = current_face
            if best_face is not None:
                img_name = f"{folder_name}_{img_counter}.jpg"
                img_path = os.path.join(save_path, img_name)
                cv2.imwrite(img_path, best_face)
                print(f"Saved {img_path}")
                img_counter += 1
            else:
                print("Could not crop face for saving.")
        elif k%256 == 32 and not face_detected:
            print("Please position your face correctly before pressing SPACE.")

    cam.release()
    cv2.destroyAllWindows()
    print(f"Saved {img_counter} face images for {name}.")
    
    # Validate minimum image count
    if img_counter < 3:
        print(f"⚠️ Warning: Only {img_counter} images captured. Recommended: 10-20 images for better accuracy.")
        confirm = input("Continue with registration anyway? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Registration cancelled. Please try again with more images.")
            # Clean up: remove the folder if user cancels
            import shutil
            if os.path.exists(save_path):
                shutil.rmtree(save_path)
            return
    
    print("--- Registration Complete ---")

 
    print("Updating embeddings (automatic)... This may take a little while depending on number of images...")
    update_embeddings()


    try:
        add_student_to_firestore(name, reg_no)
    except Exception as e:
        print(f"Warning: Failed to add student to Firestore: {e}")


def update_embeddings():
    print("\n==============================")
    print("🔄 Generating/Updating Embeddings")
    print("==============================")

    init_models()

    new_embs = []
    new_names = []

    for person_folder in os.listdir(DATASET_DIR):
        pf = os.path.join(DATASET_DIR, person_folder)
        if not os.path.isdir(pf):
            continue
        print(f"Processing folder: {person_folder}")
        for img_name in os.listdir(pf):
            img_path = os.path.join(pf, img_name)
            if not (img_path.endswith(('.jpg', '.jpeg', '.png'))):
                continue
            img = cv2.imread(img_path)
            if img is None or img.size == 0:
                print(f"  Skipping corrupted/empty file: {img_name}")
                continue
            try:
                processed_arr = preprocess_face(img)
                emb = EMBEDDING_MODEL.predict(processed_arr, verbose=0)[0]
                new_embs.append(emb)
                new_names.append(person_folder)
            except Exception as e:
                print(f"  Error processing image {img_name}: {e}")

    if new_embs:
        new_embs = np.array(new_embs)
    else:
        new_embs = np.array([])

    save_embeddings(new_embs, np.array(new_names))

    global KNOWN_EMBS, KNOWN_NAMES
    KNOWN_EMBS = new_embs
    KNOWN_NAMES = np.array(new_names)

    print("--- Embedding Update Complete ---")


def mark_attendance_live():
    print("\n==============================")
    print("🎯 Mark Attendance - Live Webcam")
    print("==============================")

    init_models()

    if len(KNOWN_NAMES) == 0:
        print("⚠️ No registered faces found. Please register students first.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    last_seen = {}
    COOLDOWN_SECONDS = 30

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        FACE_DETECTOR.setInput(blob)
        detections = FACE_DETECTOR.forward()

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype('int')
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                live_face_arr = preprocess_face(face)
                live_embedding = EMBEDDING_MODEL.predict(live_face_arr, verbose=0)[0]

                name, score = find_best_match(live_embedding, KNOWN_EMBS, KNOWN_NAMES, threshold=0.75)

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                label = f"{name} ({score:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                if name != "Unknown":
                    last_marked_time = last_seen.get(name)
                    if last_marked_time is None or (datetime.now() - last_marked_time).total_seconds() > COOLDOWN_SECONDS:
                        ok, msg = mark_attendance(name)
                        print(msg)
                        if ok:
                            last_seen[name] = datetime.now()

        cv2.imshow('Attendance - Press q to quit', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("--- Live Attendance Stopped ---")


def show_todays_attendance():
    user_date = input("Enter date to view attendance (YYYY-MM-DD) or press ENTER for today: ").strip()
    if user_date == '':
        user_date = date.today().isoformat()
    else:
        # Validate date format
        try:
            datetime.strptime(user_date, '%Y-%m-%d')
        except ValueError:
            print(f"❌ Error: Invalid date format '{user_date}'. Please use YYYY-MM-DD format.")
            return

    print("\n==============================")
    print(f"📅 Attendance for {user_date}")
    print("==============================")

    entries = get_attendance_by_date_firestore(user_date)
    if not entries:
        print("No attendance records found for this date.")
        return

    
    print(f"{'Reg No':<15} {'Name':<30} {'Time':<10}")
    print('-'*60)
    for reg_no, name, time_str in entries:
        print(f"{reg_no:<15} {name:<30} {time_str or '-':<10}")


def show_all_students():
    print("\n==============================")
    print("📚 All Registered Students")
    print("==============================")
    students = get_all_students_firestore()
    if not students:
        print("No students registered yet.")
        return
    print(f"{'Reg No':<15} {'Name':<40}")
    print('-'*60)
    for reg_no, name in students:
        print(f"{reg_no:<15} {name:<40}")


def main_menu():
    while True:
        print("\n\n---------------------------------------------------")
        print("👤 Facial Recognition Attendance System (OpenCV/Keras + Firestore)")
        print("---------------------------------------------------")
        print("1. Mark Attendance (Live Webcam)")
        print("2. Register New Student (Capture Images)")
        print("3. Show All Students")
        print("4. Show Attendance by Date")
        print("5. Exit")
        print("---------------------------------------------------")

        choice = input("Enter choice (1-5): ").strip()
        if choice == '1':
            mark_attendance_live()
        elif choice == '2':
            register_new_student()
        elif choice == '3':
            show_all_students()
        elif choice == '4':
            show_todays_attendance()
        elif choice == '5':
            print('👋 Goodbye!')
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")


if __name__ == '__main__':
    main_menu()
