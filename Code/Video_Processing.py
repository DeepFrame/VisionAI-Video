import os
import cv2
import json
import torch
import pyodbc
import logging
import numpy as np
from datetime import datetime
from retinaface import RetinaFace
from deep_sort_realtime.deepsort_tracker import DeepSort

import tensorflow as tf
from keras_facenet import FaceNet
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# GPU Setup
# ------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] {len(gpus)} GPU(s) detected: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"[ERROR] GPU setup failed: {e}")
else:
    print("[INFO] No GPU detected, running on CPU")

# ------------------------------
# Logging Setup
# ------------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "visionai_video.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")]
)

logger = logging.getLogger("VisionAI-Video")

# ------------------------------
# SQL Connection
# ------------------------------
SQL_CONNECTION_STRING = (
    "Driver={ODBC Driver 17 for SQL Server};"
    "Server=(localdb)\\MSSQLLocalDB;"
    "Database=VisionAI;"
    "Encrypt=no;"
    "TrustServerCertificate=no;"
)
conn_str = SQL_CONNECTION_STRING
THUMBNAIL_SAVE_PATH = "Faces_Extracted_RetinaFace"
os.makedirs(THUMBNAIL_SAVE_PATH, exist_ok=True)

embedder = FaceNet()
# ------------------------------
# Database functions
# ------------------------------
def get_unprocessed_files():
    try:
        conn = pyodbc.connect(SQL_CONNECTION_STRING)
        cursor = conn.cursor()
        query = """
        SELECT MI.Id, MF.FilePath, MI.Name
        FROM dbo.MediaItems MI
        JOIN dbo.MediaFile MF ON MI.MediaFileId = MF.Id
        WHERE MI.IsFacesExtracted = 0
          AND LOWER(MF.Extension) IN ('.mp4', '.avi', '.mov', '.mkv')
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"Database fetch failed: {e}")
        return []

def update_database(media_item_id, face_metadata, filename=None):
    try:
        conn = pyodbc.connect(SQL_CONNECTION_STRING)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE dbo.MediaItems
            SET IsFacesExtracted = 1,
                FacesExtractedOn = ?
            WHERE Id = ?
        """, datetime.now(), media_item_id)

        inserted_count = 0
        for face in face_metadata:
            bbox = face["bbox"]
            filename = face["filename_of_face"]
            frame_number = face["frame_number"]
            bbox_str = json.dumps([round(float(c), 3) for c in bbox])

            cursor.execute("""
                INSERT INTO dbo.Faces (MediaItemId, BoundingBox, Name, FrameNumber, CreatedAt, IsUserVerified)
                VALUES (?, ?, ?, ?, ?, ?)
            """, media_item_id, bbox_str, filename, frame_number, datetime.now(), 0)
            inserted_count += 1

        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"✅ Inserted {inserted_count} faces for MediaItemId {media_item_id}")
    except Exception as e:
        logger.error(f"Database update failed: {e}")
        if 'conn' in locals():
            conn.rollback()

def get_faces_for_embedding():
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT Id, Name
        FROM dbo.Faces
        WHERE Embedding IS NULL AND Name IS NOT NULL
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    results = []
    for face_id, file_name in rows:
        file_path = os.path.join(THUMBNAIL_SAVE_PATH, file_name)
        if os.path.exists(file_path):
            results.append({"FaceId": face_id, "FilePath": file_path})
        else:
            logger.warning(f"Thumbnail not found: {file_path}")
    return results

def update_face_embedding(face_id: int, embedding: np.ndarray, dry_run: bool = False):
    """
    Update the Embedding column for a given FaceId in dbo.Faces.
    """
    try:
        if embedding is None or not isinstance(embedding, np.ndarray):
            logger.warning(f"[SKIP] Invalid embedding for FaceId={face_id}")
            return

        emb_bytes = embedding.astype(np.float32).tobytes()

        if dry_run:
            logger.info(f"[Dry Run] Would update FaceId={face_id} with embedding of shape {embedding.shape}")
            return

        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE dbo.Faces
            SET Embedding = ?, ModifiedAt = ?
            WHERE Id = ?
        """, emb_bytes, datetime.now(), face_id)
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"[OK] Updated embedding for FaceId={face_id}")

    except pyodbc.Error as db_err:
        logger.error(f"[DB ERROR] Failed to update embedding for FaceId={face_id}: {db_err}")
        if 'conn' in locals():
            conn.rollback()
            cursor.close()
            conn.close()
    except Exception as e:
        logger.error(f"[ERROR] Unexpected error updating embedding for FaceId={face_id}: {e}")

def get_unassigned_faces(recluster=False, labelled=False):
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    if recluster:
        cursor.execute("""
            SELECT Id, Embedding
            FROM dbo.Faces
            WHERE Embedding IS NOT NULL
        """)
    elif labelled:
        cursor.execute("""
            SELECT Id, Embedding, PersonId
            FROM dbo.Faces
            WHERE PersonId IS NOT NULL AND Embedding IS NOT NULL
        """)
    else:
        cursor.execute("""
            SELECT Id, Embedding
            FROM dbo.Faces
            WHERE PersonId IS NULL AND Embedding IS NOT NULL
        """)

    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

# ------------------------------
# Keyframe Extraction
# ------------------------------
def get_motion_score(frame1, frame2, kernel=np.ones((9,9), dtype=np.uint8)):
    frame_diff = cv2.subtract(frame2, frame1)
    frame_diff = cv2.medianBlur(frame_diff, 3)
    mask = cv2.adaptiveThreshold(frame_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 3)
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return np.sum(mask) / (mask.shape[0] * mask.shape[1])

def extract_keyframes(video_path, threshold=10.0):
    cap = cv2.VideoCapture(video_path)
    keyframes = []
    ret, prev_frame = cap.read()
    if not ret:
        return []
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    keyframes.append((0, prev_frame))
    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_score = get_motion_score(prev_gray, gray)
        if motion_score > threshold:
            keyframes.append((frame_idx, frame))
            prev_gray = gray
        frame_idx += 1
    cap.release()
    return keyframes

def alternative_algorithm(video_path, skip_frames=10):
    cap = cv2.VideoCapture(video_path)
    keyframes = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip_frames == 0:
            keyframes.append((frame_idx, frame))
        frame_idx += 1
    cap.release()
    return keyframes

# ------------------------------
# Face detection + cropping helpers
# ------------------------------
def detect_faces(frame):
    faces_detected = RetinaFace.detect_faces(frame)
    detections = []
    if isinstance(faces_detected, dict):
        for _, face_data in faces_detected.items():
            x1, y1, x2, y2 = face_data["facial_area"]
            conf = face_data.get("score", 0.99)
            detections.append(([x1, y1, x2-x1, y2-y1], conf, 0))
    return detections

def process_face_square(img, face, margin_ratio=0.2, target_size=(112, 112)):    
    h, w = img.shape[:2]
    x1, y1, x2, y2 = face["facial_area"]

    bw = x2 - x1
    bh = y2 - y1
    margin_x = int(bw * margin_ratio)
    margin_y = int(bh * margin_ratio)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)

    crop_w = x2 - x1
    crop_h = y2 - y1
    if crop_w > crop_h:
        diff = crop_w - crop_h
        expand_top = diff // 2
        expand_bottom = diff - expand_top
        if y1 - expand_top >= 0 and y2 + expand_bottom <= h:
            y1 -= expand_top
            y2 += expand_bottom
        else:
            x1 += diff // 2
            x2 -= (diff - diff // 2)
    elif crop_h > crop_w:
        diff = crop_h - crop_w
        expand_left = diff // 2
        expand_right = diff - expand_left
        if x1 - expand_left >= 0 and x2 + expand_right <= w:
            x1 -= expand_left
            x2 += expand_right
        else:
            y1 += diff // 2
            y2 -= (diff - diff // 2)

    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    cropped_face = img[y1:y2, x1:x2].copy() 
    if cropped_face.size == 0:
        return None, None

    resized_face = cv2.resize(cropped_face, target_size, interpolation=cv2.INTER_AREA)
    updated_bbox = [int(x1), int(y1), int(x2), int(y2)]  
    return resized_face, updated_bbox

def sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# ------------------------------
# Face Embedding
# ------------------------------
def parse_embedding(raw_value):
    try:
        if isinstance(raw_value, bytes):
            return np.frombuffer(raw_value, dtype=np.float32)
        elif isinstance(raw_value, str):
            return np.array(json.loads(raw_value), dtype=np.float32)
        return None
    except Exception as e:
        logger.error(f"Embedding parse error: {e}")
        return None
    
def embed_faces(faces, dry_run=False):
    for face in faces:
        face_id = face["FaceId"]
        file_path = face["FilePath"]

        img = cv2.imread(file_path)
        if img is None:
            logger.warning(f"Could not read {file_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (160, 160))

        emb = embedder.embeddings([img_resized])[0]

        if not dry_run:
            update_face_embedding(face_id, emb)
            logger.info(f"[OK] Embedded FaceId={face_id}")

# ------------------------------
# Cluster Assignment
# ------------------------------
def assign_clusters(labels, face_ids, recluster=False, existing_person_id=None, dry_run=False):
    """
    Assign clusters to faces.
    - If existing_person_id is provided, assign faces directly to that person.
    - Otherwise, create new person entries per cluster.
    """
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        # === Direct assignment to existing person ===
        if existing_person_id is not None:
            if dry_run:
                logger.info(f"Dry run: Assigning {len(face_ids)} faces to PersonId={existing_person_id}")
            else:
                logger.info(f"Assigning {len(face_ids)} faces to existing PersonId={existing_person_id}")
                for face_id in face_ids:
                    cursor.execute("EXEC dbo.UpsertFace @FaceId=?, @PersonId=?", (face_id, existing_person_id))
                    cursor.execute("EXEC dbo.UpsertPerson ?, ?, ?, ?", (int(existing_person_id), None, None, None))

            conn.commit()
            cursor.close()
            conn.close()
            return

        # === Normal clustering mode ===
        logger.info(f"Assigning {len(set(labels))} clusters to {len(face_ids)} faces...")
        for cluster_id in set(labels):
            if cluster_id == -1: 
                continue
            if not dry_run:
                cursor.execute("""
                    INSERT INTO dbo.Persons (Name, Rank, Appointment, CreatedAt, Type)
                    OUTPUT INSERTED.Id
                    VALUES (?, ?, ?, ?, ?)
                """, f"Unknown-{int(cluster_id)}", None, None, datetime.now(), 0)
            person_id = int(cursor.fetchone()[0])

            logger.info(f"Created new PersonId={person_id} for cluster {cluster_id}")

            for face_id, label in zip(face_ids, labels):
                if label == cluster_id and not dry_run:
                    cursor.execute("""
                        UPDATE dbo.Faces
                        SET PersonId = ?, ModifiedAt = ?
                        WHERE Id = ?
                    """, person_id, datetime.now(), int(face_id))

            if not dry_run:
                cursor.execute("""
                    UPDATE dbo.Persons
                    SET ModifiedAt = ?
                    WHERE Id = ?
                """, datetime.now(), person_id)

        conn.commit()
        cursor.close()
        conn.close()

    except Exception as e:
        logger.error(f"DB update failed (assign clusters): {e}")

def cluster_faces(eps=0.35, min_samples=3, dry_run=False):
    rows = get_unassigned_faces(recluster=False)
    if not rows:
        logger.info("No unassigned faces to cluster")
        return

    face_ids, embeddings = [], []
    for row in rows:
        face_id, raw_emb = row
        emb = parse_embedding(raw_emb)
        if emb is not None:
            face_ids.append(face_id)
            embeddings.append(emb)

    embeddings = np.array(embeddings)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = clustering.fit_predict(embeddings)

    logger.info(f"Clusters found: {set(labels)}")
    assign_clusters(labels, face_ids, dry_run=dry_run)

# ------------------------------
# Face Embedding + Clustering Pipeline
# ------------------------------
def process_thumbnails(dry_run=False):
    print("[Step 1] Fetching thumbnails for embedding...")
    faces = get_faces_for_embedding()
    if not faces:
        print("[INFO] No thumbnails to embed.")
        return

    print(f"[INFO] Embedding {len(faces)} faces...")
    embed_faces(faces, dry_run=dry_run)

    print("[Step 2] Clustering unassigned faces...")
    cluster_faces(dry_run=dry_run)

# ------------------------------
# Main Processing with DeepSort + Keyframes
# ------------------------------
OUTPUT_FACES_DIR = "Faces_Extracted_RetinaFace"
os.makedirs(OUTPUT_FACES_DIR, exist_ok=True)

def process_videos_from_db():
    videos = get_unprocessed_files()
    logger.info(f"Found {len(videos)} unprocessed videos")
    
    for media_item_id, video_path, name in videos:
        logger.info(f"\n🎥 Processing {name} ({video_path})")
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration_sec = frame_count / fps if fps > 0 else 0
        cap.release()

        keyframes = (alternative_algorithm(video_path) if duration_sec < 60
                     else extract_keyframes(video_path, threshold=28.0))

        print(f"Extracted {len(keyframes)} keyframes from {name}")
        
        face_metadata_all = []
        
        if len(keyframes) < 3:
            logger.warning(f"Too few keyframes ({len(keyframes)}), skipping DeepSort and saving faces directly.")
            for idx, frame in keyframes:
                faces_detected = RetinaFace.detect_faces(frame)
                if isinstance(faces_detected, dict):
                    for fid, face_data in faces_detected.items():
                        face_crop, bbox = process_face_square(frame, face_data, margin_ratio=0.2, target_size=(160,160))
                        if face_crop is None:
                            continue
                        face_file = os.path.join(OUTPUT_FACES_DIR, f"{video_name}_frame{idx}_face.png")
                        cv2.imwrite(face_file, face_crop)
                        face_metadata_all.append({
                            "bbox": bbox,
                            "cropped_face_path": face_file,
                            "filename_of_face": os.path.basename(face_file),
                            "frame_number": idx
                        })
            logger.info(f"Saved {len(face_metadata_all)} faces directly for {name}")
        else:
            tracker = DeepSort(max_age=10, n_init=1)
            track_faces = {}
            
            for idx, frame in keyframes:
                detections = detect_faces(frame)
                logger.info(f"Frame {idx}: Detected {len(detections)} faces")

                tracks = tracker.update_tracks(detections, frame=frame)
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    x1, y1, x2, y2 = map(int, track.to_ltrb())
                    face_data = {"facial_area": [x1, y1, x2, y2]}
                    face_crop, bbox = process_face_square(frame, face_data, margin_ratio=0.2, target_size=(160,160))
                    if face_crop is None:
                        continue
                    if track.track_id not in track_faces:
                        track_faces[track.track_id] = []
                    track_faces[track.track_id].append((face_crop, bbox, idx))

            face_metadata_all = []
            for track_id, faces in track_faces.items():
                valid_faces = [f for f in faces if f[0] is not None]
                if not valid_faces:
                    continue
                sharpness_scores = [(sharpness(f[0]), f) for f in valid_faces]
                
                best_score, (sharpest_face, bbox, frame_number) = max(sharpness_scores, key=lambda x: x[0])
                
                logger.info(f"Track ID {track_id}: Selected face from frame {frame_number} with **Sharpness (Laplacian) Score: {best_score:.2f}**")
                face_file = os.path.join(OUTPUT_FACES_DIR, f"{video_name}_track{track_id}_face.png")
                cv2.imwrite(face_file, sharpest_face)
                face_metadata_all.append({
                    "bbox": bbox,
                    "cropped_face_path": face_file,
                    "filename_of_face": os.path.basename(face_file),
                    "frame_number": frame_number
                })
            logger.info(f"Saved {len(face_metadata_all)} representative faces for {name}")

        update_database(media_item_id, face_metadata_all, filename=name)

    logger.info("Detection complete, and database updated!")
    process_thumbnails(dry_run=False)

    logger.info("Recognition complete, database updated!")

if __name__ == "__main__":
    process_videos_from_db()
    print("🎯 Processing complete, database updated!")
