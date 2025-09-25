import os
import cv2
import json
import torch
import pyodbc
import logging
from typing import Optional
import numpy as np
from datetime import datetime
from retinaface import RetinaFace
from deep_sort_realtime.deepsort_tracker import DeepSort

import tensorflow as tf
from keras_facenet import FaceNet
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from dotenv import load_dotenv

import shutil
from pathlib import Path
from tabulate import tabulate
# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv()

SQL_CONNECTION_STRING = os.getenv("SQL_CONNECTION_STRING")

CURRENT_DIR = os.getcwd()
THUMBNAIL_SAVE_PATH = os.path.join(CURRENT_DIR, "Faces_Extracted_RetinaFace")


os.makedirs(THUMBNAIL_SAVE_PATH, exist_ok=True)

conn_str = SQL_CONNECTION_STRING

DEST_ROOT = os.path.join(CURRENT_DIR, "ByPerson")
os.makedirs(DEST_ROOT, exist_ok=True)

LOG_DIR_OR_FILE = os.path.join(CURRENT_DIR, "logs") 
os.makedirs(LOG_DIR_OR_FILE, exist_ok=True)

KEEP_COPY = True
DRY_RUN = False

embedder = FaceNet()

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
LOG_FILE = os.path.join(LOG_DIR_OR_FILE, "visionai_video.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")]
)

logger = logging.getLogger("VisionAI-Video")

# -------------------------------
# Person-Wise Folder 
# -------------------------------
def get_thumb_logger(log_path: Optional[str] = None, name: str = "move_thumbnails") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if log_path is None:
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(sh)
        return logger

    p = Path(log_path)
    if (p.exists() and p.is_dir()) or not p.suffix:
        p = p / f"{name}.log"
    p.parent.mkdir(parents=True, exist_ok=True)

    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(p)
               for h in logger.handlers):
        try:
            fh = logging.FileHandler(p, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(fh)
        except Exception as e:
            # Fall back to console if file can't be opened
            if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
                sh = logging.StreamHandler()
                sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
                logger.addHandler(sh)
            logger.warning(f"Could not open log file '{p}': {e}. Falling back to console.")
    return logger

def move_thumbnails_by_person(
    sql_connection_string: str,
    thumbnail_save_path: str,
    dest_root: Optional[str] = None,
    keep_copy: bool = False,
    dry_run: bool = False,
    log_path: Optional[str] = None,
) -> None:
    logger_thumb = get_thumb_logger(log_path, "by_person_faces")

    if not sql_connection_string or not isinstance(sql_connection_string, str):
        raise ValueError(
            "sql_connection_string is empty. Set SQL_CONNECTION_STRING in your environment/.env "
            "or pass a non-empty string to move_thumbnails_by_person()."
        )

    src_root = Path(thumbnail_save_path).resolve()
    if dest_root is None:
        dest_root = src_root
    dest_root = Path(dest_root).resolve()

    try:
        with pyodbc.connect(sql_connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        p.Id       AS PersonId,
                        f.Id       AS FaceId,
                        f.Name     AS ThumbFileName
                    FROM dbo.Faces f
                    INNER JOIN dbo.Persons p
                        ON f.PersonId = p.Id
                    WHERE f.Name IS NOT NULL AND LTRIM(RTRIM(f.Name)) <> ''
                """)
                rows = cur.fetchall()

        if not rows:
            print("[INFO] No faces with PersonId and thumbnail file name found.")
            return
        if rows:
            table_data = [list(row) for row in rows]
            table_headers = ["PersonId", "FaceId", "ThumbFileName"]
            table_str = tabulate(table_data, headers=table_headers, tablefmt="grid")

            # Log to logger_thumb
            logger_thumb.info("\n" + table_str)

        moved = copied = skipped_missing = skipped_errors = 0

        for person_id, face_id, thumb_name in rows:
            try:
                src = src_root / thumb_name
                if not src.exists():
                    logger.warning(f"[MISS] Source thumbnail not found: {src}")
                    skipped_missing += 1
                    continue

                person_dir = dest_root / f"Person_{int(person_id)}"
                dst = person_dir / src.name

                if dst.exists():
                    stem, suffix = dst.stem, dst.suffix
                    dst = person_dir / f"{stem}_Face{int(face_id)}{suffix}"
                    k = 1
                    while dst.exists():
                        dst = person_dir / f"{stem}_Face{int(face_id)}_{k}{suffix}"
                        k += 1

                action = "COPY" if keep_copy else "MOVE"
                if dry_run:
                    print(f"[DryRun] {action}: {src}  ->  {dst}")
                    continue

                person_dir.mkdir(parents=True, exist_ok=True)
                if keep_copy:
                    shutil.copy2(src, dst)
                    copied += 1
                    logger_thumb.info(f"[COPY] PersonId={person_id}, FaceId={face_id}, File={src} -> {dst}")
                else:
                    shutil.move(str(src), str(dst))
                    moved += 1
                    logger_thumb.info(f"[MOVE] PersonId={person_id}, FaceId={face_id}, File={src} -> {dst}")

            except Exception as e:
                logger.error(f"[ERROR] Failed FaceId={face_id}, file={thumb_name}: {e}")
                skipped_errors += 1

        print(
            f"[DONE] Persons Grouped Under: {dest_root}\n"
            f"       moved={moved}, copied={copied}, "
            f"missing={skipped_missing}, errors={skipped_errors}"
        )
        logger.info(
            f"[GROUPED] dest_root={dest_root} moved={moved} copied={copied} "
            f"missing={skipped_missing} errors={skipped_errors}"
        )

    except Exception as e:
        logger.error(f"[FATAL] move_thumbnails_by_person failed: {e}")
        print(f"[ERROR] move_thumbnails_by_person failed: {e}")    

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

def get_mediafile_id_for_face(face_id: int) -> Optional[int]:
    """
    Returns MediaFileId for a given FaceId
    """
    try:
        with pyodbc.connect(conn_str) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT mf.Id
                FROM dbo.Faces f
                JOIN dbo.MediaItems mi ON f.MediaItemId = mi.Id
                JOIN dbo.MediaFile mf ON mi.MediaFileId = mf.Id
                WHERE f.Id = ?
            """, face_id)
            row = cursor.fetchone()
            cursor.close()
            return row[0] if row else None
    except Exception as e:
        logger.error(f"Failed to get MediaFileId for FaceId={face_id}: {e}")
        return None

def update_person_portrait(person_id: int, face_id: int, dry_run: bool = False):
    """
    Sets PortraitFaceId and PortraitMediaFileId for a given PersonId
    """
    mediafile_id = get_mediafile_id_for_face(face_id)
    if mediafile_id is None:
        logger.warning(f"No MediaFileId found for FaceId={face_id}, skipping portrait update.")
        return

    if dry_run:
        logger.info(f"[DryRun] Would update PersonId={person_id} with PortraitFaceId={face_id}, PortraitMediaFileId={mediafile_id}")
        return

    try:
        with pyodbc.connect(conn_str) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE dbo.Persons
                SET PortraitFaceId = ?, PortraitMediaFileId = ?, ModifiedAt = ?
                WHERE Id = ?
            """, face_id, mediafile_id, datetime.now(), person_id)
            conn.commit()
            cursor.close()
            logger.info(f"[OK] Updated PersonId={person_id} with PortraitFaceId={face_id}, PortraitMediaFileId={mediafile_id}")
    except Exception as e:
        logger.error(f"Failed to update portrait for PersonId={person_id}: {e}")

def assign_portraits_to_persons(dry_run: bool = False):
    """
    For each person, pick a medoid/sharpest face as portrait and update the DB
    """
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT Id FROM dbo.Persons
        """)
        persons = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()

        for person_id in persons:
            conn = pyodbc.connect(conn_str)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT Id, Embedding
                FROM dbo.Faces
                WHERE PersonId = ? AND Embedding IS NOT NULL
            """, person_id)
            rows = cursor.fetchall()
            cursor.close()
            conn.close()

            if not rows:
                continue

            face_ids, embeddings = zip(*rows)
            embeddings = np.array([parse_embedding(e) for e in embeddings])
            dist_matrix = cosine_distances(embeddings)
            medoid_index = np.argmin(dist_matrix.sum(axis=1))
            portrait_face_id = int(face_ids[medoid_index])

            update_person_portrait(person_id, portrait_face_id, dry_run=dry_run)
    except Exception as e:
        logger.error(f"assign_portraits_to_persons failed: {e}")

# -------------------------------
# Process Missing Thumbnails or Faces
# -------------------------------
def get_frame_from_video(video_path, frame_number):
    """
    Extracts a single frame from a video file at a specific frame number.
    Returns the frame (numpy array) or None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file: {video_path}")
        return None
    
    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return frame
    else:
        logger.error(f"Could not read frame {frame_number} from video: {video_path}")
        return None

def reprocess_media_missing_faces(dry_run: bool = False):
    """
    Find MediaItems marked as extracted (IsFacesExtracted=1 and FacesExtractedOn IS NOT NULL)
    but with no corresponding rows in dbo.Faces. Re-run detection and write into dbo.Faces.
    """
    try:
        with pyodbc.connect(SQL_CONNECTION_STRING) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT mi.Id AS MediaItemId, mf.FilePath, mf.FileName
                    FROM dbo.MediaItems mi
                    INNER JOIN dbo.MediaFile mf ON mi.MediaFileId = mf.Id
                    WHERE mi.IsFacesExtracted = 1
                      AND mi.FacesExtractedOn IS NOT NULL
                      AND NOT EXISTS (
                          SELECT 1
                          FROM dbo.Faces f
                          WHERE f.MediaItemId = mi.Id
                      )
                """)
                rows = cur.fetchall()

        if not rows:
            logger.info("All processed MediaItems have Faces records. Nothing to reprocess.")
            return

        logger.info(f"Reprocessing {len(rows)} MediaItems with missing Faces rows...")
        for media_item_id, file_path, file_name in rows:
            try:
                full_path = file_path
            except Exception as e:
                logger.error(f"Path mapping failed for MediaItemId {media_item_id}: {e}")
                continue

            ok = processing(full_path, media_item_id=media_item_id, name=file_name)
            if not ok:
                logger.warning(f"Re-detect skipped/failed for MediaItemId {media_item_id}")

    except Exception as e:
        logger.error(f"reprocess_media_missing_faces failed: {e}")

# Updated check_thumbnails function:
def check_thumbnails(dry_run=False):
    """
    Check if faces thumbnails exist for each entry in dbo.Faces.
    If missing, recreate them using the stored bounding box and frame number 
    from the original video file.
    """
    thumbnail_base_path = THUMBNAIL_SAVE_PATH

    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        query = """
        SELECT 
            f.Id, f.BoundingBox, f.Name, mf.FilePath, mi.Name, f.FrameNumber
        FROM dbo.Faces f
        INNER JOIN dbo.MediaItems mi ON f.MediaItemId = mi.Id
        INNER JOIN dbo.MediaFile mf ON mi.MediaFileId = mf.Id
        WHERE mi.FacesExtractedOn IS NOT NULL -- Only check faces from processed media
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if not rows:
            logger.info("No faces found in DB.")
            return

        for face_id, bbox_str, face_name, file_path, media_item_name, frame_number in rows: 
            try:
                full_video_path = file_path 
                
                if face_name:
                    thumb_path = os.path.join(thumbnail_base_path, face_name)
                else:
                    thumb_name = f"{Path(media_item_name).stem}_frame{frame_number}_face.png"
                    thumb_path = os.path.join(thumbnail_base_path, thumb_name)

                if os.path.exists(thumb_path):
                    logger.debug(f"Thumbnail exists: {thumb_path}")
                    continue  # skip

                logger.info(f"[Missing] Recreating thumbnail: {os.path.basename(thumb_path)} from frame {frame_number}")

                frame = get_frame_from_video(full_video_path, frame_number)
                if frame is None:
                    logger.error(f"Could not retrieve frame {frame_number} from video: {full_video_path}")
                    continue

                if not bbox_str:
                    logger.warning(f"No bounding box for FaceId {face_id}")
                    continue

                bbox = json.loads(bbox_str) if isinstance(bbox_str, str) else None
                if not bbox or len(bbox) != 4:
                    logger.warning(f"Invalid bbox for FaceId {face_id}: {bbox_str}")
                    continue

                x1, y1, x2, y2 = map(int, bbox)
                h, w = frame.shape[:2]
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)

                cropped = frame[y1:y2, x1:x2]

                if cropped.size == 0:
                    logger.warning(f"Empty crop for FaceId {face_id} with bbox {bbox}")
                    continue

                target_size = (160, 160)
                resized_cropped = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)

                if not dry_run:
                    cv2.imwrite(thumb_path, resized_cropped) 
                    logger.info(f"[OK] Created {thumb_path}")
                    
                    if not face_name:
                        conn_update = pyodbc.connect(conn_str)
                        cursor_update = conn_update.cursor()
                        cursor_update.execute("UPDATE dbo.Faces SET Name = ?, ModifiedAt = ? WHERE Id = ?", 
                                              os.path.basename(thumb_path), datetime.now(), face_id)
                        conn_update.commit()
                        cursor_update.close()
                        conn_update.close()

                else:
                    logger.info(f"[DryRun] Would create {thumb_path}")

            except Exception as e:
                logger.error(f"Error processing FaceId {face_id}: {e}")

    except Exception as e:
        logger.error(f"check_thumbnails() failed: {e}")

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

# -------------------------------
# Process Video with Tracking
# -------------------------------
def processing(video_path, media_item_id, name):
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

# ------------------------------
# Main Processing with DeepSort + Keyframes
# ------------------------------
OUTPUT_FACES_DIR = "Faces_Extracted_RetinaFace"
os.makedirs(OUTPUT_FACES_DIR, exist_ok=True)

def process_videos_from_db():
    reprocess_media_missing_faces()
    check_thumbnails()

    videos = get_unprocessed_files()
    logger.info(f"Found {len(videos)} unprocessed videos")
    
    for media_item_id, video_path, name in videos:
        processing(video_path, media_item_id, name)

    logger.info("Detection complete, and database updated!")
    print("Detection complete, and database updated!")
    process_thumbnails(dry_run=False)

    logger.info("Recognition complete, database updated!")
    print("Recognition complete, database updated!")

    move_thumbnails_by_person(
            sql_connection_string=SQL_CONNECTION_STRING,
            thumbnail_save_path=THUMBNAIL_SAVE_PATH,
            dest_root=DEST_ROOT,
            keep_copy=KEEP_COPY,
            dry_run=DRY_RUN,
            log_path=LOG_DIR_OR_FILE,  # dir or file both fine
        )
    
    print("Person Wise grouping complete!")

    assign_portraits_to_persons()

    print("Assigned portrait faces to persons...")

if __name__ == "__main__":
    process_videos_from_db()
    print("🎯 Processing complete, database updated!")
