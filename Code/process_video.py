#  Import Libraries
import os
import cv2
import re
import math
import json
import pyodbc
import logging
import numpy as np
from glob import glob
from datetime import datetime
import matplotlib.pyplot as plt
from retinaface import RetinaFace

#  Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VisionAI-Video")

#  SQL Server Connection
SQL_CONNECTION_STRING = (
    "Driver={ODBC Driver 17 for SQL Server};"
    "Server=(localdb)\\MSSQLLocalDB;"
    "Database=Face_Recognition_System;"
    "Encrypt=no;"
    "TrustServerCertificate=no;"
)

#  DB Functions
def get_unprocessed_files():
    """Fetch videos that have not had faces extracted yet."""
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
    """
    Inserts detected faces into dbo.Faces with bounding boxes and file paths.
    Updates MediaItems to mark IsFacesExtracted.
    """
    try:
        conn = pyodbc.connect(SQL_CONNECTION_STRING)
        cursor = conn.cursor()

        # Update MediaItems status
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

#  Frame Extraction
def get_motion_score(frame1, frame2, kernel=np.ones((9,9), dtype=np.uint8)):
    frame_diff = cv2.subtract(frame2, frame1)
    frame_diff = cv2.medianBlur(frame_diff, 3)
    mask = cv2.adaptiveThreshold(frame_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 3)
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    motion_score = np.sum(mask) / (mask.shape[0] * mask.shape[1])
    return motion_score


def extract_keyframes(video_path, threshold=10.0):
    """Extract frames based on motion detection."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: Could not open {video_path}")
        return []

    keyframes = []
    ret, prev_frame_bgr = cap.read()
    if not ret:
        logger.error(f"Error: Could not read first frame of {video_path}")
        return []

    prev_frame_gray = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2GRAY)
    keyframes.append((0, prev_frame_bgr))

    frame_idx = 1
    while True:
        ret, curr_frame_bgr = cap.read()
        if not ret:
            break
        curr_frame_gray = cv2.cvtColor(curr_frame_bgr, cv2.COLOR_BGR2GRAY)
        motion_score = get_motion_score(prev_frame_gray, curr_frame_gray)

        if motion_score > threshold:
            keyframes.append((frame_idx, curr_frame_bgr))
            prev_frame_gray = curr_frame_gray

        frame_idx += 1

    cap.release()
    logger.info(f"✅ Extracted {len(keyframes)} keyframes from {video_path}")
    return keyframes


def alternative_algorithm(video_path, skip_frames=10):
    """Extract frames by skipping fixed number of frames."""
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
    logger.info(f"✅ Extracted {len(keyframes)} frames (skipped {skip_frames}) from {video_path}")
    return keyframes

#  Face Detection + Cropping
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

    cropped_face = img[y1:y2, x1:x2]

    if cropped_face.size == 0:
        return None, None

    resized_face = cv2.resize(cropped_face, target_size, interpolation=cv2.INTER_AREA)
    updated_bbox = [int(x1), int(y1), int(x2), int(y2)]  
    return resized_face, updated_bbox

def detect_and_crop_faces(frame, margin_ratio=0.2, target_size=(112, 112)):
    faces_detected = RetinaFace.detect_faces(frame)
    cropped_faces = []

    if not isinstance(faces_detected, dict) or len(faces_detected) == 0:
        return []

    for _, face_data in faces_detected.items():
        cropped_face, updated_bbox = process_face_square(frame, face_data, margin_ratio, target_size)
        if cropped_face is None:
            continue
        cropped_faces.append((cropped_face, updated_bbox))

    return cropped_faces

#  Directories Setup
video_directory = "Videos"
output_dir = "Processed_Video_Frames"
cropped_faces_directory = "Faces_Extracted_RetinaFace"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(cropped_faces_directory, exist_ok=True)

duration_threshold_sec = 60

#  Main Processing Loop
def process_videos_from_db():
    videos = get_unprocessed_files()
    logger.info(f"Found {len(videos)} unprocessed videos")

    for media_item_id, video_path, name in videos:
        logger.info(f"\n🎥 Processing {name} ({video_path})")

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        save_path = os.path.join(output_dir, video_name)
        os.makedirs(save_path, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration_sec = frame_count / fps if fps > 0 else 0
        cap.release()

        if duration_sec < duration_threshold_sec:
            keyframes = alternative_algorithm(video_path)
        else:
            keyframes = extract_keyframes(video_path, threshold=28.0)

        face_metadata_all = []
        for i, (idx, frame) in enumerate(keyframes):
            frame_path = os.path.join(save_path, f"{video_name}_keyframe_{i}_frame{idx}.png")
            cv2.imwrite(frame_path, frame)

            faces = detect_and_crop_faces(frame)
            for fidx, (face_img, bbox) in enumerate(faces):
                face_file = os.path.join(
                    cropped_faces_directory,
                    video_name,
                    f"{video_name}_frame{idx:05d}_face{fidx}.png"
                )
                os.makedirs(os.path.dirname(face_file), exist_ok=True)
                cv2.imwrite(face_file, face_img)

                filename_of_face = os.path.basename(face_file)

                face_metadata_all.append({
                    "bbox": bbox,
                    "cropped_face_path": face_file,
                    "filename_of_face": filename_of_face,
                    "frame_number": idx 
                })
        
        update_database(media_item_id, face_metadata_all, filename=name)

#  Run Pipeline
if __name__ == "__main__":
    process_videos_from_db()
    print("🎯 Processing complete, database updated!")
