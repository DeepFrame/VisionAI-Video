# VisionAI Video Processing Pipeline

This project is a **video face extraction and processing pipeline** using Python, OpenCV, and RetinaFace. It connects to a SQL Server database to fetch unprocessed videos, extract keyframes, detect faces, crop them, and store metadata back in the database.

---

## Table of Contents

- [Features](#features)  
- [Requirements](#requirements)  
- [Setup](#setup)  
- [Configuration](#configuration)  
- [Usage](#usage)  
- [Pipeline Workflow](#pipeline-workflow)  

---

## Features

- Fetch unprocessed video files from SQL Server database.
- Extract **keyframes** using either:
  - Motion detection
  - Frame skipping (for short videos)
- Detect faces in frames using **RetinaFace**.
- Crop faces to square format with configurable margins.
- Save frames and cropped faces to disk.
- Store face metadata (bounding boxes, frame number, filename) in the database.
- Logging for monitoring progress and errors.

---

## Requirements

- Python 3.8+
- Packages:

```bash
pip install opencv-python retina-face numpy matplotlib pyodbc
````

* Microsoft SQL Server (LocalDB or other instance)
* ODBC Driver 17 for SQL Server

---

## Setup

1. Clone this repository:

```bash
git clone <repo_url>
cd <repo_name>
```

2. Conda Environment Setup:

```bash
conda create -n VideoProcessing python=3.8.20
conda activate VideoProcessing
```

4. Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

### SQL Server Connection

Update the `SQL_CONNECTION_STRING` in the script with your server, database, and authentication details:

```python
SQL_CONNECTION_STRING = (
    "Driver={ODBC Driver 17 for SQL Server};"
    "Server=(localdb)\\MSSQLLocalDB;"
    "Database=Face_Recognition_System;"
    "Encrypt=no;"
    "TrustServerCertificate=no;"
)
```

### Thresholds & Parameters

* `duration_threshold_sec = 60` — Use frame skipping for videos shorter than this.
* `extract_keyframes(video_path, threshold=28.0)` — Motion detection threshold for keyframes.
* `process_face_square` — Adjust `margin_ratio` or `target_size` as needed.

---

## Usage

Run the pipeline:

```bash
python process_video.py
```

* The script will:

  * Fetch unprocessed videos from the database.
  * Extract keyframes and detect faces.
  * Save frames and cropped faces to disk.
  * Insert face metadata into the database.
  * Mark the video as processed (`IsFacesExtracted = 1`).

---

## Pipeline Workflow

1. **Fetch unprocessed videos** from the database.
2. **Decide extraction method**:

   * Short videos (< 60s): skip frames
   * Long videos: motion detection
3. **Extract keyframes** and save them.
4. **Detect faces** with RetinaFace.
5. **Crop faces** to square images and save.
6. **Store metadata** in the database (bounding box, frame number, filename).
7. **Update MediaItems** to mark as processed.

---
