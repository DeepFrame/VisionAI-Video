# VisionAI - Video

VisionAI Video is a **Python-based video processing pipeline** for **keyframe extraction**, **face detection**, **face embedding**, and **person clustering**. It automatically processes videos, crops faces using **RetinaFace**, generates embeddings via **FaceNet**, tracks faces across frames with **DeepSort**, and stores metadata in a **SQL Server database**.

The pipeline supports GPU acceleration via **TensorFlow** for embeddings and ensures high-quality, representative face selection.

---

## Features

* Fetch unprocessed videos from a `SQL Server` database
* Extract keyframes using:

  * Motion detection-based scoring
  * Frame skipping for short videos
* Detect faces in frames using `RetinaFace`
* Crop faces to **square format** with configurable margins
* Track faces across frames using `DeepSort`
* Select the **sharpest face** per tracked person for storage and embedding
* Compute **face embeddings** using `FaceNet` (via `keras-facenet`)
* Cluster embeddings with `DBSCAN` for person identification
* Save keyframes and cropped faces to disk
* Store face metadata (bounding boxes, frame number, filename) and embeddings in the database
* Assign **portrait faces** for each person using **medoid/sharpest face selection**
* Logging for monitoring progress and errors

---

## Technical Stack

### Programming Language

* Python 3.8.20

### Libraries & Models

| Category                | Libraries / Models                                                          | Purpose                                                                               |
| ----------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Video I/O & Processing  | `opencv-python (cv2)`                                                       | Load videos, extract frames, resize, crop faces                                       |
| Face Detection          | `retinaface`                                                                | Detect faces with bounding boxes                                                      |
| Face Embedding          | `tensorflow`, `keras-facenet`                                               | Generate 512-dim embeddings per face                                                  |
| Tracking                | `deep_sort_realtime`                                                        | Track faces across keyframes for consistent face selection                            |
| Clustering & Similarity | `scikit-learn` (`DBSCAN`, `cosine_similarity`)                              | Cluster faces and assign person IDs                                                   |
| Database                | `pyodbc`                                                                    | Connect and update SQL Server database (`dbo.Faces`, `dbo.Persons`, `dbo.MediaItems`) |
| Utilities               | `numpy`, `os`, `pathlib`, `shutil`, `logging`, `tabulate`, `dotenv`, `json` | File management, logging, environment config, and data handling                       |

---

## Database Structure

* **SQL Server Tables**

  * `MediaItems` — Tracks videos and extraction status
  * `MediaFile` — Stores video file paths and metadata
  * `Faces` — Stores bounding boxes, cropped face filenames, embeddings, frame numbers, and person assignments
  * `Persons` — Stores aggregated person info and portrait face references

---

## Pipeline Overview

1. **Fetch Videos**

   * Retrieve unprocessed videos from SQL Server where `IsFacesExtracted=0`.

2. **Keyframe Extraction**

   * Extract keyframes using motion detection for long videos or frame skipping for short videos.

3. **Face Detection & Cropping**

   * Detect faces in keyframes using **RetinaFace**.
   * Crop faces to square format with optional margin.

4. **Face Tracking & Sharpest Face Selection**

   * Track faces across frames using **DeepSort**.
   * Collect all cropped faces for each track (i.e., person).
   * Compute **sharpness scores** (Laplacian variance) for each face.
   * Select the **sharpest face** as the representative for embedding and database storage.

5. **Face Embedding**

   * Generate 512-dimensional embeddings using **FaceNet**.
   * Store embeddings in the `Faces` table.

6. **Clustering**

   * Cluster unassigned embeddings using **DBSCAN**.
   * Assign person IDs and update the database.

7. **Thumbnail & Portrait Management**

   * Save representative faces in `Faces_Extracted_RetinaFace`.
   * Group faces by person under `ByPerson` folder.
   * Assign **medoid/sharpest face** as portrait for each person.

8. **Database Update**

   * Update `Faces` table with bounding boxes, embeddings, and person IDs.
   * Update `Persons` table with portrait face references.

9. **Logging**

   * Detailed logs saved in `logs/visionai_video.log`.
   * Optional per-function logs for thumbnail processing.

---

## GPU Support

* TensorFlow detects GPUs automatically.
* Memory growth is enabled to prevent allocation issues.
* Falls back to CPU if GPU is unavailable.

---

## Environment Variables

* `SQL_CONNECTION_STRING` — Connection string to SQL Server

Example `.env`:

```
SQL_CONNECTION_STRING=Driver={SQL Server};Server=YOUR_SERVER;Database=YOUR_DB;UID=USER;PWD=PASSWORD
```

---

## Usage

* Automatically processes unprocessed videos.
* Extracts faces, generates embeddings, clusters, and assigns persons.
* Updates SQL Server with face metadata and embeddings.
* Saves cropped faces and groups them by person.
* Assigns portrait faces for each person.

---

## Adjustable Parameters

* **Face margin ratio:** Default `0.2` (adds context around detected face)
* **Keyframe motion threshold:** Default `28.0`
* **DBSCAN clustering:** `eps=0.35`, `min_samples=3`
* **Short video frame skip:** Default `10` frames

---

## Notes

* Supports both **short videos** (frame skip) and **long videos** (motion detection + DeepSort tracking).
* Handles missing thumbnails and embeddings via `reprocess_media_missing_faces()` and `check_thumbnails()`.
* Only the **sharpest face per tracked person** is stored, improving embedding quality, clustering accuracy, and storage efficiency.

---
