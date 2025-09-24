# VisionAI - Video

VisionAI Video is a Python-based pipeline for **extracting keyframes** and **detecting faces** from videos. 
It automatically processes videos, crops faces using `RetinaFace`, and stores metadata in a `SQL Server database`.

## Features

- Fetch unprocessed video files from a `SQL Server database`  
- Extract keyframes using motion detection or frame skipping  
- Detect faces in frames using `RetinaFace`  
- Crop faces to square format with configurable margins  
- Save frames and cropped faces to disk  
- Store face metadata (bounding boxes, frame number, filename) in the database  
- Logging for monitoring progress and errors
