# Face Recognition with OpenVINO and FAISS

This project implements a face recognition system using OpenVINO for face detection and feature extraction, and FAISS for efficient similarity search. It supports live video processing, detailed video analysis, and embedding generation for large datasets.

## Features
- Face detection using SCRFD models.
- Face recognition using AdaFace models.
- Efficient similarity search with FAISS.
- Video processing for face recognition and performance analysis.
- Embedding generation for large-scale datasets.

## Technologies Used
- **OpenVINO**: For optimized inference of face detection and recognition models.
- **FAISS**: For fast similarity search and clustering.
- **OpenCV**: For video processing and visualization.
- **Python**: Core programming language for implementation.

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/face-recognition-openvino.git
   cd face-recognition-openvino
    ```

2. **Install Dependencies:** Use pip to install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download Models:** Place the SCRFD and AdaFace models in the appropriate directories under `buffalo_l/`.

4. **Prepare Embeddings:** Generate embeddings for your dataset using the `go_through()` function in `main.py`.

5. **Run the Application:** Execute the main script for video processing:
    ```bash
    python main.py
    ```

## Usage
* Modify the `main.py` file to specify input video paths, thresholds, and output paths.
* Use the provided functions for embedding generation, similarity comparison, and video analysis.

## License
This project is licensed under the Apache 2.0 License.
