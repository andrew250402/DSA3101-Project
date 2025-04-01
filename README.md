# Streamlit App
This repository contains a Streamlit application that can be run inside a Docker container or the Python Virtual Environment (venv).

## Prerequisites
- Install [Docker](https://docs.docker.com/get-docker/)
- Python 3.9+ (optional, only required if you want to run the app without Docker)

## 🚀 Setting up Docker

### 1. Clone the Repository
```bash
git clone https://github.com/andrew250402/DSA3101-Project.git
cd DSA3101-Project
```

### 2. Build the Docker Image
```bash
docker build -t dsa3101-project .
```

### 3. Run the Streamlit App
```bash
docker-compose up --build
```

### 5. Access the App
Open your browser and go to:
```
http://localhost:8501
```

---

### ⛔ Stop Running Containers
Find the container ID:
```bash
docker ps
```
Stop the container:
```bash
docker stop <container_id>
```
Alternatively, Docker Desktop can be used to stop and delete images.

---

### 🔧 Troubleshooting

- **Port Already in Use?** Run:
  ```bash
  docker stop $(docker ps -q)
  ```
- **Changes Not Reflecting? (for developers)** Use `-v $(pwd):/app` to mount local files for live code updates.

---
## 🚀 Setting up Python Virtual Environment (venv)

### 1. Clone the Repository
```bash
git clone https://github.com/andrew250402/DSA3101-Project.git
cd DSA3101-Project
```
### 2. Create and Activate venv
```bash
python -m venv myenv
myenv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run main.py
```
This command automatically opens htpp://localhost:8501

### 🔧 Troubleshooting

- `ModuleNotFoundError` → Reinstall dependencies:
    ```bash
    pip install -r requirements.txt
    ```
- **Port conflict** → Use a different port:
    ```bash
    streamlit run main.py --server.port 8502
    ```
- **Reset environment**:
    ```bash
    deactivate
    rm -rf venv/
    ```

- `Python was not found` error → Use py
    ```bash
    py -m venv myenv
    ```
- `myenvScriptsactivate: command not found` error → Use forward slashes
    ```bash
    myenv/Scripts/activate
    ```
