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

### 2. Run the Streamlit App
```bash
docker-compose up --build
```

### 3. Access the App
Open your browser and go to:
```
http://localhost:8501
```

---

### ⛔ Stop Running Containers
If the original Git Bash window is **still open**, make sure it’s active (click on it), then press `Ctrl` + `C`.

If the original Git Bash window has already been **closed**, perform the following:

1. Open Git Bash on your machine (this can be done from any directory).

2. Find the container ID:
    ```bash
    docker ps
    ```
3. Stop the container:
    ```bash
    docker stop <container_id>
    ```

Alternatively, Docker Desktop can be used to stop and delete images.

---

### 🔄 Restarting a Stopped Container
If you want to rerun the docker container, you can do so without having to perform the 3 listed steps again.

1. Open Git Bash on your machine (This can be done from any directory).

2. Find the container ID:
    ```bash
    docker ps -a
    ```
    The container should be named dsa3101-project-streamlit-app

3. Start the container
    ```bash
    docker start <container_id>
    ```
---
### 🔧 Troubleshooting

- **Port Already in Use?** Run:
  ```bash
  docker stop $(docker ps -q)
  ```

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
This command automatically opens htpp://localhost:8501 in your browser.

---
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
