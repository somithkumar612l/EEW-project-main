# 🌏 Earthquake Early Warning (EEW) System

The **Earthquake Early Warning (EEW) System** is an AI-based project that helps
predict the **earthquake epicenter (origin location)** using seismic station data
and Machine Learning.

This system shows **how AI + Web + Maps** can be used for **disaster management**
in a simple and understandable way.

---

## 🔗 Live Website
👉 **Website (Frontend):**  
https://eew-system.netlify.app

👉 **GitHub Repository:**  
https://github.com/rajkumar-rgb/EEW-project

---

## 🎯 What is the main goal of this project?

The main goal is to:
- Collect **arrival-time data** from seismic stations
- Use a **Machine Learning model**
- Predict **where the earthquake started (epicenter)**
- Show the result clearly on a **map**

This helps in **early awareness** and understanding earthquake behavior.

---

## 🧠 How the System Works (Step by Step)

### 🔹 Step 1: Data Collection
- Seismic stations record earthquake signals
- Each station provides:
  - Station latitude
  - Station longitude
  - Arrival time of the earthquake signal
- The data is taken from **Japan Meteorological Agency (JMA)**

---

### 🔹 Step 2: Data Cleaning
- Remove incorrect or missing data
- Select only useful earthquake events
- Prepare clean data for Machine Learning

This step improves **accuracy and reliability**.

---

### 🔹 Step 3: Machine Learning Model
- A **regression-based ML model** is trained
- Input to the model:
  - Station coordinates
  - Arrival-time information
- Output from the model:
  - Predicted epicenter (latitude & longitude)

The model learns patterns from past earthquakes.

---

### 🔹 Step 4: Backend (FastAPI)
- The trained model is loaded into a **FastAPI backend**
- Backend:
  - Receives input data
  - Sends it to the ML model
  - Returns the predicted epicenter
- Backend is fast and lightweight

---

### 🔹 Step 5: Frontend (Website)
- Built using **HTML, CSS, and JavaScript**
- User enters arrival-time related inputs
- Prediction result is displayed on:
  - A **map**
  - With location markers

This makes the system **user-friendly and visual**.

---

### 🔹 Step 6: Map Visualization
- Uses **Leaflet.js**
- Shows predicted epicenter on the map
- Easy to understand for non-technical users

---

## Dataset

The full earthquake dataset used in this project is large and cannot be
uploaded directly to GitHub due to file size limitations.

Therefore, the dataset is hosted externally. The links are provided below:

Dataset File 1:
https://drive.google.com/file/d/1FcO-pYkeWxe3fq7eougFuFEjWvtJiYD4/view?usp=drive_link

Dataset File 2:
https://drive.google.com/file/d/14ydiSR3SbDk6QtViUvuwCL1LHbvNaREx/view?usp=drive_link

Dataset File 3:
https://drive.google.com/file/d/11qWJJqKdHXWyPtSopWXOYZ2fHwTp7b-t/view?usp=drive_link

Note:
Only a trained model and sample data are included in this repository
for easy testing and understanding.

---

## 🚀 Key Features
- AI-based earthquake epicenter prediction
- Interactive map visualization
- Clean and simple user interface
- Fast response using FastAPI
- Real-world dataset usage (JMA)

---

## 🗂 Project Structure

<img width="1024" height="1536" alt="ChatGPT Image Dec 16, 2025, 08_14_28 PM" src="https://github.com/user-attachments/assets/999e9d29-b6b2-490f-92db-9898805e8a0b" />

---

## ⚙️ How to Run the Project Locally

### ▶ Run Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Backend will run at:
http://127.0.0.1:8000

▶ Run Frontend
frontend/index.html

📊 Final Output

- Predicts earthquake epicenter location  
- Displays the result clearly on an interactive map  
- Helps users understand the earthquake origin location

## Academic Details

- Project Type: Mini Project  
- Degree: B.Tech  
- Branch: Computer Science Engineering (AI & ML)  
- Purpose: Academic & educational learning  

## Author

Raj Kumar Paswan  
B.Tech – Computer Science Engineering (Artificial Intelligence & Machine Learning)

