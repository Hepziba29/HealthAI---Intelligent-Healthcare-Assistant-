# HealthAI---Intelligent-Healthcare-Assistant-

🏥 HealthAI – Intelligent Healthcare Assistant

HealthAI is an AI-powered healthcare assistant that predicts diseases based on user symptoms and provides treatment recommendations. It uses IBM Granite models from Hugging Face for natural language understanding and is deployed using Gradio for an interactive user interface.


📌 Project Overview

Goal: Make healthcare guidance accessible, fast, and easy-to-understand

Model Used: ibm-granite/granite-3.2-2b-instruct

Platform: Google Colab (T4 GPU)

Frontend: Gradio Web App
✨ Features

✅ Symptom-Based Disease Prediction – ML-driven predictions for possible conditions
✅ AI Treatment Recommendations – Suggests next steps (self-care/consult a doctor)
✅ Conversational Interface – Users can chat naturally with the AI
✅ Fast & Accessible – Runs in Google Colab with shareable Gradio link
✅ Scalable – Can be enhanced with additional datasets & models

🏗️ Project Architecture

flowchart TD
    A[User Inputs Symptoms] --> B[Gradio Frontend]
    B --> C[Python Backend in Colab]
    C --> D[IBM Granite Model on Hugging Face]
    D --> E[Prediction + Treatment Plan]
    E --> F[Gradio Output Display]


🛠️ Tech Stack

Programming Language: Python 3.8+

Libraries: transformers, torch, gradio

Platform: Google Colab

AI Model: IBM Granite (Hugging Face)

⚙️ Setup Instructions

Prerequisites

Google Colab account

Hugging Face account (optional, for API key if needed)


Installation

1. Open the provided Colab Notebook


2. Set runtime to GPU (T4)


3. Install dependencies:

!pip install transformers torch gradio -q


4. Run all cells to launch the app

▶️ How to Run

1. Open the notebook in Google Colab


2. Execute all cells in order


3. Launch the Gradio interface


4. Enter symptoms (e.g., "fever, cough, fatigue")


5. Get predicted disease and treatment recommendations


🚀 Future Enhancements

Add statistical probability calculation (logistic regression, chi-square)

Include explainability using SHAP or LIME

Visualize top predicted diseases in Tableau dashboard

Deploy permanently on Hugging Face Spaces or Streamlit Cloud
HealthAI – Julia + MongoDB + ML (Simple README)

This project loads patient data, stores it in MongoDB, and uses Julia to build a simple machine-learning model that predicts Disease Risk (Low / Medium / High).


---

1. What this project does

✔ Loads patient_data.csv
✔ Cleans and prepares data
✔ Creates a new column: Disease_Risk
✔ Stores all patient records in MongoDB
✔ Trains a Decision Tree model using MLJ
✔ Shows accuracy & confusion matrix
✔ Saves model results back to MongoDB


---

2. Requirements

Install:

Julia 1.8+

MongoDB (local or Atlas)

Julia packages:


using Pkg
Pkg.add(["CSV","DataFrames","CategoricalArrays","MLJ","MLJModels","DecisionTree","Statistics","Mongo"])

Place your file:

patient_data.csv
HealthAI.jl


---

3. How to run

Open terminal in your project folder:

julia HealthAI.jl

Or inside Julia REPL:

include("HealthAI.jl")


---

4. MongoDB Setup

Local MongoDB:
Use default URI:

mongodb://localhost:27017

Atlas (Cloud):
Replace the URI in your script with your Atlas connection string.


---

5. Expected Output

When your script runs, you will see:

First few rows of patient data

Confirmation that patient records were inserted into MongoDB

Model training logs

Accuracy score

Confusion matrix

Message confirming results stored in MongoDB


Example:

Accuracy = 0.83
Successfully inserted 200 records
Model metrics saved to MongoDB


6. File Structure

/HealthAI_Project
  patient_data.csv
   HealthAI.jl
    README.md
  Tableau for patient data set
  
  

---

7. Summary

This project shows a complete end-to-end pipeline:

1. Data →


2. Store →


3. Train Model →


4. Evaluate →


5. Save Results




