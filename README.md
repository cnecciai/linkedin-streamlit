# 🧠 Who's A User? - Predicting LinkedIn Usage  

## 🚀 Overview  
This project is a **machine learning application** that predicts whether an individual is a **LinkedIn user** based on demographic and socioeconomic factors. The prediction model is built using **Logistic Regression**, trained on **Pew Research data**, and deployed with **Streamlit** for interactive user input.  

📌 **Prepared By:** Clark P. Necciai Jr.  
📌 **For:** Dr. Gregory Lyon - Programming II  
📌 **Task:** Logistic Regression Model - LinkedIn User  
📌 **Live App:** [Try it here!](https://linkedin-app-cnecciai.streamlit.app/)  

## 📖 Table of Contents  
- [Overview](#-overview)  
- [How It Works](#-how-it-works)  
- [Model Details](#-model-details)  
- [Features](#-features)  
- [Example Prediction](#-example-prediction)  
- [License](#-license)  

## 🔍 How It Works  
This **Streamlit application** allows users to input demographic information such as **income, education, marital status, gender, and age**. The **logistic regression model** then predicts:  
✅ Whether the person is likely a **LinkedIn user** or **not**  
✅ The **probability** associated with the prediction  

The model is trained on a **cleaned dataset** derived from the **Pew Research social media usage survey**.  

To interact with the application, visit:  
🔗 **[Who's A User? - Live App](https://linkedin-app-cnecciai.streamlit.app/)**  

## 🏆 Model Details  

### 📌 Dataset  
- **Data Source:** Pew Research - Social Media Usage  
- **25 continuous variables** (ratings of social media usage)  
- **5,455 observations**  

### 📌 Data Preprocessing  
- **Feature selection:** Extracted relevant demographic variables  
- **Handling missing values:**  
  - Imputed using **median values** for skewed distributions  
  - Dropped rows with missing age/income/education data  
- **Categorical transformations:** Converted categorical variables to numeric format  

### 📌 Logistic Regression Model  
- **Target variable:** LinkedIn usage (binary classification)  
- **Features:**  
  - **Income Level** (1-9)  
  - **Education Level** (1-8)  
  - **Parental Status** (Yes/No)  
  - **Marital Status** (Yes/No)  
  - **Gender** (Female/Not Female)  
  - **Age** (Continuous)  
- **Train/Test Split:** 80% Training | 20% Testing  
- **Classification Algorithm:** Logistic Regression (Balanced Class Weights)  

## 🌟 Features  
✅ **Interactive UI** powered by **Streamlit**  
✅ **Machine Learning Model** trained on real data  
✅ **Dynamic probability output**  
✅ **Progress bar animation** for enhanced UX  
✅ **Video demonstration of the app in action**  

## 🎥 Example Prediction  

### 🔹 User Input:  
- **Income:** $100,000+  
- **Education:** Postgraduate Degree  
- **Marital Status:** Married  
- **Parental Status:** No  
- **Gender:** Female  
- **Age:** 42  

### 🔹 Model Prediction:  
✔ **Prediction:** **LinkedIn User**  
✔ **Probability:** **85.3%**  

🎬 **Watch a demo video inside the application!**  

## 📜 License  
This project is licensed under the **MIT License**.  

📌 **Author:** Clark P. Necciai Jr.  
📌 **Date:** February 27, 2024  
📌 **Live App:** [Try it here!](https://linkedin-app-cnecciai.streamlit.app/)  
