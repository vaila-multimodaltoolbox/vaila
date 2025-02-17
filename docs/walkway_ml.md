## 🚀 New Feature: *vailá* ML Walkway

### 👥 Created by: Abel G. Chinaglia & Paulo R. P. Santiago
**Lab:** LaBioCoM - Laboratory of Biomechanics and Motor Control
📅 **Date:** 10.Feb.2025
🔄 **Update:** 14.Feb.2025

### 🏃‍♂️ What is *vailá* ML Walkway?

*vailá* ML Walkway is a **graphical user interface (GUI)** designed to facilitate various **machine learning (ML) tasks** related to gait analysis. With this module, users can:

✅ **Process gait features** using pixel data from MediaPipe (**via Makerless 2D in *vailá***).
✅ **Train ML models** usando as extracting features e targets.
✅ **Validate ML models** after training.
✅ **Run ML predictions** using pre-trained models ou modelos novos treinados por você.

Each function is accessible through buttons in the user-friendly GUI, automating complex ML workflows with ease!

## 🎯 Running the *vailá* ML Walkway

🔹 **From the terminal:**

```bash
python vaila_mlwalkway.py
```

🔹 **From the applications menu:** Select *vailá* ML Walkway from the installed applications.

This will open a **graphical interface** where you can select different **machine learning tasks** for gait analysis.

### 🎯 What Does Each Step of *vailá* ML Walkway Do?

#### **Process Gait Features**
The **Process Gait Features** functionality uses a `.csv` file containing pixel data from MediaPipe (**via Markerless 2D in *vailá***) to calculate the necessary features for training new models using **Train ML Models** or for applying predictions of gait variables using the **Run ML Predictions** button. This functionality calculates features based on the number of steps that occurred during the task.

**Important:**  
If the feature calculations are not performed using pixel data from MediaPipe, it will not be possible to train new models with **Train ML Models**, validate trained models with **Validate ML Models**, or run predictions with **Run ML Predictions**.

---

#### **Train ML Models**
The **Train ML Models** functionality allows you to train models using data from new patients, optionally adding them to the test data available in the folder. Training is performed by selecting the `.csv` file containing the processed features from **Process Gait Features**, applied to target variables such as gait metrics typically measured by instrumented walkways like the GAITRite™ and Zeno Walkway systems.  

The selected gait variables for training include:  
- **Step Length**  
- **Stride Length**  
- **Support Base**  
- **Support Time Single Stand**  
- **Support Time Double Stand**  
- **Step Width**  
- **Stride Width**  
- **Stride Velocity**  
- **Step Time**  
- **Stride Time**  

For each gait variable, the following machine learning algorithms are applied:  
- **XGBoost**  
- **KNN (K-Nearest Neighbors)**  
- **MLP (Multilayer Perceptron)**  
- **SVR (Support Vector Regression)**  
- **Random Forest**  
- **Gradient Boosting**  
- **Linear Regression**  

In addition to saving the trained models, performance evaluation metrics are saved in a separate folder. These metrics can be used to assess which model achieved the best performance for each gait variable.  

The following metrics were used to evaluate model performance:  
- **MSE (Mean Squared Error):** Average of the squared differences between actual and predicted values.  
- **RMSE (Root Mean Squared Error):** Square root of MSE, providing an error measure in units similar to the original data.  
- **MAE (Mean Absolute Error):** Average of the absolute differences between actual and predicted values.  
- **MedAE (Median Absolute Error):** Median of the absolute differences between actual and predicted values. Less sensitive to outliers.  
- **Max Error:** Largest absolute difference between actual and predicted values.  
- **RAE (Relative Absolute Error):** Relative absolute error, comparing total absolute error to the mean absolute error relative to the mean of the actual values.  
- **Accuracy (%):** Percentage of predictions within a specific tolerance.  
- **R² (R-squared):** Coefficient of determination, measuring the proportion of variance in the data explained by the model. Values closer to 1 indicate better fit.  
- **Explained Variance:** Measures the proportion of data variance explained by the model. Similar to R² but penalizes systematic errors.  

Each gait variable has a `.csv` file containing the performance metrics for each model. Additionally, a `.png` file visualizing the performance metrics is generated for each variable, allowing for easy comparison of model performance.  

---

#### **Validate ML Models**
The **Validate ML Models** functionality is one of the most important steps after training new models. It allows you to select the directory where the trained models are saved and validate them using test data not used during training. These new data are crucial for evaluating whether the model generalizes well to unseen data. The validation data must be in the form of features, requiring processing by **Process Gait Features**. Target data for validation is also required to verify model performance.  

This step provides the same performance evaluation metrics as those used during training, enabling assessment of which model achieved the best performance for each gait variable.  

---

#### **Run ML Predictions**
The **Run ML Predictions** functionality allows you to apply gait variable predictions using pre-trained models or new models trained by you. In this functionality, you can choose which gait metrics to predict, and the output will be a `.csv` file containing the predicted values for the selected metrics.  

For additional details, refer to the main documentation.