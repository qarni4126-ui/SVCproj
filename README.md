# 🍷 Wine Classification using SVM and GridSearchCV

This project uses the **Wine dataset** from `sklearn.datasets` to classify different types of wines based on their chemical composition.  
An **SVM (Support Vector Machine)** model is optimized using **GridSearchCV** to find the best hyperparameters for maximum accuracy.

---

## 📊 Dataset
- Source: `sklearn.datasets.load_wine()`
- Samples: 178  
- Features: 13 (chemical properties such as alcohol, magnesium, flavanoids, etc.)  
- Classes: 3 wine cultivars (`class_0`, `class_1`, `class_2`)

---

## ⚙️ Project Workflow
1. Load and explore the Wine dataset  
2. Split data into training and testing sets  
3. Train an **SVM classifier (SVC)**  
4. Perform **GridSearchCV** for hyperparameter tuning  
5. Evaluate model accuracy and generate a classification report  
6. Visualize Grid Search results (heatmap and line plot)

---

## 🧠 Model and Parameters
Parameters tuned with GridSearchCV:
- `C`: [0.1, 1, 10, 100]  
- `kernel`: ['linear', 'rbf', 'poly']  
- `gamma`: ['scale', 'auto', 0.01, 0.1, 1]  
- `degree`: [2, 3, 4]

Best parameters are printed after training using:
```python
grid.best_params_
grid.best_score_
