# **AutoClean AI** - ***Clean Faster. Effort Less.***

---

## **Description**

## **Project structure**
- `app.py`: main Streamlit python script u
- `tool.py`: Python script that consists of data cleaning functions and visualisation scripts
- `load_data.py`: script to handle file uploading
- `config.toml `: configuration settings for the app 
- `requirements.txt`: a list of all packages required to run the project
- `README.md`: markdown file

## **Features** 
### ✨ *Smart Data Type Recognition* – Automatically detects numerical and categorical data.

### ✨ *Intelligent Number Extraction* – Extracts numeric values from messy text columns.
  
| Messy text    | Extracted number |  
|--------------|-----------------|  
| 400 dollars  | 400             |  
| $380 per week | 380            |  
| 200pw CHEAP  | 200             |  
| only 450$    | 450             | 
<br>

### ✨ *Year Extraction* - Extracts the year from date columns.  

| Date        | Year |  
|------------|------|  
| 21/09/1999 | 1999 |  
| 03/08/2000 | 2000 |  
| 08/11/2000 | 2000 |  
| 18/07/2003 | 2003 |  
<br>

### ✨ *Auto-Generated Visualizations* – Instantly creates the most suitable charts for your data.

AutoClean AI intelligently categorizes data and selects appropriate visualization techniques for clear and insightful analysis.

| **Category**                | **Condition**                                        | **Chart type**     | **Description** |
|-----------------------------|------------------------------------------------------|--------------------|----------------|
| **Single variable distribution** | | | |
|  **Categorical variables**  | Unique values ≤ 4                                 | **Pie chart**      | Displays proportion of each category |
|                              | Unique values > 4 & ≤ 15                          | **Bar chart**      | Shows frequency distribution |
|  **Numerical variables**    | Unique values ≤ 10                                | **Bar chart**      | Highlights specific discrete values |
|                              | Unique values > 10                                | **Histogram**      | Displays data distribution over a range |
| **Two-variable relationship** | | | |
|  **Boxplot**                | One variable categorical, one numerical           | **Boxplot**        | Shows distribution and outliers |
|  **Bar chart** | One variable is *year* or *month*          | **Bar chart**      | Displays count trends over time. |
<br>

### ✨ *Standardized Formatting* – Removes excess spaces, corrects typos, and ensures consistency.
<br>

### ✨ *Missing Value Handling* – Detects and fills missing values intelligently.

AutoClean AI employs a **strategic, data-driven approach** to handling missing values, ensuring data integrity while minimizing bias. The method adapts to different data types and distributions for optimal imputation.

| **Data type**       | **Strategy** |
|---------------------|----------------------------------------------------|
| **High-missing columns** | Columns with **>50% missing values** are removed to prevent unreliable analysis. |
| **Numerical data** | Uses **mean** for normally distributed data and **median** for highly skewed data to avoid distortion. |
| **Categorical data** | Missing values are filled with **"Unknown"** to preserve all records while maintaining interpretability. |

**Why is this approach powerful?**
- **Higher accuracy**: Skewness is detected to decide between **mean or median**, preventing extreme values from distorting the dataset.
- **Preserves valuable data**: Missing values are intelligently imputed to **retain critical information**.
By leveraging **smart imputation techniques**, AutoClean AI enhances data quality, ensuring **reliable analysis and accurate machine learning models**.
<br>

### ✨ *Duplicate Data Removal* – Identifies and removes duplicate records.

### ✨ *Outlier Detection* – Detect and removes anomalies using Isolation Forest model
This tool employs **Isolation Forest**, an **unsupervised Machine Learning algorithm** specifically designed for **highly accurate and efficient outlier detection**. Unlike traditional methods like **Z-score, IQR, or Standard Deviation**, which rely on predefined statistical thresholds, **Isolation Forest intelligently isolates anomalies by learning patterns from the data itself**

**How it works?*
| **Step** | **Process** | **Why It’s Powerful?** |
|---------|------------|------------------------|
| 📊 **Detect Outliers** | **Isolation Forest** is trained on all numerical columns, learning the underlying structure of the data. | Instead of setting arbitrary thresholds, the model *learns* what constitutes an anomaly. |
| 🔎 **Assign Anomaly Score** | Each data point receives a score based on how easy it is to isolate. | Outliers are detected **without assumptions about normal distribution**. |
| 🗑 **Remove Outliers** | Points classified as anomalies (approx. **5% of the dataset**) are removed. | More precise than traditional methods, which often misclassify valid data. |

---

### ⚖ **Comparison with Standard Methods**
| **Method** | **How It Works** | **Limitations** |
|------------|-----------------|-----------------|
| **Z-score** | Flags values beyond 3 standard deviations from the mean. | Assumes a **normal distribution**, leading to errors in skewed data. |
| **IQR (Interquartile Range)** | Removes values outside **Q1 - 1.5×IQR** and **Q3 + 1.5×IQR**. | **Ineffective for large datasets** with non-Gaussian distributions. |
| **Standard Deviation** | Identifies outliers as points far from the mean. | Sensitive to **extreme values**, causing misclassification. |
| **Isolation Forest (Used in AutoClean AI)** | Uses **machine learning** to detect anomalies based on how easily a data point is isolated. | ✅ **Works on any distribution** ✅ **More accurate on large datasets** ✅ **Reduces false positives** |

---

### Why it is superior**
✔ **No Manual Thresholds** – Unlike Z-score or IQR, Isolation Forest **adapts dynamically** to different datasets.  
✔ **Handles Any Data Type** – Works **even for skewed or non-Gaussian distributions**.  
✔ **More Reliable for Large Datasets** – Traditional methods fail with **big data**, while AutoClean AI efficiently processes millions of records.  
