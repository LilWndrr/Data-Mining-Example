# 📍 Location History Data Mining and Analysis Project

This project analyzes personal movement activities using Google Location History data (Google Takeout `.json` files). It processes raw data to extract meaningful features, performs statistical analysis, implements K-Means clustering, and uses Decision Tree classification for activity prediction.

---

## 📂 Project Structure

The project consists of 5 main files:

| File | Description |
|------|-------------|
| **`Хронология.json`** | Raw location history data from Google Takeout (input file) |
| **`reformating.py`** | Data preprocessing script that cleans JSON, groups activities (`WALKING`, `BUS`, `CAR`), calculates duration/speed, and generates `my_location_features.csv` |
| **`dataMiningMath.py`** | Statistical analysis script that creates distribution charts, calculates class balance, and reports speed/distance statistics |
| **`kMeans.py`** | Clustering script that standardizes data and applies K-Means algorithm to create 5 clusters, visualized with scatter plots |
| **`decisionTree.py`** | Classification script that trains a Decision Tree model to predict activity types and displays results with a confusion matrix |
<img width="1268" height="997" alt="image" src="https://github.com/user-attachments/assets/714902d8-bd29-4ccb-8b65-ecdf16b06ac3" />
---

## 🛠️ Requirements

Make sure you have Python installed. Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

Or use the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## 🚀 Installation and Usage

Follow these steps to run the project:

### Step 1: Data Preparation

Add your raw data file to the project folder.

> **Important:** The JSON file must be named exactly `Хронология.json`. If your file has a different name, either rename it or update the file path in `reformating.py`.

### Step 2: Data Preprocessing

Process the data and convert it to CSV format:

```bash
python reformating.py
```

*This will create a new file named `my_location_features.csv` in the project folder.*
<img width="854" height="770" alt="image" src="https://github.com/user-attachments/assets/e74df72c-d572-49be-ae11-22c6ab94753c" />


### Step 3: Analysis and Modeling

After the CSV file is created, you can run the following scripts in any order:

#### A. Statistical Analysis

View activity distributions and speed box plots:

```bash
python dataMiningMath.py
```

#### B. Clustering

Explore how activities cluster based on duration and distance:

```bash
python kMeans.py
```

#### C. Classification

See how accurately the model predicts activities and view the confusion matrix:

```bash
python decisionTree.py
```

---

## 📊 Outputs and Visualizations

When you run the scripts, you'll get:

**Console Outputs:**
- Model accuracy rates
- Average values for each cluster
- Activity percentages

**Visualizations:**
- **Bar Chart:** Distribution of activity counts
- **Box Plot:** Speed distributions by activity type
- **Scatter Plot:** K-Means clusters plotted by duration and distance
- **Confusion Matrix:** Heatmap showing which activities the Decision Tree model confuses with each other

---

## 📝 Notes

- `reformating.py` excludes `FLYING` and `IN_FERRY` activities from analysis
- Metro, Train, and Tram data are merged under the `RAIL` category
- Time periods (Morning, Afternoon, Evening, Night) are automatically assigned based on hour ranges
- The project processes personal location data - ensure you comply with privacy regulations when sharing or deploying

---

