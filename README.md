# Heart Disease Classification & Clustering

A machine learning project for the "Fundamentals of Artificial Intelligence" course at Riga Technical University. This project applies both unsupervised and supervised algorithms to the UCI Cleveland Heart Disease dataset to predict the presence of heart disease in patients.

Built with Python (Scikit-Learn, Pandas, Seaborn)

## Team members

| Name | Responsibilities | Files | Issues |
| :--- | :--- | :--- | :--- |
| **Ayma Rehman** | Data Preprocessing, ANN Implementation | `src/preprocessing.py`, `src/ann_classification.py`, `main.py` | `#1`, `#4` |
| **Dhanusha Udhayakumar** | EDA and Statistical Visualizations | `notebooks/exploration.ipynb` | `#2` |
| **Nandana Subhash** | Hierarchical Clustering Analysis | `src/hierarchical_clustering.py` | `#3` |
| **Rashid** | Model Comparison | `src/classification.py` | `#5` |
| **Ruslan Mishura** | K-Means Clustering | `src/Kmeans_clustering.py` | `#6` |


## Project Objectives
* **Preprocessing:** Handle missing values, outliers, and feature scaling.
* **Unsupervised Learning:** Use Hierarchical Clustering and K-Means to identify natural groupings in patient data.
* **Supervised Learning:** Train an Artificial Neural Network (ANN) and two other classifiers to predict diagnosis.

## Tech Stack

| Library | Purpose |
| :--- | :--- |
| **pandas** | Data loading, cleaning, duplicate removal, and structured output tables |
| **scikit-learn** | Preprocessing, ANN implementation, and model evaluation |

## Project Structure
```text
heart-disease-ml/
├── data/
│   └── heart.csv                   # UCI Cleveland Dataset
├── notebooks/
│   └── exploration.ipynb           # Initial EDA and visualizations
├── src/
│   ├── preprocessing.py            # Cleaning, outliers, and scaling
│   ├── Kmeans_clustering.py        # K-Means logic
│   ├── hierarchical_clustering.py  # Hierarchical logic
│   ├── ann_classification.py       # ANN implementation
│   └── classification.py           # Comparison of supervised models
├── tests/
│   └── test_data.py                # Validation for data shapes/types
├── README.md
├── requirements.txt
└── main.py                         # Entry point to run the full pipeline
```

## To set up on your end

**Requirements:** Python 3.10 or higher.

### 1. Clone the repository and set up the environment:

```bash
git clone https://github.com/AymaRehman/heart-disease-ml.git
cd heart-disease-ml
```

### 2. Create and activate the virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
On Windows: venv\Scripts\activate

### 3. Install dependencies

```bash
pip install -r requirements.txt
```
### 4. Download the Dataset
Download the dataset from Kaggle, rename it to `heart.csv` and place it in the `data/` folder locally on your computer:  
[Heart Disease Dataset on Kaggle](https://www.kaggle.com/datasets/ritwikb3/heart-disease-cleveland?resource=download)
> **Note:** We are using the Kaggle version as it is pre-formatted for Python.   
> **Original Source:** [UCI Machine Learning Repository - Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)

### 5. Run the preprocessing:
```bash
python src/preprocessing.py
```

### 6. Run the main analysis:
```bash
python main.py
```

## Git Workflow
Each team member works on their own branch and merges to `main`.

#### Initial Setup (do only once after cloning):   
`git checkout -b branch-name`

#### Syncing (Update your branch with the latest `main`): 
```bash
1. git checkout main
2. git pull
3. git checkout your-branch-name
4. git merge main
```
#### Saving & Submitting Work
```bash
1. git add .
2. git commit -m "Your commit message"
3. git push origin your-branch-name 
```

> After pushing changes to your branch, go to:  
> [Repository on GitHub](https://github.com/AymaRehman/heart-disease-ml), click "Compare & pull request", and submit your changes for review.


## Note: 
- Keep in mind, every time you come back to work on this project, you need to reactivate the venv first: source venv/bin/activate 

- Never push directly to the `main` branch - always push to your own branch, as simultaneous pushes can cause conflicts.

- Ensure your branch is always up to date with `main`.
