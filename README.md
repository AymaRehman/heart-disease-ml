# Heart Disease Classification & Clustering

A machine learning project for the "Fundamentals of Artificial Intelligence" course at Riga Technical University. This project applies both unsupervised and supervised algorithms to the UCI Cleveland Heart Disease dataset to predict the presence of heart disease in patients.

Built with Python (Scikit-Learn, Pandas, Seaborn)

## Team members
| Name | Role |
| :--- | :--- |
| Ayma Rehman | |
| Dhanusha Udhayakumar |  |
| Nandana Subhash |  |
| Rashid |  |

## Project Objectives
* **Preprocessing:** Handle missing values, outliers, and feature scaling.
* **Unsupervised Learning:** Use Hierarchical Clustering and K-Means to identify natural groupings in patient data.
* **Supervised Learning:** Train an Artificial Neural Network (ANN) and two other classifiers to predict diagnosis.

## Project Structure
```text
heart-disease-ml/
├── data/
│   └── heart.csv            # UCI Cleveland Dataset
├── notebooks/
│   └── exploration.ipynb    # Initial EDA and visualizations
├── src/
│   ├── preprocessing.py     # Cleaning, outliers, and scaling
│   ├── clustering.py        # Hierarchical and K-Means logic
│   └── classification.py    # ANN and other supervised models
├── tests/
│   └── test_data.py         # Validation for data shapes/types
├── README.md
├── requirements.txt
└── main.py                  # Entry point to run the full pipeline
```

## To set up on your end

**Requirements:** Python 3.10 or higher.

### 1. Clone the repository and set up the environment:

```
git clone https://github.com/AymaRehman/heart-disease-ml.git
cd heart-disease-ml
```

### 2. Create and activate the virtual environment
```
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
On Windows: venv\Scripts\activate

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Download the dataset
Download the dataset from Kaggle and place it in the data/ folder locally on your computer  
[UC Irvine Heart Disease Dataset on Kaggle](https://www.kaggle.com/datasets/ritwikb3/heart-disease-cleveland?resource=download)

### 5. Run the analysis:
```
python main.py
```

## Git Workflow
Each team member works on their own branch and merges to `main`.

Setup (do only once after cloning): 
`git checkout -b branch-name`

Syncing: 
```
1. git checkout main
2. git pull
3. git checkout your-branch
4. git merge main
```

## Note: 
- Keep in mind, every time you come back to work on this project, you need to reactivate the venv first: source venv/bin/activate 

- Never push directly to the `main` branch - always push to your own branch, as simultaneous pushes can cause conflicts.


