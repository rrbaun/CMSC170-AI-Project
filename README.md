# Neural Three - AI Project

**Proposed Dataset:** The Number of Registered Deaths by Age Group, Sex, Region of Usual Residence and Cause of Death (Philippines: 2024)

**Project Details:** The project aims to analyze regional mortality distribution patterns across the Philippines in 2024 and classify public health risk tiers based on demographic factors (age group, sex) and causes of death by applying and comparing three from-scratch AI/ML methods—**K-Means Clustering**, **K-Nearest Neighbors (KNN)**, and **Perceptron**

---

**Members**  
Perez, Antonette Mae  
Iloso, Jazmin Joy  
Baun, Ryla Sofia Ysabel

---

## Requirements & Setup

### Prerequisites

- **Python 3.8+**
- **NumPy** (arrays and basic linear algebra only)
- **Matplotlib** (for generating visualization plots)

_(No high-level machine learning or data libraries such as `pandas`, `scikit-learn`, `PyTorch`, or `TensorFlow` are used.)_

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/rrbaun/CMSC170-AI-Project.git
   cd CMSC170-AI-Project
   ```
2. Install the allowed dependencies:
   ```bash
   pip install numpy matplotlib
   ```
3. Verify that `data/CMSC170-Data.csv` is present in the `data/` directory.

---

## How to Run & Reproduce Results

Run the main entry script from the root directory of the repository:

```bash
python main.py
```

Upon execution, the script loads and cleans the dataset, prints the before-and-after dataset dimensions (`928 x 65` to `920 x 62`), and prompts you to choose a method:

```text
Select an algorithm to run:
[1] K-Means Clustering
[2] K-Nearest Neighbor
[3] Perceptron
Enter choice (1, 2, 3):
```

### Reproduce Our Results

#### Option `[1]` — K-Means Clustering (Unsupervised)

- **Setup:** Evaluates candidate cluster counts ($k = 2$ to $25$) using Euclidean distance and iterative centroid updates on the 920 unlabeled feature vectors.
- **Selected Configuration:** $k = 12$ (converges at iteration 58).
- **Expected Result:** Overall **Silhouette Score = `0.6992`**, isolating 12 distinct epidemiological profiles.

#### Option `[2]` — K-Nearest Neighbors (Supervised)

- **Setup:** Uses an **80:20 stratified train-test split** (`737` training observations, `183` testing observations) based on the median mortality threshold, testing neighborhood sizes $k \in \{3, 5, 7, 9\}$ with Euclidean distance.
- **Selected Configuration:** $k = 5$.
- **Expected Result:** **Macro F1-Score = `0.9836`** (`180 / 183` correct predictions; `91 / 91` Low Mortality Impact and `89 / 92` High Mortality Impact).

#### Option `[3]` — Perceptron (Supervised Linear Baseline)

- **Setup:** Uses the same **80:20 stratified train-test split**, trained with learning rate $\eta = 0.1$ for up to `100` epochs.
- **Expected Result:** **Test Accuracy = `65%`**, **Macro F1-Score = `0.6170`** (`103 / 157` evaluated records; `40` false positives and `14` false negatives). The model never reaches zero training errors (minimum `212` errors), demonstrating that High vs. Low mortality segments are not linearly separable by a single hyperplane.

---
