# A-Novel-Stacked-CNN-Ensemble-for-Enhanced-Feature-Extraction
This notebook reproduces the MNIST experiments from our paper:

Optimizing Handwritten Digit Recognition: A Novel Stacked CNN Ensemble for Enhanced Feature Extraction

Overview
Base models: ResNet50, VGG16, VGG19 (transfer learning from ImageNet)

Feature fusion: Penultimate layer embeddings concatenated

Meta-learner: Multinomial Logistic Regression

Training: 50 epochs per base model (to match paper learning curves)

Evaluation: Hold-out test and k-fold cross-validation (multi-seed)

Statistical analysis: Wilcoxon signed-rank test, McNemar test, 95% Confidence Intervals

Notebook Highlights
Colab-ready with one-line %pip install setup

Generates all figures and CSVs used in the paper:

training_curves_3x2.png – Accuracy/Loss curves for each backbone

feature_maps_block1conv1_vgg16.png – Early convolutional feature maps

meta_accuracy_over_iterations.png – Meta-learner iterative accuracy

model_accuracy_comparison_holdout.png – Hold-out accuracy comparison

meta_confusion_matrix.png – Confusion matrix for meta-learner

cv_boxplot.png – CV accuracies distribution

model_accuracy_comparison_cv.png – CV mean±std bar chart

Outputs saved in paper_outputs_mnist/:

Histories (history_*.csv)

Metrics (*_metrics.json)

Cross-validation summary (cv_summary.csv)

Significance test results (significance_tests.csv)

How to Run
bash
Copy
Edit
# Clone this repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# (Optional) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate     # Linux/Mac
.venv\Scripts\activate.ps1    # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Open the notebook in Jupyter or VS Code
jupyter notebook Notebook_A_MNIST_Stacked_Ensemble_50epochs.ipynb
Or run on Google Colab:

Open the .ipynb file from the repo in Colab.

Run the first cell to install dependencies.

Execute all cells (runtime may vary depending on GPU availability).

Notes
Default subset sizes (train_subset_n=500, test_subset_n=100) are for quick execution.
Increase these for full-scale runs.

50 epochs are used per backbone to replicate the curves shown in the paper.

All outputs are automatically stored in paper_outputs_mnist/ to keep the repository organized.
