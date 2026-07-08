# Stacked CNN Ensemble for Handwritten Digit Recognition

[![DOI](https://img.shields.io/badge/DOI-10.61416%2Fceai.v27i4.9454-blue.svg)](https://doi.org/10.61416/ceai.v27i4.9454)
[![Journal](https://img.shields.io/badge/Journal-CEAI%20v27(4)%202025-green.svg)](https://doi.org/10.61416/ceai.v27i4.9454)

Reproducibility artefact (notebook and paper PDF) for the article:

> Amara, M., Benrhiem, O., Zhagdoud, R., Ebad, S. A., & Zidouri, A.
> *Feature Extraction of Handwritten Digit Recognition Using Stacked
> Convolutional Neural Network Ensemble.* Journal of Control Engineering and
> Applied Informatics (CEAI), **27**(4), 16–28, 2025.
> <https://doi.org/10.61416/ceai.v27i4.9454>

The notebook reproduces the MNIST experiments reported in the paper.

---

## Overview

A stacked ensemble that fuses penultimate-layer embeddings from three
ImageNet-pretrained CNN backbones and trains a meta-learner on the
concatenated features. Evaluated on MNIST, the ensemble reaches **99 %**
accuracy.

* **Base models:** ResNet50, VGG16, VGG19 (transfer learning from ImageNet)
* **Feature fusion:** penultimate-layer embeddings concatenated
* **Meta-learner:** multinomial logistic regression
* **Training:** 50 epochs per base model (to match the paper's learning curves)
* **Evaluation:** hold-out test and k-fold cross-validation (multi-seed)
* **Statistical analysis:** Wilcoxon signed-rank test, McNemar test,
  95 % confidence intervals

---

## Repository layout

```
.
├── README.md
├── CITATION.cff                                   machine-readable citation
├── requirements.txt                               Python dependencies
├── Notebook_A_MNIST_Stacked_Ensemble_50epochs.ipynb   main experiment notebook
└── paper/
    └── Amara2025_CEAI_stacked_cnn_ensemble.pdf    published article (PDF)
```

---

## Getting started

```bash
# Clone this repository
git clone https://github.com/marwaamara/A-Novel-Stacked-CNN-Ensemble-for-Enhanced-Feature-Extraction.git
cd A-Novel-Stacked-CNN-Ensemble-for-Enhanced-Feature-Extraction

# (Optional) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\Activate.ps1    # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Open the notebook
jupyter notebook Notebook_A_MNIST_Stacked_Ensemble_50epochs.ipynb
```

**Google Colab:** open the `.ipynb` from the repo in Colab, run the first
cell to install dependencies, then execute all cells. Runtime depends on GPU
availability.

> **Note:** the notebook's default subset sizes (`train_subset_n=500`,
> `test_subset_n=100`) are set for quick execution. Increase them for a
> full-scale run. 50 epochs per backbone reproduce the curves shown in the paper.

---

## Generated outputs

Running the notebook writes all figures and CSVs to `paper_outputs_mnist/`:

| Output | Description |
|--------|-------------|
| `training_curves_3x2.png` | Accuracy / loss curves for each backbone |
| `feature_maps_block1conv1_vgg16.png` | Early convolutional feature maps |
| `meta_accuracy_over_iterations.png` | Meta-learner iterative accuracy |
| `model_accuracy_comparison_holdout.png` | Hold-out accuracy comparison |
| `meta_confusion_matrix.png` | Confusion matrix for the meta-learner |
| `cv_boxplot.png` | Cross-validation accuracy distribution |
| `model_accuracy_comparison_cv.png` | CV mean ± std bar chart |
| `history_*.csv` | Per-backbone training histories |
| `*_metrics.json` | Per-model metrics |
| `cv_summary.csv` | Cross-validation summary |
| `significance_tests.csv` | Wilcoxon / McNemar significance results |

---

## Citation

If you use this code or the accompanying data, please cite the article:

```bibtex
@article{Amara2025CEAI,
  title   = {Feature Extraction of Handwritten Digit Recognition Using Stacked Convolutional Neural Network Ensemble},
  author  = {Amara, Marwa and Benrhiem, Olfa and Zhagdoud, Radhia and Ebad, Shouki A. and Zidouri, Abdelmalek},
  journal = {Journal of Control Engineering and Applied Informatics},
  volume  = {27},
  number  = {4},
  pages   = {16--28},
  year    = {2025},
  doi     = {10.61416/ceai.v27i4.9454},
  issn    = {1454-8658}
}
```

---

## Funding

Deanship of Scientific Research at Northern Border University, Arar,
Kingdom of Saudi Arabia.

---

## Contact

Dr Marwa Amara — Northern Border University, Arar, Kingdom of Saudi Arabia —
[Marwa.amara@nbu.edu.sa](mailto:Marwa.amara@nbu.edu.sa)
