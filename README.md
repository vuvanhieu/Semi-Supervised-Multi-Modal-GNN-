
# Trust-Aware Semi-Supervised Multi-Modal GNN for Skin Lesion Classification

## 1. Overview

This repository contains the code and experimental results for the paper:

> **Trust-Aware Semi-Supervised Multi-Modal Graph Neural Network for Skin Lesion Classification**

A novel graph-based method is proposed to classify skin lesions by fusing deep features, handcrafted features, and clinical metadata. Trust-aware pseudo-labeling is applied to improve performance with high-confidence unlabeled samples.

## 2. Methodology

- Deep features are extracted from pretrained CNNs (VGG19, MobileNet, DenseNet121).
- Handcrafted features are computed (color histograms, HSV histograms, fractal features).
- Clinical metadata (sex, age, anatomical site) is preprocessed by one-hot encoding and normalization.
- Features are fused using a lightweight attention module.
- A k-nearest neighbors (KNN) graph is constructed.
- A GNN model (Graph Convolutional Network) is trained on the fused features.
- Trust-aware entropy-based pseudo-labeling is applied for semi-supervised learning.

## 3. Dataset

- **ISIC 2020 Challenge Dataset**
- A sample version with 1000 images is used for demonstration.
- Dataset Link: [ISIC 2020 Sample on Kaggle](https://www.kaggle.com/code/eliasgatternig/isic2020-sample1000)

## 4. Model Architecture

The model consists of five key components:

1. Deep Feature Extraction
2. Handcrafted Feature Extraction
3. Metadata Processing
4. Attention-based Fusion
5. GNN Classifier

A diagram is provided below:

![Model Architecture](Multi-modal-GCN.png)

```text
Deep Features + Handcrafted Features → Fusion with Clinical Metadata → 
KNN Graph Construction → GCN → Classification
```

## 5. Experimental Results

| Scenario | Phase | Test Accuracy | Macro F1 | Macro AUC |
|:---|:---|:---|:---|:---|
| Scenario 1 (Deep + Handcrafted + Metadata) | Phase 1 | 0.962 | 0.962 | 0.988 |
| Scenario 1 (Deep + Handcrafted + Metadata) | Phase 2 | 0.983 | 0.932 | 0.920 |
| Scenario 2 (Deep Only) | Phase 1 | 0.998 | 0.998 | 0.999 |
| Scenario 3 (Handcrafted Only) | Phase 1 | 0.987 | 0.987 | 0.997 |
| Scenario 4 (Metadata Only) | Phase 1 | 0.793 | 0.792 | 0.880 |

## 6. Directory Structure

```bash
├── data9_small/                  # Input feature directories (Deep, Handcrafted, Metadata)
├── training_data9_small_MultiModal_GNN_scenario_1/
│   ├── GNN_deep_handcrafted_clinical/     # Phase 1 training results
│   ├── GNN_pseudo_finetune/               # Phase 2 retraining results
│   ├── reports/                           # Best fold XAI and analysis
│   ├── plots_before_vs_after/             # Metric comparison plots (before vs after)
```

## 7. Key Visualizations

- Trust level distribution after entropy-based segmentation.
- Entropy distribution histogram.
- Improvements in F1, Precision, Recall, Macro-F1, Macro-Recall, and Macro-AUC after trust-aware pseudo-labeling.

## 8. How to Run

```bash
python your_main_script.py
```

Make sure the dataset paths, feature directories, and environment are set correctly.

## 9. Future Work

- Testing the model on larger datasets.
- Extending the model to multi-class classification.
- Enhancing metadata fusion using dynamic graph construction.
- Applying advanced pseudo-labeling strategies.

## 10. Contact

For any questions, please contact the author.

---

# Notes

- Results might vary slightly depending on random seed settings.
- All experiments were run on a standard GPU environment.

---

Thank you for using this repository! 🚀
