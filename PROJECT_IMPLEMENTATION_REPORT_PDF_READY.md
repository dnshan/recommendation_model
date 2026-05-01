<div align="center">

# ADAPTIVE PRACTICE SEQUENCING MODEL
## Project Implementation Report

### AI-Based Recommendation Engine for Early-Learning Pronunciation Practice

---

**Student Name:** ____________________  
**Registration Number:** ____________________  
**Module Code:** ____________________  
**Module Name:** ____________________  
**Supervisor:** ____________________  
**Institution:** ____________________  
**Date:** ____________________

</div>

\newpage

## Declaration

I declare that this report is my own work and has not been submitted, in whole or in part, for any other academic award. Where the work of others has been used, it has been properly acknowledged.

**Signature:** ____________________  
**Date:** ____________________

\newpage

## Abstract

This document reports the implementation of an Adaptive Practice Sequencing Model that recommends the next word for pronunciation practice based on learner performance. The implemented approach combines a machine learning decision policy with rule-based pedagogical constraints. A decision tree model is trained using generated learner-state data and evaluated on a stratified holdout test set. The system supports explainable action prediction and safe fallback behavior, and produces evaluation outputs including classification metrics, confusion matrix, feature-importance visualization, and decision-tree rule exports. The current prototype achieves 0.81 overall test accuracy and provides a practical baseline for future real-user integration.

\newpage

## Table of Contents

1. Introduction  
2. Final Implementation Details  
3. Dataset Details  
4. Preprocessing and Training Pipeline  
5. Evaluation and Results  
6. Completion Status  
7. Screenshots and Evidence  
8. Limitations  
9. Future Enhancements  
10. Conclusion  

\newpage

## 1. Introduction

Adaptive educational systems must personalize progression while protecting learner confidence. This project addresses that need by recommending appropriate next words according to recent pronunciation performance, mastery progression, and content difficulty.

The system is designed as a hybrid architecture:
- machine learning for adaptive policy prediction
- deterministic rules for pedagogical safety and fallback reliability

---

## 2. Final Implementation Details

### 2.1 Core Model
- Model: `DecisionTreeClassifier` (scikit-learn)
- Hyperparameters:
  - `max_depth=5`
  - `class_weight='balanced'`
  - `random_state=42`
- Saved artifact: `policy_model.pkl`

### 2.2 Predicted Pedagogical Actions
- `REPEAT`
- `SWITCH_EASIER`
- `SAME_LEVEL_NEW`
- `ADVANCE_LEVEL`
- `REVIEW_MASTERED`

### 2.3 Feature Set Used for Prediction
- `last_score`
- `avg_last3`
- `consecutive_fails`
- `attempts`
- `current_level`
- `percent_level_mastered`
- `word_difficulty`

### 2.4 Technical Stack
- Python
- scikit-learn
- pandas
- numpy
- matplotlib
- pickle
- Graphviz DOT export

### 2.5 Runtime Mode
- Local model inference only (no cloud deployment in this version)
- CLI demonstration via `demo_ml.py`

---

## 3. Dataset Details

### 3.1 Source
Training data is generated synthetically in `generate_data.py` from simulated learner sessions driven by rule-based teaching policy.

### 3.2 Data Schema
- Input columns: 7 learner-state features
- Target column: `action_label`
- Output file: `training_data.csv`

### 3.3 Split Strategy
- 80% train / 20% test
- Stratified splitting (`stratify=y`) to preserve class distribution

---

## 4. Preprocessing and Training Pipeline

The practical pipeline is:

1. Generate labeled learner-state data (`generate_data.py`)
2. Build feature vectors from learner history (`adaptive_model.py`)
3. Train decision tree model (`train_model.py`)
4. Evaluate model on holdout split
5. Export model and visual artifacts
6. Run recommendation simulation (`demo_ml.py`)

Exported outputs:
- `policy_model.pkl`
- `plots/classification_report.txt`
- `plots/confusion_matrix.png`
- `plots/feature_importance.png`
- `plots/tree.dot`
- `plots/tree_rules.txt`

---

## 5. Evaluation and Results

From `plots/classification_report.txt`:

- Overall accuracy: **0.81**
- Total test support: **38,000**

Class-level metrics:
- `ADVANCE_LEVEL`: P=0.80, R=1.00, F1=0.89
- `REPEAT`: P=1.00, R=1.00, F1=1.00
- `SAME_LEVEL_NEW`: P=0.81, R=1.00, F1=0.89
- `REVIEW_MASTERED`: P=1.00, R=0.00, F1=0.00

Interpretation: high performance on major classes, with weak minority-class recall for `REVIEW_MASTERED`.

---

## 6. Completion Status

- Adaptive decision engine: Completed
- Data generation pipeline: Completed
- ML model training and evaluation: Completed
- Model persistence and reuse: Completed
- Plot/report artifact generation: Completed
- Production integration (API/app/database): Partial (future scope)

---

## 7. Screenshots and Evidence

Include the following figures/files in final submission appendix:

- `plots/confusion_matrix.png`
- `plots/feature_importance.png`
- `decision_tree.png`
- `decision_tree_high_quality.png`
- `plots/tree_rules.txt`
- `plots/classification_report.txt`
- `PIPELINE_FLOW.md`

---

## 8. Limitations

- Dataset is simulation-based, not real deployment logs.
- Minority class (`REVIEW_MASTERED`) is not well captured in prediction.
- Single decision-tree model may limit broader generalization.
- No production integration layer in this repository.

---

## 9. Future Enhancements

- Integrate real learner interaction data.
- Improve class balance and minority-action coverage.
- Evaluate ensemble alternatives (e.g., Random Forest, Gradient Boosting).
- Build API service for app-level consumption.
- Add monitoring and periodic retraining strategy.

---

## 10. Conclusion

This implementation delivers a complete prototype of an explainable hybrid recommendation engine for adaptive pronunciation practice sequencing. It demonstrates end-to-end feasibility and provides a structured foundation for scaling into production educational environments.

\newpage

## Appendix A - File Checklist

- [ ] `PROJECT_IMPLEMENTATION_REPORT_PDF_READY.md`
- [ ] `plots/classification_report.txt`
- [ ] `plots/confusion_matrix.png`
- [ ] `plots/feature_importance.png`
- [ ] `decision_tree_high_quality.png`
- [ ] `plots/tree_rules.txt`
- [ ] `PIPELINE_FLOW.md`

