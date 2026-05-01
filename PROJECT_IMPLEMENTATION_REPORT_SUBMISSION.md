# Adaptive Practice Sequencing Model
## Formal Project Implementation Report

### Student Information
- Student Name: ____________________
- Registration Number: ____________________
- Module Code: ____________________
- Supervisor: ____________________
- Date: ____________________

---

## Abstract

This report presents the implementation of an Adaptive Practice Sequencing Model designed for early-learning pronunciation practice. The system applies a hybrid approach that combines a machine learning classifier with deterministic pedagogical rules to recommend the most suitable next word for practice. A decision tree classifier is trained on simulated learner-session data derived from rule-based expert behavior. The implementation demonstrates a full prototype pipeline, including data generation, model training, evaluation, artifact export, and inference simulation. Experimental results from the current setup report an overall test accuracy of 0.81, with strong performance on major action classes. The project provides an explainable and extensible baseline for AI-assisted learning progression.

---

## 1. Introduction

Adaptive tutoring systems must personalize learning flow while maintaining educational safety and clarity. In early-childhood pronunciation learning, recommending a suitable next practice word can reduce frustration and improve confidence. This project focuses on building an explainable recommendation engine for such progression decisions.

The implemented system predicts pedagogical actions (for example, repeat current word, switch to easier content, or advance level) and then maps each action to a concrete next-word recommendation.

---

## 2. Final Implementation Details

### 2.1 Model and Architecture
- Core ML model: `DecisionTreeClassifier` (scikit-learn)
- Main training parameters:
  - `max_depth=5`
  - `class_weight='balanced'`
  - `random_state=42`
- Saved model: `policy_model.pkl`

### 2.2 Hybrid Decision Strategy
- Primary decision source: ML-predicted action.
- Safety mechanism: rule-based fallback when ML is unavailable or fails.
- Predicted actions include:
  - `REPEAT`
  - `SWITCH_EASIER`
  - `SAME_LEVEL_NEW`
  - `ADVANCE_LEVEL`
  - `REVIEW_MASTERED`

### 2.3 Implementation Stack
- Python
- scikit-learn
- pandas
- numpy
- matplotlib
- pickle
- Graphviz DOT export support

### 2.4 Deployment Scope
- Inference mode: local model loading from `policy_model.pkl`
- Cloud inference: not used in this version
- Frontend/database integration: not included as a production layer in this repository

---

## 3. Dataset Details

### 3.1 Data Source
The current training dataset is synthetically generated through simulation (`generate_data.py`) using the rule-based adaptive policy as a teaching oracle.

### 3.2 Dataset Structure
- File: `training_data.csv`
- Features:
  - `last_score`
  - `avg_last3`
  - `consecutive_fails`
  - `attempts`
  - `current_level`
  - `percent_level_mastered`
  - `word_difficulty`
- Label:
  - `action_label`

### 3.3 Data Splitting
- 80% training / 20% testing
- Stratified split to preserve class distribution (`stratify=y`)

### 3.4 Ethics Note
Since the current training dataset is simulation-generated, participant consent and personal-data handling do not apply to this specific training pipeline stage.

---

## 4. Processing and Training Pipeline

The implemented end-to-end pipeline is:

1. Generate learner-session records and labels (`generate_data.py`)
2. Extract learner-state features (`adaptive_model.py`)
3. Save labeled examples into `training_data.csv`
4. Train decision tree model (`train_model.py`)
5. Evaluate on held-out test split
6. Export model outputs and visual artifacts
7. Run inference simulation demo (`demo_ml.py`)

Generated artifacts include:
- `policy_model.pkl`
- `plots/classification_report.txt`
- `plots/confusion_matrix.png`
- `plots/feature_importance.png`
- `plots/tree.dot`
- `plots/tree_rules.txt`

---

## 5. Evaluation and Results

Based on the saved evaluation report (`plots/classification_report.txt`):

- Overall accuracy: **0.81**
- Test support: **38,000** samples

Class-wise highlights:
- `ADVANCE_LEVEL`: Precision 0.80, Recall 1.00, F1-score 0.89
- `REPEAT`: Precision 1.00, Recall 1.00, F1-score 1.00
- `SAME_LEVEL_NEW`: Precision 0.81, Recall 1.00, F1-score 0.89
- `REVIEW_MASTERED`: Precision 1.00, Recall 0.00, F1-score 0.00

The low recall for `REVIEW_MASTERED` indicates a class imbalance/representation issue that should be addressed in future data and policy refinement.

---

## 6. Project Status

- Adaptive logic implementation: Completed
- Data generation pipeline: Completed
- Model training and evaluation pipeline: Completed
- Model artifact persistence and reuse: Completed
- Visualization/report export: Completed
- Production integration layer: Partial / future work

**Current state:** Fully implemented prototype-level recommendation component.

---

## 7. Evidence and Artifacts

Primary evidence files:
- `plots/confusion_matrix.png`
- `plots/feature_importance.png`
- `plots/classification_report.txt`
- `plots/tree.dot`
- `plots/tree_rules.txt`
- `decision_tree.png`
- `decision_tree_high_quality.png`
- `PIPELINE_FLOW.md`

---

## 8. Limitations

- Reliance on synthetic training data rather than real deployment interaction logs.
- Class imbalance impacts minority-action prediction quality.
- Single-tree model may limit generalization under more complex learner patterns.
- No production API, web/mobile frontend, or persistent database integration in this repository.

---

## 9. Future Work

- Integrate real user-session data collection and iterative retraining.
- Improve minority-class handling via sampling or class-sensitive optimization.
- Introduce ensemble models for stronger generalization.
- Build API endpoints for direct app integration.
- Add monitoring for data drift and recommendation quality over time.

---

## 10. Conclusion

The project successfully delivers an explainable, hybrid adaptive recommendation prototype that can guide next-word practice decisions using learner performance state. The current implementation establishes a strong baseline and provides clear expansion paths toward production integration and data-driven improvement.

