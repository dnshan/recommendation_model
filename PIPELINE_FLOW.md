# 🔄 Complete Pipeline Flow: From Data Generation to ML Demo

## Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ADAPTIVE LEARNING PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────┘

    PHASE 1: DATA GENERATION          PHASE 2: MODEL TRAINING         PHASE 3: ML DEMO
    ────────────────────────          ─────────────────────           ────────────────
    
    generate_data.py                  train_model.py                  demo_ml.py
    ────────────────                  ────────────────                ────────────────
```

---

## 📊 PHASE 1: Data Generation (`generate_data.py`)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Setup Syllabus                                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  syllabus = {                                                          │
│      'වතුර': 1,    # Level 1 (Simple)                                  │
│      'කතුර': 1,                                                         │
│      'අම්මා': 2,    # Level 2 (Medium)                                 │
│      'සමනලයා': 3,  # Level 3 (Hard)                                    │
│      ...                                                                │
│  }                                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Simulate Learning Sessions                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  FOR each session (500 sessions):                                       │
│      │                                                                 │
│      ├─► Create NEW AdaptiveLearningPath(syllabus)                    │
│      │   └─► Student starts fresh (no history)                       │
│      │                                                                 │
│      ├─► FOR each step (20 steps per session):                        │
│      │   │                                                             │
│      │   ├─► [1] Extract Features                                     │
│      │   │   └─► model.get_features(current_word, last_score)         │
│      │   │       Returns: [last_score, avg_last3, consecutive_fails,   │
│      │   │                attempts, current_level, pct_mastered,       │
│      │   │                word_difficulty]                             │
│      │   │                                                             │
│      │   ├─► [2] Get Ground Truth Action                              │
│      │   │   └─► model._determine_rule_based_action(...)              │
│      │   │       Uses pedagogical rules to decide:                     │
│      │   │       • REPEAT (if score < 60)                             │
│      │   │       • SAME_LEVEL_NEW (if score 60-79)                     │
│      │   │       • ADVANCE_LEVEL (if 80% mastered)                     │
│      │   │       • etc.                                                │
│      │   │                                                             │
│      │   ├─► [3] Save Training Example                                │
│      │   │   └─► Write to CSV: features + action_label                │
│      │   │       Example row:                                          │
│      │   │       [75, 72, 0, 2, 1, 0.3, 1, "SAME_LEVEL_NEW"]          │
│      │   │                                                             │
│      │   ├─► [4] Execute Action                                       │
│      │   │   └─► model._execute_action(action, current_word)           │
│      │   │       Returns: {'word': 'කතුර', 'reason': '...'}            │
│      │   │                                                             │
│      │   ├─► [5] Simulate Student Response                             │
│      │   │   └─► score = simulate_student_score(word, diff, ability)   │
│      │   │       Formula: 100 - (difficulty×10) + (ability×5) ± noise│
│      │   │                                                             │
│      │   └─► [6] Update Model State                                   │
│      │       └─► model.update_state([{'word': word, 'score': score}]) │
│      │           Updates: attempts, scores[], status                  │
│      │                                                             │
│      └─► Student ability evolves (improves on failures)                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ training_data.csv│
                    │                  │
                    │ 10,000+ rows    │
                    │ Features +      │
                    │ Action Labels   │
                    └──────────────────┘
```

**Output:** `training_data.csv` with ~10,000 training examples

---

## 🎓 PHASE 2: Model Training (`train_model.py`)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Load Training Data                                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  df = pd.read_csv('training_data.csv')                                 │
│                                                                         │
│  Columns:                                                               │
│    • last_score              (Feature)                                 │
│    • avg_last3               (Feature)                                 │
│    • consecutive_fails       (Feature)                                 │
│    • attempts                (Feature)                                 │
│    • current_level           (Feature)                                 │
│    • percent_level_mastered  (Feature)                                 │
│    • word_difficulty         (Feature)                                 │
│    • action_label            (TARGET) ← What we want to predict        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Prepare Data for Training                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  X = df.drop('action_label', axis=1)  # Features (7 columns)           │
│  y = df['action_label']                # Labels (action strings)        │
│                                                                         │
│  Split:                                                                 │
│    • 80% → Training Set (X_train, y_train)                            │
│    • 20% → Test Set (X_test, y_test)                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Train Decision Tree Classifier                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  clf = DecisionTreeClassifier(                                         │
│      max_depth=5,              # Limit tree complexity                │
│      random_state=42,          # Reproducibility                       │
│      class_weight='balanced'   # Handle imbalanced classes             │
│  )                                                                      │
│                                                                         │
│  clf.fit(X_train, y_train)                                             │
│                                                                         │
│  The tree learns patterns like:                                        │
│    IF percent_level_mastered > 0.88                                    │
│      AND current_level <= 2.5                                          │
│      THEN → ADVANCE_LEVEL                                               │
│                                                                         │
│    IF last_score < 60                                                   │
│      AND consecutive_fails >= 3                                        │
│      THEN → SWITCH_EASIER                                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 4: Evaluate Model                                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  y_pred = clf.predict(X_test)                                          │
│                                                                         │
│  Metrics Generated:                                                     │
│    • Classification Report (precision, recall, F1)                     │
│    • Confusion Matrix (how often each action is predicted correctly)  │
│    • Feature Importance (which features matter most)                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 5: Save Model & Visualizations                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Saved Files:                                                           │
│    ✓ policy_model.pkl              ← Trained model (pickle)            │
│    ✓ plots/classification_report.txt                                   │
│    ✓ plots/confusion_matrix.png                                        │
│    ✓ plots/feature_importance.png                                      │
│    ✓ plots/tree_rules.txt          ← Human-readable rules             │
│    ✓ plots/tree.dot                ← Graphviz tree structure           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Output:** `policy_model.pkl` - A trained decision tree that can predict teaching actions

---

## 🚀 PHASE 3: ML Demo (`demo_ml.py`)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Initialize Model with Trained ML                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  syllabus = {                                                          │
│      'වතුර': 1, 'කතුර': 1, 'අම්මා': 2, ...                            │
│  }                                                                      │
│                                                                         │
│  model = AdaptiveLearningPath(                                          │
│      syllabus,                                                          │
│      model_path='policy_model.pkl'  ← Loads trained decision tree      │
│  )                                                                      │
│                                                                         │
│  Inside AdaptiveLearningPath:                                           │
│    • Loads policy_model.pkl                                            │
│    • Stores it as self.ml_model                                        │
│    • Ready to use ML predictions!                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Run Learning Session Simulation                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  FOR each step (1 to 15):                                               │
│      │                                                                 │
│      ├─► [1] Predict Next Word                                         │
│      │   └─► prediction = model.predict_next_word(                     │
│      │                       current_word,                              │
│      │                       last_score                                │
│      │                   )                                             │
│      │                                                                 │
│      │   Inside predict_next_word():                                   │
│      │   ├─► Extract features: get_features(...)                       │
│      │   │   → [75, 72, 0, 2, 1, 0.3, 1]                              │
│      │   │                                                             │
│      │   ├─► Try ML Prediction FIRST                                   │
│      │   │   └─► X = np.array([features])                              │
│      │   │       action = self.ml_model.predict(X)[0]                  │
│      │   │       → "SAME_LEVEL_NEW"                                    │
│      │   │                                                             │
│      │   ├─► Fallback to Rules (if ML fails)                          │
│      │   │   └─► action = _determine_rule_based_action(...)            │
│      │   │                                                             │
│      │   └─► Execute Action                                           │
│      │       └─► result = _execute_action(action, current_word)        │
│      │           → {'word': 'කතුර',                                    │
│      │              'reason': '[ML PREDICTED: SAME_LEVEL_NEW] ...'}    │
│      │                                                                 │
│      ├─► [2] Display Prediction                                         │
│      │   └─► print(f"[{step}] {prediction['reason']}")                 │
│      │       print(f"    Assigning: '{prediction['word']}'")           │
│      │                                                                 │
│      ├─► [3] Simulate Student Attempt                                  │
│      │   └─► score = simulate_student_score(word, diff, ability)       │
│      │       → 78                                                      │
│      │                                                                 │
│      ├─► [4] Update Model State                                        │
│      │   └─► model.update_state([{'word': word, 'score': score}])      │
│      │       • Increments attempts                                     │
│      │       • Adds score to history                                   │
│      │       • Updates status (Mastered/Satisfactory/Needs Practice)   │
│      │                                                                 │
│      └─► Prepare for next iteration                                     │
│          current_word = next_word                                       │
│          last_score = score                                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Output:** Console output showing ML-driven recommendations step-by-step

---

## 🔄 Complete Flow Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    END-TO-END PIPELINE FLOW                             │
└─────────────────────────────────────────────────────────────────────────┘

    [1] generate_data.py
         │
         │ Simulates 500 sessions × 20 steps
         │ Uses RULE-BASED policy as "teacher"
         │ Records: (features, action_label)
         │
         ▼
    training_data.csv
         │
         │ Contains ~10,000 examples
         │ Format: [7 features] + [1 label]
         │
         ▼
    [2] train_model.py
         │
         │ Loads CSV → Trains DecisionTreeClassifier
         │ Evaluates → Generates metrics & plots
         │ Saves → policy_model.pkl
         │
         ▼
    policy_model.pkl
         │
         │ Trained ML model
         │ Can predict: REPEAT, SAME_LEVEL_NEW, etc.
         │
         ▼
    [3] demo_ml.py
         │
         │ Loads policy_model.pkl
         │ Runs 15-step simulation
         │ Uses ML predictions (with rule fallback)
         │ Shows adaptive recommendations
         │
         ▼
    Console Output
         │
         │ Step-by-step learning recommendations
         │ ML-predicted actions with explanations
         │ Student performance simulation
```

---

## 🎯 Key Differences: Rule-Based vs ML-Driven

### Rule-Based (`demo_adaptive.py`)
```
┌─────────────────────────────────────┐
│  Student State                      │
│  └─► Extract Features               │
│      └─► Rule-Based Logic           │
│          └─► Action                 │
│              └─► Execute            │
└─────────────────────────────────────┘
```
- **No ML model needed**
- Uses hardcoded pedagogical rules
- Deterministic decisions
- Good for initial testing

### ML-Driven (`demo_ml.py`)
```
┌─────────────────────────────────────┐
│  Student State                      │
│  └─► Extract Features               │
│      └─► ML Model (Decision Tree)  │
│          └─► Predicted Action      │
│              └─► Execute            │
│                  (Fallback to Rules │
│                   if ML fails)      │
└─────────────────────────────────────┘
```
- **Requires `policy_model.pkl`**
- Learned patterns from training data
- Can generalize to new situations
- More adaptive and data-driven

---

## 📋 Execution Order

To run the complete pipeline:

```bash
# Step 1: Generate training data
python generate_data.py
# Output: training_data.csv

# Step 2: Train the ML model
python train_model.py
# Output: policy_model.pkl + plots/

# Step 3: Run ML demo
python demo_ml.py
# Output: Console simulation with ML predictions
```

---

## 🔍 What Happens at Each Step?

### During Data Generation:
- **Rule-based system** acts as the "expert teacher"
- Records what a good teacher would do in each situation
- Creates labeled dataset for ML to learn from

### During Training:
- **Decision tree** learns to mimic the rule-based teacher
- Finds patterns in the features that predict actions
- Becomes a "learned teacher" that can generalize

### During Demo:
- **ML model** makes predictions based on learned patterns
- Can handle situations it wasn't explicitly trained on
- Falls back to rules if ML prediction fails (safety net)

---

## 💡 Why This Hybrid Approach?

1. **Safety**: Rules ensure pedagogical constraints are always followed
2. **Adaptability**: ML learns from data and can improve over time
3. **Explainability**: Decision trees provide clear reasoning paths
4. **Robustness**: Fallback to rules if ML model fails or is unavailable

This creates a **child-safe, adaptive, and explainable** AI tutoring system! 🎓✨

