🧠 Adaptive Practice Sequencing Model

Smart AI-based word recommendation engine for SpeechCoach Montessori App

📌 Overview

The Adaptive Practice Sequencing Model is an intelligent recommendation system that decides which word a child should practice next based on their pronunciation performance.

It is designed for early learners (ages 3–6) and follows Montessori learning principles:

Self-paced learning

Mastery before progression

Confidence-first approach

This system is adaptive, safe, and explainable, making it suitable for real-world educational applications.

🎯 Key Objectives

Personalize learning paths for each child
Reduce frustration by adjusting difficulty dynamically
Reinforce learning through repetition and review
Ensure safe progression through predefined syllabus levels
Provide clear reasons for every recommendation
🧩 How It Works (Simple Flow)

Child practices a word
System assigns a score (0–100)
Performance history is analyzed
AI selects a learning strategy
Rules select the safest next word
Next word + reason is returned

🧠 Core Idea: Hybrid AI System

This project uses a Hybrid AI approach:

1️⃣ Machine Learning (Decision Tree)

Predicts the best learning strategy

Learns from student performance patterns

Outputs actions like:

REPEAT
SWITCH_EASIER
NEW_WORD
ADVANCE_LEVEL
REVIEW_MASTERED

2️⃣ Rule-Based Engine

Selects the exact word

Enforces:
Difficulty levels
Syllabus boundaries
Safety constraints

✅ This ensures adaptability without sacrificing control or explainability

🧪 Example Learning Scenario
Word: "වතුර" → Score: 45
→ Action: REPEAT (child is struggling)

Word: "වතුර" → Score: 88
→ Action: NEW_WORD ("කතුර")

80% of Level 1 mastered
→ Action: ADVANCE_LEVEL (Level 2)

📊 Features Used for Decision Making

Last pronunciation score
Average of recent scores
Consecutive failures
Number of attempts
Current learning level
Percentage of mastered words
Word difficulty level



📂 Project Structure
.
├── adaptive_model.py        # Core recommendation engine
├── demo_ml.py               # Demo simulation (start here)
├── train_model.py           # Train ML decision tree
├── generate_data.py         # Generate synthetic training data
├── policy_model.pkl         # Trained ML model
├── plots/                   # Model visualizations
│   ├── tree.png
│   ├── feature_importance.png
│   └── confusion_matrix.png
└── INTEGRATION_GUIDE.md     # API & database integration guide

⚙️ Setup Instructions
1️⃣ Create Virtual Environment
python -m venv recommendation

2️⃣ Activate Environment

Windows

recommendation\Scripts\activate


Linux / macOS

source recommendation/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

▶️ Run Demo
python demo_ml.py


This simulates a student session and prints system decisions step-by-step.

🧑‍💻 Basic Usage Example
from adaptive_model import AdaptiveLearningPath

syllabus = {
    'වතුර': 1,
    'අම්මා': 2,
    'සමනලයා': 3
}

model = AdaptiveLearningPath(syllabus, model_path='policy_model.pkl')

prediction = model.predict_next_word()
print(prediction)

model.update_state([{'word': 'වතුර', 'score': 75}])
next_step = model.predict_next_word('වතුර', 75)
print(next_step)

🔄 Training Model
python generate_data.py
python train_model.py


This generates:
Trained decision tree model
Feature importance plot
Confusion matrix
Tree visualization

