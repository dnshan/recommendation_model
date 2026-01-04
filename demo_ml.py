import random
import warnings
from adaptive_model import AdaptiveLearningPath
from generate_data import simulate_student_score

import sys

# Suppress feature name warnings from sklearn (harmless difference between dict/df training)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Ensure UTF-8 encoding for printing Sinhala characters
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Fallback for older python versions if needed
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

def run_ml_demo():
    print("--- Initializing ML-Driven Adaptive Coach ---")
    syllabus = {
    # Level 1 – very simple, common words
    'වතුර': 1,
    'කතුර': 1,
    'බලා': 1,
    'පුසා': 1,
    'ගම': 1,
    'මල්': 1,

    # Level 2 – slightly longer / compound sounds
    'කැම': 2,
    'ගෙදර': 2,
    'අම්මා': 2,
    'තාත්තා': 2,
    'පාසල': 2,
    'පොත': 2,

    # Level 3 – clearer multi-syllable structure
    'සමනලයා': 3,
    'ළමයා': 3,
    'ගුරුවරයා': 3,
    'කුරුල්ලා': 3,
    'මිතුරා': 3,

    # Level 4 – longer nouns, more complex sounds
    'පරිසරය': 4,
    'විදුහල': 4,
    'කෘෂිකර්මය': 4,
    'සංගීතය': 4,
    'චිත්‍රකලාව': 4,

    # Level 5 – abstract ideas / academic words
    'අධ්‍යාපනය': 5,
    'සංස්කෘතිය': 5,
    'සංවිධානය': 5,
    'පර්යේෂණය': 5,
    'තාක්ෂණය': 5
    }
    
    # Initialize with Model
    model = AdaptiveLearningPath(syllabus, model_path='policy_model.pkl')
    
    student_ability = 6
    current_word = None
    last_score = None
    
    for step in range(1, 15):
        # Predict using ML
        prediction = model.predict_next_word(current_word, last_score)
        
        print(f"[{step}] {prediction['reason']}")
        print(f"    Assigning: '{prediction['word']}'")
        
        # Simulate
        next_word = prediction['word']
        diff = syllabus.get(next_word, 1)
        score = simulate_student_score(next_word, diff, student_ability)
        
        print(f"    Result: Score {score}")
        
        model.update_state([{'word': next_word, 'score': score}])
        current_word = next_word
        last_score = score
        print("-" * 40)

if __name__ == "__main__":
    run_ml_demo()
