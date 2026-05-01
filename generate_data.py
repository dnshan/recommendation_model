import csv
import random
from adaptive_model import AdaptiveLearningPath

def simulate_student_score(word, difficulty, student_ability):
    base_score = 100 - (difficulty * 10) + (student_ability * 5)
    noise = random.randint(-15, 15)
    final_score = base_score + noise
    return max(0, min(100, final_score))

def generate_dataset(num_sessions=10000, steps_per_session=20, output_file='training_data.csv'):
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
    
    headers = ['last_score', 'avg_last3', 'consecutive_fails', 'attempts', 
               'current_level', 'percent_level_mastered', 'word_difficulty', 'action_label']
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        count = 0
        for _ in range(num_sessions):
            model = AdaptiveLearningPath(syllabus)
            current_word = None
            last_score = None
            student_ability = random.choice([3, 5, 7, 9]) # Varied student types
            
            for _ in range(steps_per_session):
                # 1. Capture State Features BEFORE action
                features = model.get_features(current_word, last_score)
                
                # 2. Get GROUND TRUTH Action from Rules
                action = model._determine_rule_based_action(current_word, last_score)
                
                # 3. Save to Dataset
                if current_word is not None: # Don't train on cold start nulls
                     writer.writerow(features + [action])
                     count += 1
                
                # 4. Execute to move state forward
                result = model._execute_action(action, current_word)
                next_word = result['word']
                
                # 5. Simulate Interaction
                diff = syllabus.get(next_word, 1)
                score = simulate_student_score(next_word, diff, student_ability)
                
                model.update_state([{'word': next_word, 'score': score}])
                current_word = next_word
                last_score = score
                
                # Evolution
                if score < 60: student_ability += 0.2
                
    print(f"Generated {count} training samples to {output_file}")

if __name__ == "__main__":
    generate_dataset()
