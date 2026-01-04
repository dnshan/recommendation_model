import random
import time
from adaptive_model import AdaptiveLearningPath

def simulate_student_score(word, difficulty, student_ability):
    """
    Simulate a score based on word difficulty vs student ability.
    """
    base_score = 100 - (difficulty * 10) + (student_ability * 5)
    noise = random.randint(-15, 15)
    final_score = base_score + noise
    return max(0, min(100, final_score))

def run_demo():
    # 1. Setup Syllabus
    syllabus = {
        'amma': 1, 'thaththa': 1, 'balla': 1, 'pusa': 1,  # Level 1 (Simple)
        'wathura': 2, 'kema': 2, 'gedara': 2,             # Level 2 (Medium)
        'iskole': 3, 'poth': 3                            # Level 3 (Hard)
    }
    
    print("--- 🎓 Initializing Adaptive Learning Model ---")
    model = AdaptiveLearningPath(syllabus)
    
    # 2. Simulate Student Session
    student_ability = 5  # Scale 1-10
    current_word = None
    last_score = None
    
    session_length = 15
    print(f"--- 🏃 Starting {session_length}-step Simulation (Student Ability: {student_ability}/10) ---\n")
    
    for step in range(1, session_length + 1):
        # Predict Next Word
        prediction = model.predict_next_word(current_word, last_score)
        next_word = prediction['word']
        reason = prediction['reason']
        
        print(f"[{step}] Teacher AI: \"{reason}\" -> Assigning: '{next_word}'")
        
        # Simulate Student Attempt
        difficulty = syllabus[next_word]
        score = simulate_student_score(next_word, difficulty, student_ability)
        status = "⭐⭐⭐" if score >= 80 else ("⭐⭐" if score >= 60 else "⚠️")
        
        print(f"   Student attempts '{next_word}'... Score: {score} {status}")
        
        # Update Model
        practice_log = [{'word': next_word, 'score': score}]
        model.update_state(practice_log)
        
        # Prepare for next loop
        current_word = next_word
        last_score = score
        
        # Slowly improve student ability
        if score < 60: student_ability += 0.5 
        
        print("-" * 50)

if __name__ == "__main__":
    run_demo()
