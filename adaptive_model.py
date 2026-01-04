import random
import pickle
import os
import numpy as np

# Action Constants
ACTION_REPEAT = "REPEAT"
ACTION_SWITCH_EASIER = "SWITCH_EASIER"
ACTION_SAME_LEVEL_NEW = "SAME_LEVEL_NEW"
ACTION_REVIEW_MASTERED = "REVIEW_MASTERED"
ACTION_ADVANCE_LEVEL = "ADVANCE_LEVEL"
ACTION_COURSE_COMPLETE = "COURSE_COMPLETE"

class AdaptiveLearningPath:
    def __init__(self, word_difficulty_map, model_path=None):
        """
        Initialize the adaptive model.
        Args:
            word_difficulty_map (dict): {word: level}
            model_path (str): Optional path to .pkl model for ML-based prediction
        """
        self.word_difficulty_map = word_difficulty_map
        self.student_history = {} 
        self.current_level = 1
        self.ml_model = None
        
        # Load ML model if exists
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.ml_model = pickle.load(f)
                print(f"ML Model loaded from {model_path}")
            except Exception as e:
                print(f"Failed to load ML model: {e}")

        # Organize words by level
        self.levels = {}
        for word, level in word_difficulty_map.items():
            if level not in self.levels:
                self.levels[level] = []
            self.levels[level].append(word)
            
    def update_state(self, practice_log):
        for entry in practice_log:
            word = entry['word']
            score = entry['score']
            
            if word not in self.student_history:
                self.student_history[word] = {'attempts': 0, 'scores': [], 'status': 'New'}
            
            history = self.student_history[word]
            history['attempts'] += 1
            history['scores'].append(score)
            
            # Simple Status Update Logic
            if score >= 80: history['status'] = 'Mastered'
            elif score >= 60: history['status'] = 'Satisfactory'
            else: history['status'] = 'Needs Practice'
        
    def get_features(self, current_word, last_score):
        """
        Extract features for ML Prediction.
        Returns:
            list: Feature vector [last_score, avg_3, cons_fails, attempts, current_lvl, pct_mastered, diff]
        """
        # Defaults if no history
        avg_last3 = 0
        consecutive_fails = 0
        attempts = 0
        difficulty = 0
        
        if current_word:
            difficulty = self.word_difficulty_map.get(current_word, 1)
            history = self.student_history.get(current_word, {})
            if history:
                scores = history.get('scores', [])
                attempts = history.get('attempts', 0)
                if scores:
                    avg_last3 = sum(scores[-3:]) / len(scores[-3:])
                    # Calc consecutive fails (<60)
                    for s in reversed(scores):
                        if s < 60: consecutive_fails += 1
                        else: break
        
        # Level stats
        level_words = self.levels.get(self.current_level, [])
        mastered_count = len([w for w in level_words if self.student_history.get(w, {}).get('status') == 'Mastered'])
        pct_mastered = (mastered_count / len(level_words)) if level_words else 1.0
        
        last_score_safe = last_score if last_score is not None else -1
        
        # Feature Vector compatible with sklearn
        # [last_score, avg_last3, consecutive_fails, attempts, current_level, percent_level_mastered, word_difficulty]
        features_vec = [
            last_score_safe,
            avg_last3,
            consecutive_fails,
            attempts,
            self.current_level,
            pct_mastered,
            difficulty
        ]
        
        return features_vec

    def _determine_rule_based_action(self, current_word, last_score):
        """
        Determine the pedagogical action using explicit rules (Ground Truth).
        """
        # 0. Cold Start
        if current_word is None:
            return ACTION_SAME_LEVEL_NEW
            
        features = self.get_features(current_word, last_score)
        # Unpack for readability
        # [last_score, avg_last3, consecutive_fails, attempts, current_level, pct_mastered, word_difficulty]
        consecutive_fails = features[2]
        pct_mastered = features[5]
        
        # 1. Immediate Remediation (Score < 60)
        if last_score is not None and last_score < 60:
            if consecutive_fails >= 3:
                return ACTION_SWITCH_EASIER
            return ACTION_REPEAT

        # 2. Reinforcement (60-79)
        if last_score is not None and 60 <= last_score < 80:
            # Random choice in rules, but we need deterministic label for "preferred strategy" 
            # or we accept noise. Let's make it deterministic for training: 
            # If avg is good, move on, else repeat.
            if features[1] > 70: # avg_last3
                 return ACTION_SAME_LEVEL_NEW
            return ACTION_REPEAT

        # 3. Progression (80+) or Default
        
        # Spaced Repetition trigger
        if random.random() < 0.2 and self._get_mastered_global():
            return ACTION_REVIEW_MASTERED

        level_words = self.levels.get(self.current_level, [])
        # Check Level Completion
        if pct_mastered >= 0.8:
            if (self.current_level + 1) in self.levels:
                return ACTION_ADVANCE_LEVEL
            else:
                return ACTION_COURSE_COMPLETE
                
        # Normal progression
        return ACTION_SAME_LEVEL_NEW

    def _get_mastered_global(self):
         return [w for w, data in self.student_history.items() if data['status'] == 'Mastered']

    def _execute_action(self, action, current_word):
        """
        Convert an abstract ACTION into a concrete {word, reason} response.
        """
        target_level_words = self.levels.get(self.current_level, [])
        
        if action == ACTION_REPEAT:
            return {'word': current_word, 'reason': "Repeating '{}' to improve score/mastery.".format(current_word)}
            
        elif action == ACTION_SWITCH_EASIER:
            candidates = self._get_mastered_global()
            if not candidates: candidates = [w for w in target_level_words if w != current_word]
            word = random.choice(candidates) if candidates else current_word
            return {'word': word, 'reason': "Switching to '{}' to rebuild confidence.".format(word)}
            
        elif action == ACTION_REVIEW_MASTERED:
            candidates = self._get_mastered_global()
            if candidates:
                word = random.choice(candidates)
                return {'word': word, 'reason': "Spaced Repetition: Reviewing '{}'.".format(word)}
            # Fallback
            return self._execute_action(ACTION_SAME_LEVEL_NEW, current_word)
            
        elif action == ACTION_ADVANCE_LEVEL:
            self.current_level += 1
            new_words = self.levels.get(self.current_level, [])
            word = new_words[0] if new_words else "ERROR"
            return {'word': word, 'reason': "Level Up! Advancing to Level {}.".format(self.current_level)}
            
        elif action == ACTION_COURSE_COMPLETE:
            word = random.choice(target_level_words)
            return {'word': word, 'reason': "Course Complete! Free practice with '{}'.".format(word)}
            
        elif action == ACTION_SAME_LEVEL_NEW:
            # Try to find unseen, then needs practice, then random
            unseen = [w for w in target_level_words if w not in self.student_history]
            needs_practice = [w for w in target_level_words if self.student_history.get(w, {}).get('status') == 'Needs Practice']
            
            if needs_practice: word = needs_practice[0]
            elif unseen: word = unseen[0]
            else: word = random.choice(target_level_words)
            
            return {'word': word, 'reason': "Continuing Level {}. Next word: '{}'.".format(self.current_level, word)}
            
        return {'word': current_word, 'reason': "Fallback action."}

    def predict_next_word(self, current_word=None, last_score=None):
        features = self.get_features(current_word, last_score)
        
        # 1. Try ML Prediction
        action = None
        used_ml = False
        if self.ml_model:
            try:
                # Reshape for single sample
                X = np.array([features])
                action = self.ml_model.predict(X)[0]
                used_ml = True
            except Exception as e:
                print(f"ML Error: {e}")
        
        # 2. Fallback to Rules
        if not action:
            action = self._determine_rule_based_action(current_word, last_score)
            
        # 3. Execute
        result = self._execute_action(action, current_word)
        if used_ml:
            result['reason'] = "[ML PREDICTED: {}] ".format(action) + result['reason']
        
        return result
