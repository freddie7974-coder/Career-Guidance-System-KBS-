import pandas as pd
import os

class CareerInferenceEngine:
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cannot find CSV at: {os.path.abspath(file_path)}")
        
        # Load Knowledge Base
        self.df = pd.read_csv(file_path)
        
        # Standardize column names (lowercase and underscores)
        self.df.columns = self.df.columns.str.strip().str.replace(' ', '_').str.lower()
        print(f"✅ Knowledge Base Loaded. Analyzing {len(self.df)} student profiles...")

    def run_inference(self, row_index):
        """
        Week 4 Task: Implementation of Forward Chaining logic.
        Matches student facts against a predefined set of production rules.
        """
        student = self.df.iloc[row_index]
        
        # Extract facts from the student record
        name = f"{student.get('first_name', 'Student')} {student.get('last_name', '')}"
        math = student.get('math_score', 0)
        science = student.get('physics_score', student.get('chemistry_score', 0))
        verbal = student.get('english_score', 0)
        history = student.get('history_score', 0)
        study_hours = student.get('weekly_self_study_hours', 0)
        
        # Initialize default values
        recommendation = "General Studies"
        reasoning = "General academic profile."

        # Production Rule 1: STEM (Math + Physics)
        if math > 85 and science > 80:
            recommendation = "Software Engineer / Data Scientist"
            reasoning = "High quantitative skills and logic indicated by Math and Physics scores."
            
        # Production Rule 2: Medical (Biology + Chemistry)
        elif student.get('biology_score', 0) > 85 and student.get('chemistry_score', 0) > 80:
            recommendation = "Medical Professional"
            reasoning = "Strong aptitude for Life Sciences and Chemistry."
            
        # Production Rule 3: Legal/Social (English + History)
        elif verbal > 85 and history > 80:
            recommendation = "Lawyer / Diplomat"
            reasoning = "Advanced verbal reasoning and historical context analysis."
            
        # Production Rule 4: Entrepreneurial (Math + Effort)
        elif math > 75 and study_hours > 30:
            recommendation = "Business Strategist / Owner"
            reasoning = "Combination of analytical baseline and high self-discipline (study hours)."

        return {
            "student": name,
            "career": recommendation,
            "explanation": reasoning
        }

if __name__ == "__main__":
    # Path logic: assumes CSV is in career/data/
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, 'data', 'student-scores-6k.csv')
    
    try:
        engine = CareerInferenceEngine(data_path)
        
        print("\n" + "="*60)
        print("WEEK 4: RULE-BASED INFERENCE SYSTEM")
        print("="*60)
        
        # Demonstrate inference on first 10 students
        for i in range(10):
            res = engine.run_inference(i)
            print(f"[{i+1}] {res['student']} -> {res['career']}")
            print(f"    REASONING: {res['explanation']}")
            print("-" * 60)
            
    except Exception as e:
        print(f"❌ Error: {e}")