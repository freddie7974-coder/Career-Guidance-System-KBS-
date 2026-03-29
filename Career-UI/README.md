# Career Guidance System — Dashboard

## Quick Start

```bash
# 1. Install dependencies (MongoDB is optional)
pip install flask scikit-learn pandas

# 2. Make sure career_model.pkl is in this folder
#    If it's missing, run week2_decision_tree.py first (see below)

# 3. Start the app
python app.py

# 4. Open your browser
#    http://127.0.0.1:5000
```

---

## Folder Structure

```
Career-UI/
├── app.py                  ← Flask backend (all logic lives here)
├── career_model.pkl        ← Trained Decision Tree (from Week 2)
├── requirements.txt        ← Python dependencies
├── data/
│   └── student-scores-6k.csv
├── static/
│   ├── style.css
│   └── main.js
└── templates/
    ├── index.html          ← Main dashboard
    └── students.html       ← Student database page
```

---

## Integrating With Your Week 1–3 Python Files

The `app.py` file already contains all the logic from Weeks 1, 2, and 3
embedded directly. You do NOT need to import your week files separately.

If you want to rebuild the model from scratch:

```bash
# From your project root (where week2_decision_tree.py lives):
python3 - << 'EOF'
import pickle, pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

FEATURE_COLS = [
    'gender','part_time_job','absence_days','extracurricular_activities',
    'weekly_self_study_hours','math_score','history_score','physics_score',
    'chemistry_score','biology_score','english_score','geography_score'
]
df = pd.read_csv('data/student-scores-6k.csv')
df = df.drop(columns=['id','first_name','last_name','email'])
df['part_time_job']              = df['part_time_job'].map({'True':1,'False':0})
df['extracurricular_activities'] = df['extracurricular_activities'].map({'True':1,'False':0})
le = LabelEncoder(); df['gender'] = le.fit_transform(df['gender'])
ce = LabelEncoder(); y = ce.fit_transform(df['career_aspiration'])
X = df[FEATURE_COLS]
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = DecisionTreeClassifier(max_depth=8, criterion='gini',
                               min_samples_split=10, min_samples_leaf=5, random_state=42)
model.fit(X_train, y_train)
with open('Career-UI/career_model.pkl','wb') as f:
    pickle.dump({'model': model, 'encoder': ce, 'feature_cols': FEATURE_COLS}, f)
print('Model saved.')
EOF
```

---

## MongoDB (Optional)

The app works without MongoDB — it uses in-memory storage as a fallback.
A badge in the top-right of the dashboard shows which mode is active.

To enable MongoDB:
```bash
pip install pymongo
# Make sure MongoDB is running on localhost:27017
```

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET  | `/`             | Main dashboard |
| GET  | `/students`     | Student database page |
| POST | `/recommend`    | Run the full pipeline, return top 5 careers |
| POST | `/import-csv`   | Import CSV into storage |
| GET  | `/api/students` | Paginated student list (supports `?search=` and `?page=`) |
| GET  | `/api/profiles` | Saved recommendation profiles |
| GET  | `/status`       | Check MongoDB and model status |

### /recommend — Request Body

```json
{
  "name": "Alex",
  "interests": ["computers", "math"],
  "skills": ["logic", "programming"],
  "traits": ["analytical"],
  "math": 95, "history": 70, "physics": 88,
  "chemistry": 80, "biology": 65, "english": 74, "geography": 72,
  "study_hours": 30,
  "gender": 1, "part_time": 0, "absence": 2, "extracurricular": 0
}
```
