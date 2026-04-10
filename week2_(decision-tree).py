# 
#   KNOWLEDGE-BASED CAREER GUIDANCE SYSTEM
#   Week 2: Decision Tree Model
# 
#
# WHAT WE DO THIS WEEK:
#   1. Load and prepare the real student dataset (6,000 students)
#   2. Encode categorical features (gender, part_time_job, etc.)
#   3. Train a Decision Tree classifier using scikit-learn
#   4. Evaluate accuracy with a train/test split
#   5. Identify underperforming careers using RIASEC classification
#   6. Apply a Knowledge Graph confidence boost as a tie-breaker
#   7. Visualise the Decision Tree and feature importances
#
# KEY CONCEPT — RIASEC MODEL:
#   RIASEC is a widely-used career psychology framework that groups
#   careers into 6 personality types:
#     R — Realistic    (hands-on, technical: Engineer, Construction)
#     I — Investigative(analytical, research: Scientist, Doctor)
#     A — Artistic     (creative, expressive: Artist, Writer, Designer)
#     S — Social       (people-focused: Teacher, Government Officer)
#     E — Enterprising (leadership, business: Lawyer, Business Owner)
#     C — Conventional (structured, data: Accountant, Banker)
#
#   The Decision Tree struggles with Artistic (A) careers like Writer
#   and Designer because subject scores alone don't capture creativity.
#   In Week 3, the Knowledge Graph will BOOST these careers using
#   interest/trait signals as a tie-breaker when confidence is low.
#
# Run this file with: python week2_decision_tree.py
# ============================================================

import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings('ignore')


# -------------------------------------------------------
# RIASEC CAREER CLASSIFICATION
# -------------------------------------------------------
# Maps each career in our dataset to its RIASEC type.
# This helps us explain WHY the Decision Tree struggles with
# certain careers — Artistic (A) roles are hardest to predict
# from scores alone, since creativity isn't a gradable subject.

RIASEC_MAP = {
    # R — Realistic (technical, hands-on)
    "Construction Engineer": "R",
    "Software Engineer":     "R",
    "Game Developer":        "R",

    # I — Investigative (analytical, research)
    "Doctor":                "I",
    "Scientist":             "I",
    "Stock Investor":        "I",

    # A — Artistic (creative, expressive) ← struggles most
    "Artist":                "A",
    "Writer":                "A",
    "Designer":              "A",

    # S — Social (people-focused, communicative)
    "Teacher":               "S",
    "Government Officer":    "S",
    "Social Network Studies":"S",

    # E — Enterprising (leadership, persuasion)
    "Lawyer":                "E",
    "Business Owner":        "E",
    "Real Estate Developer": "E",

    # C — Conventional (structured, data-driven)
    "Accountant":            "C",
    "Banker":                "C",
}

RIASEC_LABELS = {
    "R": "Realistic",
    "I": "Investigative",
    "A": "Artistic  ⚠️  (Knowledge Graph boost needed)",
    "S": "Social",
    "E": "Enterprising",
    "C": "Conventional",
}

# Artistic careers that the Decision Tree historically underperforms on.
# In Week 3, the Knowledge Graph will apply a confidence boost to these
# when the tree returns a low-confidence prediction.
ARTISTIC_CAREERS = {"Artist", "Writer", "Designer"}

# Confidence threshold below which we hand off to the Knowledge Graph
KG_BOOST_THRESHOLD = 0.35   # i.e. if top prediction confidence < 35%



def load_dataset(filepath):
    """
    Loads the student CSV dataset into a pandas DataFrame.

    A DataFrame is like a spreadsheet in Python — rows are students,
    columns are features (scores, habits, career aspiration, etc.)
    """
    print("📂 Loading dataset...")
    df = pd.read_csv(filepath)
    print(f"   ✅ Loaded {len(df):,} student records with {len(df.columns)} columns.")
    print(f"   Columns: {list(df.columns)}\n")
    return df


# -------------------------------------------------------
# SECTION 2: PREPARE FEATURES & LABELS
# -------------------------------------------------------

def prepare_data(df):
    """
    Prepares the data for training a Decision Tree.

    Steps:
      1. Drop columns we don't need (id, name, email)
      2. Encode True/False columns as 1/0
      3. Encode 'gender' as 0/1 using LabelEncoder
      4. Separate features (X) from the label (y = career_aspiration)
      5. Encode career labels as numbers (required by scikit-learn)

    Returns X (features), y (encoded labels), label_encoder, feature_names
    """
    print("🔧 Preparing data...")

    # Drop non-predictive columns
    df = df.drop(columns=['id', 'first_name', 'last_name', 'email'])

    # Convert True/False strings to integers (1/0)
    df['part_time_job']             = df['part_time_job'].map({'True': 1, 'False': 0})
    df['extracurricular_activities'] = df['extracurricular_activities'].map({'True': 1, 'False': 0})

    # Encode gender: male=1, female=0
    gender_encoder = LabelEncoder()
    df['gender'] = gender_encoder.fit_transform(df['gender'])

    # Define features (X) — everything except the career column
    feature_cols = [
        'gender', 'part_time_job', 'absence_days',
        'extracurricular_activities', 'weekly_self_study_hours',
        'math_score', 'history_score', 'physics_score',
        'chemistry_score', 'biology_score', 'english_score', 'geography_score'
    ]

    X = df[feature_cols]
    y_raw = df['career_aspiration']

    # Encode career names as integers for the model
    # e.g. "Doctor" → 3, "Software Engineer" → 14
    career_encoder = LabelEncoder()
    y = career_encoder.fit_transform(y_raw)

    print(f"   ✅ Features: {feature_cols}")
    print(f"   ✅ Target careers: {list(career_encoder.classes_)}")
    print(f"   ✅ X shape: {X.shape}  |  y shape: {y.shape}\n")

    return X, y, career_encoder, feature_cols


# -------------------------------------------------------
# SECTION 3: TRAIN / TEST SPLIT
# -------------------------------------------------------

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.

    WHY DO WE SPLIT?
      We train the model on 80% of the data, then test how well it
      predicts careers for the remaining 20% it has NEVER seen before.
      This tells us how well the model generalises — not just memorises.

    Parameters:
        test_size    : float — fraction of data for testing (0.2 = 20%)
        random_state : int   — seed for reproducibility (same split every run)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"📊 Data Split:")
    print(f"   Training set : {len(X_train):,} students ({100-test_size*100:.0f}%)")
    print(f"   Testing set  : {len(X_test):,}  students ({test_size*100:.0f}%)\n")
    return X_train, X_test, y_train, y_test


# -------------------------------------------------------
# SECTION 4: TRAIN THE DECISION TREE
# -------------------------------------------------------

def train_decision_tree(X_train, y_train, max_depth=8, random_state=42):
    """
    Trains a Decision Tree classifier on the training data.

    WHAT IS max_depth?
      Controls how many levels deep the tree can grow.
      - Too shallow (depth 2–3): underfits, misses patterns
      - Too deep (depth 20+):    overfits, memorises training data
      - Depth 8 is a good balance for this dataset

    HOW DOES TRAINING WORK?
      The tree learns which questions (e.g. "math_score > 85?") best
      separate students into career groups — it picks the question
      that creates the purest split each time (using Gini impurity).
    """
    print(f"🌳 Training Decision Tree (max_depth={max_depth})...")

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion='gini',         # measure of split purity
        min_samples_split=10,     # minimum students needed to split a node
        min_samples_leaf=5,       # minimum students in a leaf node
        random_state=random_state
    )

    model.fit(X_train, y_train)
    print(f"   ✅ Training complete.\n")
    return model


# -------------------------------------------------------
# SECTION 5: EVALUATE THE MODEL
# -------------------------------------------------------

def evaluate_model(model, X_train, X_test, y_train, y_test, career_encoder):
    """
    Evaluates the trained model on both train and test data.

    KEY METRICS:
      - Accuracy      : % of predictions that were correct
      - Precision     : of all students predicted as Career X, how many truly are?
      - Recall        : of all true Career X students, how many did we catch?
      - F1-Score      : harmonic mean of precision and recall

    Overfitting check:
      If training accuracy >> test accuracy, the tree memorised the data.
    """
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test,  model.predict(X_test))

    print("=" * 55)
    print("📈 MODEL EVALUATION")
    print("=" * 55)
    print(f"   Training Accuracy : {train_acc*100:.2f}%")
    print(f"   Test Accuracy     : {test_acc*100:.2f}%")

    gap = train_acc - test_acc
    if gap < 0.05:
        print(f"   Overfitting Check : ✅ Good (gap = {gap*100:.1f}%)")
    else:
        print(f"   Overfitting Check : ⚠️  Possible overfit (gap = {gap*100:.1f}%)")

    print("\n📋 Classification Report (Test Set):")
    print(classification_report(
        y_test, model.predict(X_test),
        target_names=career_encoder.classes_
    ))

    return test_acc


# -------------------------------------------------------
# SECTION 5b: RIASEC PERFORMANCE ANALYSIS
# -------------------------------------------------------

def analyse_riasec_performance(model, X_test, y_test, career_encoder):
    """
    Groups career prediction accuracy by RIASEC type.

    This reveals a key insight: Artistic (A) careers score lowest
    because subject scores don't capture creative ability.
    That gap is exactly what the Week 3 Knowledge Graph boost will fix.
    """
    from collections import defaultdict

    y_pred   = model.predict(X_test)
    correct  = defaultdict(int)
    total    = defaultdict(int)

    for actual, predicted in zip(y_test, y_pred):
        career = career_encoder.classes_[actual]
        riasec = RIASEC_MAP.get(career, "?")
        total[riasec]  += 1
        if actual == predicted:
            correct[riasec] += 1

    print("\n" + "=" * 55)
    print("🔬 RIASEC GROUP ACCURACY ANALYSIS")
    print("=" * 55)
    print(f"  {'Type':<6} {'RIASEC Label':<42} {'Accuracy':>8}")
    print("  " + "-" * 52)

    for code in ["R", "I", "A", "S", "E", "C"]:
        if total[code] == 0:
            continue
        acc   = correct[code] / total[code] * 100
        label = RIASEC_LABELS[code]
        bar   = "█" * int(acc / 10)
        print(f"  {code:<6} {label:<42} {acc:>6.1f}%  {bar}")

    print()
    print("  💡 Insight: Artistic (A) careers score lowest because")
    print("     subject scores cannot measure creative ability.")
    print("  🔜 Fix (Week 3): Knowledge Graph will boost Artist,")
    print("     Writer, and Designer when tree confidence < "
          f"{KG_BOOST_THRESHOLD*100:.0f}%.")
    print("=" * 55)

def plot_feature_importances(model, feature_names, output_path):
    """
    Plots a bar chart of which features the Decision Tree
    found most useful for predicting careers.

    Higher importance = the tree uses this feature more at the top
    of its splits (i.e. it's a stronger predictor of career choice).
    """
    importances = model.feature_importances_
    sorted_idx  = importances.argsort()[::-1]

    colours = ['#2E75B6' if importances[i] > 0.05 else '#A9C4E2'
               for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(feature_names)), importances[sorted_idx], color=colours, edgecolor='white')
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels([feature_names[i] for i in sorted_idx], rotation=35, ha='right', fontsize=10)
    ax.set_ylabel("Importance Score", fontsize=11)
    ax.set_title("🌳 Decision Tree — Feature Importances\n(Which student attributes best predict career choice?)",
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(0, max(importances) * 1.2)

    for i, idx in enumerate(sorted_idx):
        ax.text(i, importances[idx] + 0.002, f"{importances[idx]:.3f}",
                ha='center', va='bottom', fontsize=8, color='#1F4E79')

    high_patch = mpatches.Patch(color='#2E75B6', label='High importance (>0.05)')
    low_patch  = mpatches.Patch(color='#A9C4E2', label='Lower importance')
    ax.legend(handles=[high_patch, low_patch], fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Feature importance chart saved → {output_path}")


# -------------------------------------------------------
# SECTION 7: VISUALISE — DECISION TREE (top levels)
# -------------------------------------------------------

def plot_decision_tree(model, feature_names, career_encoder, output_path):
    """
    Visualises the top 4 levels of the Decision Tree.

    We only show 4 levels because the full tree (depth=8) is too
    large to read. The top levels show the most important decisions.
    """
    fig, ax = plt.subplots(figsize=(28, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=career_encoder.classes_,
        filled=True,
        rounded=True,
        fontsize=7,
        max_depth=4,          # only show top 4 levels for readability
        impurity=False,
        ax=ax
    )
    ax.set_title(
        "🌳 Decision Tree — Career Prediction (Top 4 Levels)\n"
        "Each node asks a question; the tree splits students left (True) or right (False)",
        fontsize=14, fontweight='bold', pad=20
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Decision tree diagram saved → {output_path}")


# -------------------------------------------------------
# SECTION 8: VISUALISE — CONFUSION MATRIX
# -------------------------------------------------------

def plot_confusion_matrix(model, X_test, y_test, career_encoder, output_path):
    """
    A confusion matrix shows which careers the model confuses with each other.

    Rows    = actual career
    Columns = predicted career
    Diagonal (top-left to bottom-right) = correct predictions ✅
    Off-diagonal = mistakes (e.g. predicted Doctor but was actually Nurse)
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=career_encoder.classes_)
    disp.plot(ax=ax, colorbar=True, cmap='Blues', xticks_rotation=45)
    ax.set_title("Confusion Matrix — Career Predictions vs Actual\n"
                 "(Diagonal = correct predictions)",
                 fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Confusion matrix saved → {output_path}")


# -------------------------------------------------------
# SECTION 9: PREDICT FOR A NEW STUDENT
# -------------------------------------------------------

def predict_career(model, career_encoder, feature_names, student_data):
    """
    Uses the trained Decision Tree to predict a career for a new student.

    If confidence is below KG_BOOST_THRESHOLD AND the top prediction
    is an Artistic career, a flag is raised so the Week 3 Knowledge
    Graph can step in as a tie-breaker.

    Parameters:
        student_data : dict — one row of feature values for the student

    Returns:
        predicted_career : str   — top career name
        confidence       : float — confidence percentage
        top3             : list  — top 3 (career, %) tuples
        kg_boost_needed  : bool  — True if Knowledge Graph should intervene
        kg_boost_reason  : str   — explanation of why boost is triggered
    """
    row = pd.DataFrame([student_data], columns=feature_names)

    prediction_idx   = model.predict(row)[0]
    probabilities    = model.predict_proba(row)[0]
    confidence       = round(max(probabilities) * 100, 1)
    predicted_career = career_encoder.inverse_transform([prediction_idx])[0]

    top3_idx = probabilities.argsort()[::-1][:3]
    top3 = [(career_encoder.classes_[i], round(probabilities[i]*100, 1))
            for i in top3_idx]

    # ── Knowledge Graph boost check ──
    riasec_type      = RIASEC_MAP.get(predicted_career, "?")
    is_artistic      = predicted_career in ARTISTIC_CAREERS
    is_low_conf      = (confidence / 100) < KG_BOOST_THRESHOLD
    kg_boost_needed  = is_low_conf  # boost whenever confidence is low
    # (Week 3 will prioritise Artistic careers during the boost)

    if kg_boost_needed and is_artistic:
        kg_boost_reason = (
            f"Low confidence ({confidence}%) on an Artistic career ('{predicted_career}'). "
            f"Knowledge Graph will apply interest/trait signals as a tie-breaker in Week 3."
        )
    elif kg_boost_needed:
        kg_boost_reason = (
            f"Low confidence ({confidence}%) — Decision Tree is uncertain. "
            f"Knowledge Graph will cross-check student interests in Week 3."
        )
    else:
        kg_boost_reason = None

    return predicted_career, confidence, top3, kg_boost_needed, kg_boost_reason


# -------------------------------------------------------
# MAIN — RUN EVERYTHING
# -------------------------------------------------------

if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("  🌳 CAREER GUIDANCE SYSTEM — WEEK 2: DECISION TREE")
    print("=" * 55 + "\n")

    # ── Step 1: Load data ──
    df = load_dataset('data/student-scores-6k.csv')

    # ── Step 2: Prepare features and labels ──
    X, y, career_encoder, feature_names = prepare_data(df)

    # ── Step 3: Split into train / test sets ──
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    # ── Step 4: Train the Decision Tree ──
    model = train_decision_tree(X_train, y_train, max_depth=8)

    # ── Step 5: Evaluate ──
    test_accuracy = evaluate_model(model, X_train, X_test, y_train, y_test, career_encoder)

    # ── Step 5b: RIASEC group analysis ──
    analyse_riasec_performance(model, X_test, y_test, career_encoder)

    # ── Step 6: Visualisations ──
    print("\n📊 Generating visualisations...")
    plot_feature_importances(model, feature_names, 'outputs/feature_importances.png')
    plot_decision_tree(model, feature_names, career_encoder, 'outputs/decision_tree.png')
    plot_confusion_matrix(model, X_test, y_test, career_encoder, 'outputs/confusion_matrix.png')

    # ── Step 7: Print the top 5 levels as text ──
    print("\n📝 Decision Tree Rules (top 3 levels, text format):")
    print(export_text(model, feature_names=feature_names, max_depth=3))

    # ── Step 8: Predict for sample students ──
    print("\n🎯 PREDICTING CAREERS FOR SAMPLE STUDENTS")
    print("=" * 55)

    students = [
        {
            "name": "Alex (tech-oriented)",
            "features": {
                'gender': 1, 'part_time_job': 0, 'absence_days': 3,
                'extracurricular_activities': 0, 'weekly_self_study_hours': 30,
                'math_score': 95, 'history_score': 70, 'physics_score': 92,
                'chemistry_score': 88, 'biology_score': 65,
                'english_score': 75, 'geography_score': 72
            }
        },
        {
            "name": "Jordan (healthcare-oriented)",
            "features": {
                'gender': 0, 'part_time_job': 0, 'absence_days': 1,
                'extracurricular_activities': 1, 'weekly_self_study_hours': 35,
                'math_score': 78, 'history_score': 80, 'physics_score': 75,
                'chemistry_score': 90, 'biology_score': 96,
                'english_score': 85, 'geography_score': 77
            }
        },
        {
            "name": "Sam (creative-oriented)",
            "features": {
                'gender': 0, 'part_time_job': 1, 'absence_days': 5,
                'extracurricular_activities': 1, 'weekly_self_study_hours': 20,
                'math_score': 65, 'history_score': 85, 'physics_score': 60,
                'chemistry_score': 62, 'biology_score': 70,
                'english_score': 92, 'geography_score': 80
            }
        },
    ]

    for student in students:
        career, confidence, top3, kg_boost, kg_reason = predict_career(
            model, career_encoder, feature_names, student["features"]
        )
        riasec = RIASEC_MAP.get(career, "?")
        print(f"\n  Student  : {student['name']}")
        print(f"  Top Pick : {career}  (confidence: {confidence}%)  "
              f"[RIASEC: {riasec} — {RIASEC_LABELS.get(riasec, '?')}]")
        print(f"  Top 3    :")
        for c, p in top3:
            bar = "█" * int(p / 5)
            print(f"    {c:<30} {p:>5}%  {bar}")
        if kg_boost:
            print(f"  ⚡ KG Boost: {kg_reason}")

    print("\n✅ Week 2 complete! Check the output images for visualisations.")
    print(f"   Overall test accuracy: {test_accuracy*100:.2f}%\n")
