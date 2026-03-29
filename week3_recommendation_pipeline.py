#   KNOWLEDGE-BASED CAREER GUIDANCE SYSTEM
#   Week 3: Recommendation Logic — The Unified Pipeline
# 
#
# WHAT IS DONE DURING THIS WEEK:
#   This is where everything from Weeks 1 and 2 comes together
#   into one complete, working career guidance system.
#
#   Week 1 gave us:
#     → Forward Chaining Inference Engine (rule-based scoring)
#     → Knowledge Graph (interest/trait → career relationships)
#     → Conflict Resolution strategy
#
#   Week 2 gave us:
#     → A trained Decision Tree (pattern-based career classifier)
#     → RIASEC analysis (why Artistic careers underperform)
#     → KG Boost flag (trigger for low-confidence predictions)
#
#   Week 3 builds:
#     → The UNIFIED PIPELINE that connects all of the above
#     → Knowledge Graph Boost (fixes Artistic career weakness)
#     → Graph Traversal (finds related careers via KG edges)
#     → A final ranked output with full plain-English reasoning
#     → An interactive student profile questionnaire (CLI)
#
# HOW THE PIPELINE FLOWS:
#
#   [Student Profile]
#         │
#         ▼
#   [Decision Tree]  ──── confidence >= 35%? ────► [Final Recommendation]
#         │                                                  ▲
#         │ confidence < 35%                                 │
#         ▼                                                  │
#   [Knowledge Graph Boost] ──── re-scores careers ─────────┘
#         │
#         ▼
#   [Forward Chaining Engine] ── validates & adds reasoning
#         │
#         ▼
#   [Graph Traversal] ── finds related careers from KG edges
#         │
#         ▼
#   [Conflict Resolution] ── reconciles DT vs Rules disagreement
#         │
#         ▼
#   [Final Report] ── ranked careers + reasoning + related paths
#
# Run this file with: python week3_recommendation_pipeline.py
# 

import os
import sys
import pickle
import pandas as pd

#  Make sure week1 functions are importable 
# We import directly from week1 so we never duplicate code.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from week1_career_guidance import (
    CAREER_KNOWLEDGE_BASE,
    KNOWLEDGE_GRAPH,
    create_student_profile,
    calculate_match_score,
    get_career_recommendations,
    resolve_conflict,
)


### SECTION 1: LOAD THE TRAINED DECISION TREE (from Week 2)


# RIASEC map — same as Week 2, reproduced here for standalone use
RIASEC_MAP = {
    "Construction Engineer": "R", "Software Engineer": "R", "Game Developer": "R",
    "Doctor":                "I", "Scientist":         "I", "Stock Investor":  "I",
    "Artist":                "A", "Writer":            "A", "Designer":        "A",
    "Teacher":               "S", "Government Officer":"S", "Social Network Studies": "S",
    "Lawyer":                "E", "Business Owner":    "E", "Real Estate Developer":  "E",
    "Accountant":            "C", "Banker":            "C",
}

ARTISTIC_CAREERS   = {"Artist", "Writer", "Designer"}
KG_BOOST_THRESHOLD = 0.35   # confidence below this triggers KG boost

# Mapping from dataset career names → Week 1 Knowledge Base names
# (The dataset has more careers; we map the ones that overlap)
CAREER_NAME_MAP = {
    "Software Engineer":     "Software Engineer",
    "Doctor":                "Doctor",
    "Teacher":               "Teacher",
    "Scientist":             "Data Scientist",   # closest match
    "Designer":              "Graphic Designer",
    "Writer":                "Journalist",        # closest match
    "Artist":                "Graphic Designer",  # closest match
}

FEATURE_COLS = [
    'gender', 'part_time_job', 'absence_days',
    'extracurricular_activities', 'weekly_self_study_hours',
    'math_score', 'history_score', 'physics_score',
    'chemistry_score', 'biology_score', 'english_score', 'geography_score'
]


def load_model(model_path='career_model.pkl', encoder_path='career_encoder.pkl'):
    """
    Loads the trained Decision Tree and career label encoder saved in Week 2.

    If the model files aren't found, it trains a fresh model automatically
    so the pipeline works as a standalone file too.
    """
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            career_encoder = pickle.load(f)
        print("   ✅ Loaded saved Decision Tree model.")
    else:
        print("   ⚠️  Model files not found — training a fresh model...")
        model, career_encoder = _train_fresh_model()
        print("   ✅ Fresh model trained.")
    return model, career_encoder


def _train_fresh_model():
    """Trains the Week 2 Decision Tree from scratch if .pkl files are missing."""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv('data/student-scores-6k.csv')
    df = df.drop(columns=['id', 'first_name', 'last_name', 'email'])
    df['part_time_job']              = df['part_time_job'].map({'True': 1, 'False': 0})
    df['extracurricular_activities'] = df['extracurricular_activities'].map({'True': 1, 'False': 0})
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    career_encoder = LabelEncoder()
    y = career_encoder.fit_transform(df['career_aspiration'])
    X = df[FEATURE_COLS]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = DecisionTreeClassifier(max_depth=8, criterion='gini',
                                   min_samples_split=10, min_samples_leaf=5,
                                   random_state=42)
    model.fit(X_train, y_train)
    return model, career_encoder


### SECTION 2: DECISION TREE PREDICTION


def dt_predict(model, career_encoder, scores_and_habits):
    """
    Runs the Decision Tree on a student's scores and habits.

    Parameters:
        scores_and_habits : dict — numerical feature values matching FEATURE_COLS

    Returns:
        dt_career    : str   — predicted career name
        dt_confidence: float — confidence as a decimal (0.0 – 1.0)
        dt_top3      : list  — top 3 (career, confidence%) tuples
    """
    row          = pd.DataFrame([scores_and_habits], columns=FEATURE_COLS)
    probs        = model.predict_proba(row)[0]
    top_idx      = probs.argmax()
    dt_career    = career_encoder.classes_[top_idx]
    dt_confidence= probs[top_idx]

    top3_idx = probs.argsort()[::-1][:3]
    dt_top3  = [(career_encoder.classes_[i], round(probs[i] * 100, 1))
                for i in top3_idx]

    return dt_career, dt_confidence, dt_top3


### SECTION 3: KNOWLEDGE GRAPH BOOST


def knowledge_graph_boost(student_profile, dt_top3, dt_confidence):
    """
    KNOWLEDGE GRAPH BOOST
    ──────────────────────
    Activated when the Decision Tree confidence < KG_BOOST_THRESHOLD (35%).

    HOW IT WORKS:
      The Knowledge Graph stores edges between interests/traits and careers.
      When the tree is uncertain, we traverse those edges to count how many
      of the student's interests and traits point toward each candidate career.

      A career that appears many times in the student's KG edges gets a
      higher boost score, which can override the tree's low-confidence pick.

    This is specifically designed to rescue Artistic (A) careers like
    Artist, Writer, and Designer that subject scores alone cannot detect.

    Returns:
        boosted_career : str  — the KG-recommended career (may differ from DT)
        boost_scores   : dict — {career: kg_score} for all candidates
        boost_reasoning: str  — plain-English explanation
    """
    boost_scores = {}

    # Build a candidate pool from the DT's top 3 + all KG-linked careers
    candidates = set(c for c, _ in dt_top3)

    for interest in student_profile["interests"]:
        linked = KNOWLEDGE_GRAPH["interests"].get(interest, [])
        candidates.update(linked)

    for trait in student_profile["traits"]:
        linked = KNOWLEDGE_GRAPH["traits"].get(trait, [])
        candidates.update(linked)

    # Score each candidate by counting KG edge hits
    for career in candidates:
        score = 0
        for interest in student_profile["interests"]:
            if career in KNOWLEDGE_GRAPH["interests"].get(interest, []):
                score += 3   # interest match = strong signal

        for trait in student_profile["traits"]:
            if career in KNOWLEDGE_GRAPH["traits"].get(trait, []):
                score += 2   # trait match = moderate signal

        # Extra boost for Artistic careers if student has creative signals
        if career in ARTISTIC_CAREERS:
            creative_interests = {"art", "design", "writing", "music", "drawing"}
            creative_traits    = {"creative", "expressive", "imaginative"}
            art_hits = len(creative_interests & set(student_profile["interests"]))
            art_hits += len(creative_traits    & set(student_profile["traits"]))
            score += art_hits * 2   # amplify creative signal

        if score > 0:
            boost_scores[career] = score

    if not boost_scores:
        # No KG signal — fall back to DT top pick
        return dt_top3[0][0], {}, "No strong Knowledge Graph signal found. Keeping Decision Tree result."

    # Pick the career with the highest KG score
    boosted_career   = max(boost_scores, key=boost_scores.get)
    top_score        = boost_scores[boosted_career]
    boost_reasoning  = (
        f"Decision Tree confidence was low ({dt_confidence*100:.1f}%). "
        f"Knowledge Graph traversal found '{boosted_career}' has the strongest "
        f"match to student interests and traits (KG score: {top_score}). "
        f"Artistic career boost applied where relevant."
    )

    return boosted_career, boost_scores, boost_reasoning


### SECTION 4: GRAPH TRAVERSAL — RELATED CAREERS


def traverse_related_careers(primary_career, depth=2):
    """
    GRAPH TRAVERSAL
    ───────────────
    Walks the Knowledge Graph edges to find careers related to the
    primary recommendation — like a "you might also consider..." feature.

    HOW IT WORKS:
      Starting from the primary career, we follow RELATED_TO edges
      stored in the CAREER_KNOWLEDGE_BASE. We go up to `depth` hops away.

      Depth 1 → direct neighbours (e.g. Doctor → Nurse, Pharmacist)
      Depth 2 → neighbours of neighbours (e.g. Nurse → Physical Therapist)

    This is a simple Breadth-First Search (BFS) over the career graph.

    Parameters:
        primary_career : str — the top recommended career
        depth          : int — how many hops to traverse (default: 2)

    Returns a list of related career names (excluding the primary).
    """
    visited  = {primary_career}
    frontier = [primary_career]
    related  = []

    for _ in range(depth):
        next_frontier = []
        for career in frontier:
            if career in CAREER_KNOWLEDGE_BASE:
                neighbours = CAREER_KNOWLEDGE_BASE[career].get("related_careers", [])
                for neighbour in neighbours:
                    if neighbour not in visited:
                        visited.add(neighbour)
                        related.append(neighbour)
                        next_frontier.append(neighbour)
        frontier = next_frontier

    return related


### SECTION 5: THE UNIFIED PIPELINE


def run_pipeline(student_profile, scores_and_habits, model, career_encoder, verbose=True):
    """
    THE UNIFIED RECOMMENDATION PIPELINE
    ─────────────────────────────────────
    Orchestrates all components from Weeks 1, 2, and 3 into a
    single end-to-end career recommendation flow.

    Steps:
      1. Run Decision Tree on scores/habits
      2. Check confidence → if low, activate KG Boost
      3. Run Forward Chaining engine on interests/traits
      4. Resolve conflicts between DT and rules
      5. Traverse Knowledge Graph for related careers
      6. Assemble and return the final recommendation report

    Parameters:
        student_profile   : dict — from create_student_profile()
        scores_and_habits : dict — numerical features for the Decision Tree
        model             : DecisionTreeClassifier — trained Week 2 model
        career_encoder    : LabelEncoder — maps integers back to career names
        verbose           : bool — print step-by-step pipeline trace

    Returns a complete recommendation report as a dictionary.
    """
    if verbose:
        print("\n" + "─" * 60)
        print(f"  🔄 Running pipeline for: {student_profile['name']}")
        print("─" * 60)

    # ── STEP 1: Decision Tree Prediction ──────────────────────
    if verbose: print("\n  [Step 1] 🌳 Decision Tree prediction...")
    dt_career, dt_confidence, dt_top3 = dt_predict(model, career_encoder, scores_and_habits)
    if verbose:
        print(f"           Result : {dt_career}  (confidence: {dt_confidence*100:.1f}%)")

    # ── STEP 2: Knowledge Graph Boost (if needed) ─────────────
    kg_boost_applied = False
    kg_reasoning     = None
    final_dt_career  = dt_career

    if dt_confidence < KG_BOOST_THRESHOLD:
        if verbose: print(f"\n  [Step 2] ⚡ KG Boost activated (confidence {dt_confidence*100:.1f}% < {KG_BOOST_THRESHOLD*100:.0f}%)...")
        boosted, boost_scores, kg_reasoning = knowledge_graph_boost(
            student_profile, dt_top3, dt_confidence
        )
        if boosted != dt_career:
            kg_boost_applied = True
            final_dt_career  = boosted
            if verbose: print(f"           KG overrides DT: '{dt_career}' → '{boosted}'")
        else:
            if verbose: print(f"           KG agrees with DT: '{dt_career}' retained.")
    else:
        if verbose: print(f"\n  [Step 2] ✅ DT confidence sufficient — KG Boost not needed.")

    # ── STEP 3: Forward Chaining Engine ───────────────────────
    if verbose: print(f"\n  [Step 3] ⚙️  Forward Chaining Inference Engine...")
    rule_recommendations, _ = get_career_recommendations(
        student_profile, top_n=5, decision_tree_result=final_dt_career
    )
    rule_top_career = rule_recommendations[0]["career"]
    rule_top_score  = rule_recommendations[0]["score"]
    if verbose: print(f"           Top rule match: {rule_top_career}  ({rule_top_score}%)")

    # ── STEP 4: Conflict Resolution ───────────────────────────
    if verbose: print(f"\n  [Step 4] ⚖️  Conflict resolution...")
    conflict_note = resolve_conflict(
        rule_recommendations[0],
        decision_tree_result=final_dt_career
    )
    if verbose: print(f"           {conflict_note}")

    # Determine final winner
    if rule_top_score >= 70:
        final_career    = rule_top_career
        final_source    = "Forward Chaining Engine (high rule confidence)"
        final_reasoning = rule_recommendations[0]["reasoning"]
    else:
        final_career    = final_dt_career
        final_source    = "Decision Tree" + (" + KG Boost" if kg_boost_applied else "")
        final_reasoning = (
            kg_reasoning if kg_boost_applied
            else f"Decision Tree predicted '{final_dt_career}' with {dt_confidence*100:.1f}% confidence."
        )

    # ── STEP 5: Graph Traversal — Related Careers ─────────────
    if verbose: print(f"\n  [Step 5] 🕸️  Knowledge Graph traversal (related careers)...")
    related = traverse_related_careers(final_career, depth=2)
    # Also pull from KG interest edges for additional discovery
    kg_linked = set()
    for interest in student_profile["interests"]:
        for c in KNOWLEDGE_GRAPH["interests"].get(interest, []):
            if c != final_career:
                kg_linked.add(c)
    all_related = list(dict.fromkeys(related + list(kg_linked)))[:6]  # deduplicated, max 6
    if verbose: print(f"           Related careers found: {all_related}")

    # ── STEP 6: Assemble Final Report ─────────────────────────
    riasec_type = RIASEC_MAP.get(final_career, "?")

    report = {
        "student_name":       student_profile["name"],
        "final_career":       final_career,
        "final_source":       final_source,
        "final_reasoning":    final_reasoning,
        "riasec_type":        riasec_type,
        "dt_career":          dt_career,
        "dt_confidence":      round(dt_confidence * 100, 1),
        "dt_top3":            dt_top3,
        "kg_boost_applied":   kg_boost_applied,
        "kg_reasoning":       kg_reasoning,
        "rule_top3":          rule_recommendations[:3],
        "conflict_note":      conflict_note,
        "related_careers":    all_related,
        "student_profile":    student_profile,
    }

    return report


### SECTION 6: DISPLAY THE FINAL REPORT
    

def display_report(report):
    """
    Prints the final career recommendation report in a
    clear, human-readable format suitable for a counsellor.
    """
    p = report
    w = 62

    print("\n" + "═" * w)
    print(f"  🎓  CAREER GUIDANCE REPORT — {p['student_name'].upper()}")
    print("═" * w)

    # ── Student profile summary ──
    sp = p["student_profile"]
    print(f"\n  Interests  : {', '.join(sp['interests'])}")
    print(f"  Skills     : {', '.join(sp['skills'])}")
    print(f"  Traits     : {', '.join(sp['traits'])}")
    print(f"  Environment: {sp['preferred_environment']}")

    # ── Final recommendation ──
    print(f"\n  {'─'*58}")
    print(f"  🏆  PRIMARY RECOMMENDATION:  {p['final_career']}")
    print(f"      RIASEC Type  : {p['riasec_type']}")
    print(f"      Decided by   : {p['final_source']}")
    print(f"      Reasoning    : {p['final_reasoning']}")

    # ── Pipeline trace ──
    print(f"\n  {'─'*58}")
    print(f"  📊  PIPELINE TRACE")
    print(f"      Decision Tree  → {p['dt_career']}  ({p['dt_confidence']}% confidence)")
    if p["kg_boost_applied"]:
        print(f"      KG Boost       → ⚡ Activated — overrode DT result")
    else:
        print(f"      KG Boost       → Not needed (confidence sufficient)")
    print(f"      Rules Engine   → {p['rule_top3'][0]['career']}  ({p['rule_top3'][0]['score']}% match)")
    if p["conflict_note"]:
        print(f"      Conflict Note  → {p['conflict_note']}")

    # ── DT top 3 ──
    print(f"\n  {'─'*58}")
    print(f"  🌳  DECISION TREE TOP 3:")
    for career, pct in p["dt_top3"]:
        bar = "█" * int(pct / 5)
        print(f"      {career:<30} {pct:>5}%  {bar}")

    # ── Rules top 3 ──
    print(f"\n  {'─'*58}")
    print(f"  ⚙️   FORWARD CHAINING TOP 3:")
    for rec in p["rule_top3"]:
        bar = "█" * int(rec['score'] / 10)
        print(f"      {rec['career']:<30} {rec['score']:>5}%  {bar}")

    # ── Related careers ──
    if p["related_careers"]:
        print(f"\n  {'─'*58}")
        print(f"  🕸️   RELATED CAREERS  (via Knowledge Graph traversal):")
        for c in p["related_careers"]:
            print(f"      • {c}")

    print("\n" + "═" * w + "\n")


### SECTION 7: INTERACTIVE CLI QUESTIONNAIRE
    
def run_interactive_questionnaire(model, career_encoder):
    """
    INTERACTIVE STUDENT QUESTIONNAIRE
    ───────────────────────────────────
    Guides a student through entering their profile step by step,
    then runs the full pipeline and displays their report.

    This is the "front end" of the career guidance system —
    the part a real student would interact with.
    """
    print("\n" + "═" * 62)
    print("  🎓  CAREER GUIDANCE SYSTEM — STUDENT QUESTIONNAIRE")
    print("═" * 62)
    print("  Answer the questions below. Press Enter after each one.\n")

    name = input("  Your name: ").strip() or "Student"

    print("\n  Rate your subject scores (40–100):")
    def get_score(subject):
        while True:
            try:
                val = int(input(f"    {subject} score: "))
                if 0 <= val <= 100: return val
                print("    Please enter a number between 0 and 100.")
            except ValueError:
                print("    Please enter a valid number.")

    math_score    = get_score("Math")
    physics_score = get_score("Physics")
    chemistry_score= get_score("Chemistry")
    biology_score = get_score("Biology")
    history_score = get_score("History")
    english_score = get_score("English")
    geography_score= get_score("Geography")

    print("\n  A few more questions:")

    def yes_no(question):
        while True:
            ans = input(f"    {question} (yes/no): ").strip().lower()
            if ans in ("yes", "y"): return 1
            if ans in ("no",  "n"): return 0
            print("    Please answer yes or no.")

    gender_input  = input("    Gender (male/female): ").strip().lower()
    gender        = 1 if gender_input.startswith("m") else 0
    part_time     = yes_no("Do you have a part-time job?")
    extracurricular= yes_no("Do you do extracurricular activities?")

    while True:
        try:
            absence_days   = int(input("    How many days absent per month? "))
            study_hours    = int(input("    Weekly self-study hours? "))
            break
        except ValueError:
            print("    Please enter a valid number.")

    # Interests
    print("\n  Select your interests (comma-separated).")
    print("  Options: computers, biology, math, art, writing, helping_people,")
    print("           nature, design, research, health, communication,")
    print("           problem_solving, current_events, education, technology")
    interests_raw = input("  Your interests: ")
    interests     = [i.strip().lower() for i in interests_raw.split(",") if i.strip()]

    # Skills
    print("\n  Select your skills (comma-separated).")
    print("  Options: math, logic, programming, science, communication,")
    print("           creativity, writing, research, statistics, visual_thinking,")
    print("           critical_thinking, patience, organization")
    skills_raw = input("  Your skills: ")
    skills     = [s.strip().lower() for s in skills_raw.split(",") if s.strip()]

    # Traits
    print("\n  Select your personality traits (comma-separated).")
    print("  Options: analytical, creative, empathetic, curious, detail_oriented,")
    print("           expressive, patient, persistent, passionate, calm_under_pressure")
    traits_raw = input("  Your traits: ")
    traits     = [t.strip().lower() for t in traits_raw.split(",") if t.strip()]

    # Environment
    print("\n  Preferred work environment:")
    env = input("  (indoors / outdoors / remote): ").strip().lower()
    if env not in ("indoors", "outdoors", "remote"):
        env = "indoors"

    # Build profile and features
    student_profile   = create_student_profile(name, interests, skills, traits, env)
    scores_and_habits = {
        'gender': gender, 'part_time_job': part_time,
        'absence_days': absence_days,
        'extracurricular_activities': extracurricular,
        'weekly_self_study_hours': study_hours,
        'math_score': math_score, 'history_score': history_score,
        'physics_score': physics_score, 'chemistry_score': chemistry_score,
        'biology_score': biology_score, 'english_score': english_score,
        'geography_score': geography_score
    }

    report = run_pipeline(student_profile, scores_and_habits, model, career_encoder, verbose=True)
    display_report(report)


### MAIN — RUN THE FULL PIPELINE

if __name__ == "__main__":

    print("\n" + "═" * 62)
    print("  🚀  CAREER GUIDANCE SYSTEM — WEEK 3: UNIFIED PIPELINE")
    print("═" * 62)

    # ── Load the trained model ──
    print("\n📦 Loading Decision Tree model...")
    model, career_encoder = load_model()

   ### TEST 1 — Alex (tech-oriented, high DT confidence)
   # This student has strong scores in math and physics, and interests in computers,
   # which should lead to a confident Software Engineer recommendation from the DT.
   
    print("\n\n" + "━" * 62)
    print("  TEST 1 — Alex (tech-oriented student)")
    print("━" * 62)

    alex_profile = create_student_profile(
        name="Alex",
        interests=["computers", "problem_solving", "math"],
        skills=["logic", "math", "programming"],
        traits=["analytical", "detail_oriented", "curious"],
        preferred_environment="indoors"
    )
    alex_scores = {
        'gender': 1, 'part_time_job': 0, 'absence_days': 3,
        'extracurricular_activities': 0, 'weekly_self_study_hours': 30,
        'math_score': 95, 'history_score': 70, 'physics_score': 92,
        'chemistry_score': 88, 'biology_score': 65,
        'english_score': 75, 'geography_score': 72
    }
    report1 = run_pipeline(alex_profile, alex_scores, model, career_encoder, verbose=True)
    display_report(report1)

    ## TEST 2 — Jordan (healthcare, high rule confidence)
    # This student has strong biology scores and interests in helping people,
    # which should trigger the Forward Chaining engine to recommend Doctor
    print("\n\n" + "━" * 62)
    print("  TEST 2 — Jordan (healthcare-oriented student)")
    print("━" * 62)

    jordan_profile = create_student_profile(
        name="Jordan",
        interests=["biology", "helping_people", "health"],
        skills=["science", "communication", "critical_thinking"],
        traits=["empathetic", "patient", "detail_oriented"],
        preferred_environment="indoors"
    )
    jordan_scores = {
        'gender': 0, 'part_time_job': 0, 'absence_days': 1,
        'extracurricular_activities': 1, 'weekly_self_study_hours': 35,
        'math_score': 78, 'history_score': 80, 'physics_score': 75,
        'chemistry_score': 90, 'biology_score': 96,
        'english_score': 85, 'geography_score': 77
    }
    report2 = run_pipeline(jordan_profile, jordan_scores, model, career_encoder, verbose=True)
    display_report(report2)

    ### TEST 3 — Sam (creative, triggers KG Boost)
    # This student has moderate scores but strong creative interests and traits,
    # which should lead to a low-confidence DT prediction but a strong KG Boost

    print("\n\n" + "━" * 62)
    print("  TEST 3 — Sam (creative student — KG Boost demo)")
    print("━" * 62)

    sam_profile = create_student_profile(
        name="Sam",
        interests=["art", "design", "writing"],
        skills=["creativity", "visual_thinking", "communication"],
        traits=["creative", "expressive", "curious"],
        preferred_environment="indoors"
    )
    sam_scores = {
        'gender': 0, 'part_time_job': 1, 'absence_days': 5,
        'extracurricular_activities': 1, 'weekly_self_study_hours': 20,
        'math_score': 65, 'history_score': 85, 'physics_score': 60,
        'chemistry_score': 62, 'biology_score': 70,
        'english_score': 92, 'geography_score': 80
    }
    report3 = run_pipeline(sam_profile, sam_scores, model, career_encoder, verbose=True)
    display_report(report3)

    ### INTERACTIVE MODE
    # Finally, we offer an interactive questionnaire for users to input their own profiles
    print("\n" + "─" * 62)
    try:
        ans = input("  Would you like to try the interactive questionnaire? (yes/no): ")
        if ans.strip().lower() in ("yes", "y"):
            run_interactive_questionnaire(model, career_encoder)
    except (EOFError, KeyboardInterrupt):
        print("\n  (Interactive mode skipped.)")

    print("\n✅  Week 3 complete — unified pipeline is fully operational!\n")