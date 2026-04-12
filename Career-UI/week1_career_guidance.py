# 
#   KNOWLEDGE-BASED CAREER GUIDANCE SYSTEM
#   Week 1: Research & Planning — Data Structures & Knowledge Base

#
# BEGINNER'S GUIDE TO THIS FILE:
#   - We define what "careers", "interests", "skills", and "traits" are
#   - We build a Knowledge Base (a dictionary of careers and their details)
#   - We implement a Forward Chaining Inference Engine (rule-based logic)
#   - We build a Knowledge Graph (ontology of connected concepts)
#   - Recommendations now include a "reasoning" string explanation
#   - Conflict resolution handles disagreements between rules and the model
#
# Run this file with: python week1_career_guidance.py
#


# -------------------------------------------------------
# SECTION 1: STUDENT PROFILE
# -------------------------------------------------------
# This is the structure we'll use to store a student's answers.
# Think of it like a form the student fills out.

def create_student_profile(name, interests, skills, traits, preferred_environment):
    """
    Creates a dictionary representing a student's profile.

    Parameters:
        name                 : str  — student's name
        interests            : list — topics they enjoy (e.g. ["biology", "computers"])
        skills               : list — what they're good at  (e.g. ["math", "writing"])
        traits               : list — personality traits   (e.g. ["creative", "analytical"])
        preferred_environment: str  — "indoors", "outdoors", or "remote"

    Returns a dictionary (profile) with all the above info.
    """
    profile = {
        "name": name,
        "interests": interests,
        "skills": skills,
        "traits": traits,
        "preferred_environment": preferred_environment
    }
    return profile


# -------------------------------------------------------
# SECTION 2: CAREER KNOWLEDGE BASE
# -------------------------------------------------------
# This is our system's "brain" — a dictionary where:
#   KEY   = career name (string)
#   VALUE = dictionary of what that career requires/offers

CAREER_KNOWLEDGE_BASE = {

    "Software Engineer": {
        "required_interests": ["computers", "technology", "problem_solving"],
        "required_skills":    ["math", "logic", "programming"],
        "ideal_traits":       ["analytical", "detail_oriented", "patient"],
        "environment":        "indoors",
        "description":        "Designs and builds software applications and systems.",
        "related_careers":    ["Data Scientist", "Cybersecurity Analyst", "Web Developer"]
    },

    "Data Scientist": {
        "required_interests": ["computers", "math", "research"],
        "required_skills":    ["math", "statistics", "programming"],
        "ideal_traits":       ["analytical", "curious", "detail_oriented"],
        "environment":        "indoors",
        "description":        "Analyzes large datasets to find patterns and insights.",
        "related_careers":    ["Software Engineer", "Statistician", "Machine Learning Engineer"]
    },

    "Doctor": {
        "required_interests": ["biology", "health", "helping_people"],
        "required_skills":    ["science", "communication", "critical_thinking"],
        "ideal_traits":       ["empathetic", "patient", "detail_oriented"],
        "environment":        "indoors",
        "description":        "Diagnoses and treats illnesses to improve patient health.",
        "related_careers":    ["Nurse", "Pharmacist", "Medical Researcher"]
    },

    "Graphic Designer": {
        "required_interests": ["art", "technology", "design"],
        "required_skills":    ["creativity", "communication", "visual_thinking"],
        "ideal_traits":       ["creative", "detail_oriented", "expressive"],
        "environment":        "indoors",
        "description":        "Creates visual content for brands, media, and products.",
        "related_careers":    ["UI/UX Designer", "Illustrator", "Art Director"]
    },

    "Teacher": {
        "required_interests": ["education", "helping_people", "communication"],
        "required_skills":    ["communication", "patience", "organization"],
        "ideal_traits":       ["empathetic", "patient", "expressive"],
        "environment":        "indoors",
        "description":        "Educates and mentors students in academic subjects.",
        "related_careers":    ["School Counselor", "Education Administrator", "Tutor"]
    },

    "Environmental Scientist": {
        "required_interests": ["nature", "biology", "research"],
        "required_skills":    ["science", "critical_thinking", "writing"],
        "ideal_traits":       ["curious", "analytical", "passionate"],
        "environment":        "outdoors",
        "description":        "Studies the environment and works to solve ecological problems.",
        "related_careers":    ["Biologist", "Conservation Officer", "Geologist"]
    },

    "Journalist": {
        "required_interests": ["writing", "communication", "current_events"],
        "required_skills":    ["writing", "communication", "research"],
        "ideal_traits":       ["curious", "expressive", "persistent"],
        "environment":        "indoors",
        "description":        "Investigates and reports on news and important events.",
        "related_careers":    ["Content Writer", "Editor", "Broadcaster"]
    },

    "Cybersecurity Analyst": {
        "required_interests": ["computers", "technology", "problem_solving"],
        "required_skills":    ["logic", "programming", "critical_thinking"],
        "ideal_traits":       ["analytical", "detail_oriented", "curious"],
        "environment":        "indoors",
        "description":        "Protects computer systems and networks from cyber threats.",
        "related_careers":    ["Software Engineer", "Network Administrator", "Ethical Hacker"]
    },

    "Architect": {
        "required_interests": ["design", "math", "art"],
        "required_skills":    ["math", "creativity", "visual_thinking"],
        "ideal_traits":       ["creative", "detail_oriented", "analytical"],
        "environment":        "indoors",
        "description":        "Designs buildings and structures for function and aesthetics.",
        "related_careers":    ["Civil Engineer", "Interior Designer", "Urban Planner"]
    },

    "Nurse": {
        "required_interests": ["health", "helping_people", "biology"],
        "required_skills":    ["communication", "science", "critical_thinking"],
        "ideal_traits":       ["empathetic", "patient", "calm_under_pressure"],
        "environment":        "indoors",
        "description":        "Provides direct patient care and supports medical teams.",
        "related_careers":    ["Doctor", "Pharmacist", "Physical Therapist"]
    },
}


# -------------------------------------------------------
# SECTION 3: KNOWLEDGE GRAPH ONTOLOGY
# -------------------------------------------------------
# An Ontology is a formal map of concepts and their relationships.
# Our Knowledge Graph IS our ontology — it defines the nodes (concepts)
# and edges (connections) of the career guidance domain.
#
# NODES in our ontology:
#   - Interest nodes  : computers, biology, math, art, writing ...
#   - Trait nodes     : analytical, creative, empathetic, curious ...
#   - Career nodes    : Software Engineer, Doctor, Teacher ...
#
# EDGES (relationships):
#   - interest  ──LEADS_TO──>  career
#   - trait     ──SUITS──>     career
#   - career    ──RELATED_TO─> career  (stored in CAREER_KNOWLEDGE_BASE)
#
# Diagram (simplified):
#
#   [computers] ──LEADS_TO──> [Software Engineer] ──RELATED_TO──> [Data Scientist]
#   [biology]   ──LEADS_TO──> [Doctor]            ──RELATED_TO──> [Nurse]
#   [analytical]──SUITS────> [Software Engineer]
#   [empathetic]──SUITS────> [Doctor]
#
# In Week 3, we'll traverse these edges to suggest related careers.

KNOWLEDGE_GRAPH = {
    # Interests → Careers they relate to
    "interests": {
        "computers":       ["Software Engineer", "Data Scientist", "Cybersecurity Analyst"],
        "biology":         ["Doctor", "Nurse", "Environmental Scientist"],
        "art":             ["Graphic Designer", "Architect"],
        "math":            ["Data Scientist", "Software Engineer", "Architect"],
        "writing":         ["Journalist", "Teacher"],
        "helping_people":  ["Doctor", "Nurse", "Teacher"],
        "nature":          ["Environmental Scientist"],
        "design":          ["Graphic Designer", "Architect"],
        "research":        ["Data Scientist", "Environmental Scientist", "Journalist"],
        "health":          ["Doctor", "Nurse"],
        "communication":   ["Teacher", "Journalist"],
        "problem_solving": ["Software Engineer", "Cybersecurity Analyst"],
        "current_events":  ["Journalist"],
        "education":       ["Teacher"],
        "technology":      ["Software Engineer", "Cybersecurity Analyst", "Graphic Designer"],
    },

    # Traits → Careers they suit
    "traits": {
        "analytical":        ["Software Engineer", "Data Scientist", "Cybersecurity Analyst", "Architect"],
        "creative":          ["Graphic Designer", "Architect"],
        "empathetic":        ["Doctor", "Nurse", "Teacher"],
        "curious":           ["Data Scientist", "Environmental Scientist", "Journalist"],
        "detail_oriented":   ["Software Engineer", "Data Scientist", "Graphic Designer", "Nurse"],
        "expressive":        ["Graphic Designer", "Journalist", "Teacher"],
        "patient":           ["Doctor", "Teacher", "Nurse"],
    }
}


# -------------------------------------------------------
# SECTION 4: FORWARD CHAINING INFERENCE ENGINE
# -------------------------------------------------------
# This is the core AI logic of our system.
#
# WHAT IS FORWARD CHAINING?
#   Forward Chaining is an inference strategy used in expert systems.
#   It starts with KNOWN FACTS (the student's profile) and repeatedly
#   applies IF-THEN rules to move FORWARD toward a CONCLUSION (career).
#
#   Direction:  Facts → Rules → Conclusion
#   Example:
#     FACT:  student likes "computers" and is "analytical"
#     RULE:  IF computers AND analytical THEN consider Software Engineer
#     CONCLUSION: Recommend Software Engineer (score: 80.8%)
#
# This is the opposite of Backward Chaining, which starts from a
# goal and works backwards to check if the facts support it.

def calculate_match_score(student_profile, career_name):
    """
    Applies forward chaining rules to score a student against a career.

    How scoring works:
      - Each matching interest  = +3 points
      - Each matching skill     = +3 points
      - Each matching trait     = +2 points
      - Matching environment    = +2 points

    Returns (score_percentage, reasoning_string).
    """
    career = CAREER_KNOWLEDGE_BASE[career_name]
    score = 0
    max_score = 0
    matched_interests = []
    matched_skills    = []
    matched_traits    = []

    # --- Rule 1: Check Interests ---
    for interest in career["required_interests"]:
        max_score += 3
        if interest in student_profile["interests"]:
            score += 3
            matched_interests.append(interest)

    # --- Rule 2: Check Skills ---
    for skill in career["required_skills"]:
        max_score += 3
        if skill in student_profile["skills"]:
            score += 3
            matched_skills.append(skill)

    # --- Rule 3: Check Traits ---
    for trait in career["ideal_traits"]:
        max_score += 2
        if trait in student_profile["traits"]:
            score += 2
            matched_traits.append(trait)

    # --- Rule 4: Check Environment ---
    max_score += 2
    env_match = student_profile["preferred_environment"] == career["environment"]
    if env_match:
        score += 2

    # Convert to percentage
    final_score = round((score / max_score) * 100, 1) if max_score > 0 else 0

    # --- Build reasoning string ---
    reasons = []
    if matched_interests:
        reasons.append(f"shared interests in {', '.join(matched_interests)}")
    if matched_skills:
        reasons.append(f"matching skills: {', '.join(matched_skills)}")
    if matched_traits:
        reasons.append(f"aligned traits: {', '.join(matched_traits)}")
    if env_match:
        reasons.append(f"preferred environment ({student_profile['preferred_environment']}) matches")

    if reasons:
        reasoning = f"Recommended based on {'; '.join(reasons)}."
    else:
        reasoning = "Low match — few overlapping interests, skills, or traits."

    return final_score, reasoning


def resolve_conflict(rule_based_top, decision_tree_result=None):
    """
    CONFLICT RESOLUTION STRATEGY
    ─────────────────────────────
    Handles disagreements between the Rule-Based Engine and the Decision Tree.

    Strategy used: WEIGHTED CONFIDENCE
      - If the rule-based score for its top pick is >= 70%  → trust rules (high confidence)
      - If the rule-based score is < 70% AND a Decision Tree result exists → prefer the tree
      - If both agree → no conflict, return as normal
      - If both disagree with low confidence → return BOTH with a note

    Parameters:
        rule_based_top      : dict — top result from the forward chaining engine
        decision_tree_result: str  — career predicted by Decision Tree (Week 2, optional)

    Returns a conflict resolution note string.
    """
    rule_career = rule_based_top["career"]
    rule_score  = rule_based_top["score"]

    # No Decision Tree yet (Week 1 only) — no conflict possible
    if decision_tree_result is None:
        return None

    # No conflict — both agree
    if rule_career == decision_tree_result:
        return f"✅ No conflict: Both engines agree on '{rule_career}'."

    # Conflict detected — apply resolution strategy
    if rule_score >= 70:
        return (
            f"⚖️  Conflict resolved → Rules win: '{rule_career}' (score: {rule_score}%) "
            f"overrides Decision Tree suggestion of '{decision_tree_result}' "
            f"due to high rule confidence (≥70%)."
        )
    else:
        return (
            f"⚖️  Conflict resolved → Decision Tree wins: '{decision_tree_result}' preferred "
            f"over '{rule_career}' (score: {rule_score}%) "
            f"because rule confidence is below 70% threshold."
        )


def get_career_recommendations(student_profile, top_n=3, decision_tree_result=None):
    """
    FORWARD CHAINING INFERENCE ENGINE
    ───────────────────────────────────
    Starts with known facts (student profile) and applies inference rules
    to move forward toward career conclusions — classic Forward Chaining.

    Parameters:
        student_profile      : dict — the student's profile
        top_n                : int  — how many recommendations to return (default: 3)
        decision_tree_result : str  — optional prediction from Decision Tree (Week 2)

    Returns a list of recommendation dicts, each containing:
        {
            "career"   : career name,
            "score"    : match percentage,
            "reasoning": plain-English explanation of why this career was recommended
        }
    Plus a conflict_note if the Decision Tree disagrees with the top pick.
    """
    results = []

    # ── FORWARD CHAINING: apply rules to every career ──
    for career_name in CAREER_KNOWLEDGE_BASE:
        score, reasoning = calculate_match_score(student_profile, career_name)
        results.append({
            "career":    career_name,
            "score":     score,
            "reasoning": reasoning
        })

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    top_results = results[:top_n]

    # ── CONFLICT RESOLUTION (between rules and Decision Tree) ──
    conflict_note = resolve_conflict(top_results[0], decision_tree_result)

    return top_results, conflict_note


# -------------------------------------------------------
# SECTION 5: DISPLAY RESULTS
# -------------------------------------------------------

def display_recommendations(student_profile, recommendations, conflict_note=None):
    """Prints career recommendations with reasoning and conflict notes."""

    print("\n" + "=" * 60)
    print(f"  🎓 CAREER RECOMMENDATIONS FOR: {student_profile['name'].upper()}")
    print("=" * 60)

    print(f"\n  Your Interests : {', '.join(student_profile['interests'])}")
    print(f"  Your Skills    : {', '.join(student_profile['skills'])}")
    print(f"  Your Traits    : {', '.join(student_profile['traits'])}")
    print(f"  Environment    : {student_profile['preferred_environment']}")

    print("\n  TOP CAREER MATCHES  [Forward Chaining Inference Engine]")
    print("  " + "-" * 55)

    for rank, rec in enumerate(recommendations, start=1):
        career_info = CAREER_KNOWLEDGE_BASE[rec["career"]]
        print(f"\n  #{rank}  {rec['career']}  —  Match: {rec['score']}%")
        print(f"       📋 {career_info['description']}")
        print(f"       💡 Reasoning: {rec['reasoning']}")
        print(f"       🔗 Related: {', '.join(career_info['related_careers'])}")

    if conflict_note:
        print(f"\n  {conflict_note}")

    print("\n" + "=" * 60)


# -------------------------------------------------------
# SECTION 6: RUN A SAMPLE STUDENT (TEST IT!)
# -------------------------------------------------------

if __name__ == "__main__":

    print("\n🚀 Welcome to the Knowledge-Based Career Guidance System")
    print("   Week 1: Forward Chaining Inference Engine + Conflict Resolution\n")

    # ── TEST 1: Alex — Tech profile ──
    sample_student = create_student_profile(
        name="Alex",
        interests=["computers", "problem_solving", "math"],
        skills=["logic", "math", "programming"],
        traits=["analytical", "detail_oriented", "curious"],
        preferred_environment="indoors"
    )

    recommendations, conflict_note = get_career_recommendations(sample_student, top_n=3)
    display_recommendations(sample_student, recommendations, conflict_note)

    # ── TEST 2: Jordan — Healthcare profile ──
    print("\n\n--- Testing with a second student profile ---")
    student_2 = create_student_profile(
        name="Jordan",
        interests=["biology", "helping_people", "health"],
        skills=["science", "communication", "critical_thinking"],
        traits=["empathetic", "patient", "detail_oriented"],
        preferred_environment="indoors"
    )

    recommendations_2, conflict_note_2 = get_career_recommendations(student_2, top_n=3)
    display_recommendations(student_2, recommendations_2, conflict_note_2)

    # ── TEST 3: Conflict resolution demo ──
    # Simulate a case where the Decision Tree (Week 2) disagrees with the rules
    print("\n\n--- Conflict Resolution Demo ---")
    print("    (Simulating a future Decision Tree predicting 'Artist'")
    print("     while rules recommend 'Software Engineer')\n")

    student_3 = create_student_profile(
        name="Sam",
        interests=["computers", "art", "design"],
        skills=["logic", "creativity", "visual_thinking"],
        traits=["analytical", "creative", "expressive"],
        preferred_environment="indoors"
    )

    # Pass a hypothetical Decision Tree result to trigger conflict handling
    recommendations_3, conflict_note_3 = get_career_recommendations(
        student_3, top_n=3, decision_tree_result="Artist"
    )
    display_recommendations(student_3, recommendations_3, conflict_note_3)