# ============================================================
#   KNOWLEDGE-BASED CAREER GUIDANCE SYSTEM — Flask App
#   Integrates Week 1 + Week 2 + Week 3
# ============================================================
import os, csv, pickle
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__)

_mem_profiles = []
_mem_students = []

MONGO_AVAILABLE = False
students_col = profiles_col = recommend_col = None
try:
    from pymongo import MongoClient
    _client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
    _client.server_info()
    _db = _client["careerpath"]
    students_col  = _db["students"]
    profiles_col  = _db["profiles"]
    recommend_col = _db["recommendations"]
    MONGO_AVAILABLE = True
    print("✅ MongoDB connected.")
except Exception:
    print("⚠️  MongoDB unavailable — using in-memory storage.")

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "career_model.pkl")
with open(_MODEL_PATH, "rb") as f:
    _bundle      = pickle.load(f)
    DT_MODEL     = _bundle["model"]
    DT_ENCODER   = _bundle["encoder"]
    FEATURE_COLS = _bundle["feature_cols"]
print(f"✅ Model loaded. Careers: {list(DT_ENCODER.classes_)}")

CAREER_KNOWLEDGE_BASE = {
    "Software Engineer":      {"required_interests":["computers","technology","problem_solving"],"required_skills":["math","logic","programming"],           "ideal_traits":["analytical","detail_oriented","patient"],    "description":"Designs and builds software applications and systems.",       "related_careers":["Game Developer","Accountant","Stock Investor"]},
    "Doctor":                 {"required_interests":["biology","health","helping_people"],       "required_skills":["science","communication","critical_thinking"],"ideal_traits":["empathetic","patient","detail_oriented"],  "description":"Diagnoses and treats illnesses to improve patient health.",     "related_careers":["Scientist","Teacher","Government Officer"]},
    "Lawyer":                 {"required_interests":["law","communication","helping_people"],    "required_skills":["communication","writing","critical_thinking"], "ideal_traits":["analytical","expressive","persistent"],     "description":"Advises clients and represents them in legal matters.",         "related_careers":["Government Officer","Writer","Business Owner"]},
    "Teacher":                {"required_interests":["education","helping_people","communication"],"required_skills":["communication","patience","organization"],    "ideal_traits":["empathetic","patient","expressive"],        "description":"Educates and mentors students in academic subjects.",           "related_careers":["Government Officer","Doctor","Social Network Studies"]},
    "Scientist":              {"required_interests":["research","biology","math"],               "required_skills":["science","math","critical_thinking"],           "ideal_traits":["curious","analytical","detail_oriented"],   "description":"Conducts research and experiments to advance knowledge.",       "related_careers":["Doctor","Construction Engineer","Stock Investor"]},
    "Business Owner":         {"required_interests":["communication","problem_solving","law"],  "required_skills":["math","communication","critical_thinking"],     "ideal_traits":["analytical","expressive","persistent"],     "description":"Starts and manages a business enterprise.",                    "related_careers":["Real Estate Developer","Stock Investor","Banker"]},
    "Accountant":             {"required_interests":["math","problem_solving"],                 "required_skills":["math","logic","organization"],                  "ideal_traits":["detail_oriented","analytical","patient"],    "description":"Manages financial records and ensures compliance.",            "related_careers":["Stock Investor","Banker","Business Owner"]},
    "Banker":                 {"required_interests":["math","communication","problem_solving"], "required_skills":["math","communication","critical_thinking"],     "ideal_traits":["analytical","detail_oriented","expressive"], "description":"Manages financial services and banking operations.",           "related_careers":["Accountant","Lawyer","Stock Investor"]},
    "Designer":               {"required_interests":["art","technology","problem_solving"],     "required_skills":["creativity","visual_thinking","communication"], "ideal_traits":["creative","detail_oriented","expressive"],   "description":"Creates visual concepts and user experiences.",                "related_careers":["Artist","Game Developer","Software Engineer"]},
    "Artist":                 {"required_interests":["art","education","communication"],        "required_skills":["creativity","visual_thinking","writing"],       "ideal_traits":["creative","expressive","curious"],           "description":"Creates original artwork and creative expressions.",           "related_careers":["Designer","Writer","Teacher"]},
    "Writer":                 {"required_interests":["communication","education","art"],        "required_skills":["writing","communication","research"],           "ideal_traits":["expressive","curious","persistent"],         "description":"Creates written content for various media.",                   "related_careers":["Lawyer","Teacher","Artist"]},
    "Game Developer":         {"required_interests":["computers","technology","art","problem_solving"],"required_skills":["math","logic","programming"],           "ideal_traits":["analytical","creative","detail_oriented"],   "description":"Designs and builds interactive video games.",                  "related_careers":["Software Engineer","Designer","Construction Engineer"]},
    "Construction Engineer":  {"required_interests":["problem_solving","math","technology"],   "required_skills":["math","science","critical_thinking"],           "ideal_traits":["analytical","detail_oriented","patient"],    "description":"Plans and oversees construction and infrastructure projects.",  "related_careers":["Game Developer","Scientist","Real Estate Developer"]},
    "Stock Investor":         {"required_interests":["math","problem_solving","research"],     "required_skills":["math","critical_thinking","communication"],     "ideal_traits":["analytical","curious","persistent"],         "description":"Analyses markets and manages investment portfolios.",           "related_careers":["Accountant","Banker","Business Owner"]},
    "Real Estate Developer":  {"required_interests":["law","math","communication"],            "required_skills":["math","communication","critical_thinking"],     "ideal_traits":["analytical","expressive","persistent"],     "description":"Develops and manages property and real estate.",                "related_careers":["Business Owner","Government Officer","Construction Engineer"]},
    "Government Officer":     {"required_interests":["law","communication","helping_people","education"],"required_skills":["communication","writing","organization"],"ideal_traits":["empathetic","expressive","patient"],      "description":"Works in public service and policy implementation.",            "related_careers":["Lawyer","Teacher","Social Network Studies"]},
    "Social Network Studies": {"required_interests":["communication","technology","education","helping_people"],"required_skills":["communication","writing","research"],"ideal_traits":["expressive","curious","empathetic"],    "description":"Studies and manages social media and digital communities.",    "related_careers":["Teacher","Government Officer","Designer"]},
}

RIASEC_MAP   = {"Construction Engineer":"R","Software Engineer":"R","Game Developer":"R","Doctor":"I","Scientist":"I","Stock Investor":"I","Artist":"A","Writer":"A","Designer":"A","Teacher":"S","Government Officer":"S","Social Network Studies":"S","Lawyer":"E","Business Owner":"E","Real Estate Developer":"E","Accountant":"C","Banker":"C"}
RIASEC_NAMES = {"R":"Realistic","I":"Investigative","A":"Artistic","S":"Social","E":"Enterprising","C":"Conventional"}
ARTISTIC_CAREERS   = {"Artist","Writer","Designer"}
KG_BOOST_THRESHOLD = 0.35
CAREER_COLORS = {"Software Engineer":"#4f46e5","Doctor":"#dc2626","Lawyer":"#b45309","Teacher":"#0369a1","Scientist":"#0891b2","Business Owner":"#7c3aed","Accountant":"#065f46","Banker":"#1e3a5f","Designer":"#be185d","Artist":"#9d174d","Writer":"#92400e","Game Developer":"#6d28d9","Construction Engineer":"#d97706","Stock Investor":"#15803d","Real Estate Developer":"#7e22ce","Government Officer":"#1d4ed8","Social Network Studies":"#0e7490"}

def forward_chaining_score(interests, skills, traits):
    scores, reasons = {}, {}
    for name, career in CAREER_KNOWLEDGE_BASE.items():
        score     = 0
        max_score = len(career["required_interests"])*3 + len(career["required_skills"])*3 + len(career["ideal_traits"])*2
        matched   = []
        for i in career["required_interests"]:
            if i in interests: score += 3; matched.append(f"interest: {i}")
        for s in career["required_skills"]:
            if s in skills:    score += 3; matched.append(f"skill: {s}")
        for t in career["ideal_traits"]:
            if t in traits:    score += 2; matched.append(f"trait: {t}")
        scores[name]  = score / max_score if max_score > 0 else 0
        reasons[name] = matched
    return scores, reasons

def decision_tree_predict(gender, part_time, absence, extracurricular, study_hours, math, history, physics, chemistry, biology, english, geography):
    row   = pd.DataFrame([[gender,part_time,absence,extracurricular,study_hours,math,history,physics,chemistry,biology,english,geography]], columns=FEATURE_COLS)
    probs = DT_MODEL.predict_proba(row)[0]
    d     = dict(zip(DT_ENCODER.classes_, probs))
    top3  = sorted(d.items(), key=lambda x: x[1], reverse=True)[:3]
    return d, top3

def knowledge_graph_boost(dt_probs, fc_scores, interests, traits):
    top_dt  = max(dt_probs, key=dt_probs.get)
    top_dtc = dt_probs[top_dt]
    top_fc  = max(fc_scores, key=fc_scores.get) if fc_scores else top_dt
    top_fcc = fc_scores.get(top_fc, 0)
    kg_boost_applied = False
    conflict_winner  = None

    if top_dt != top_fc:
        if top_fcc >= 0.70:
            conflict_note   = f"Rules Engine wins: '{top_fc}' ({top_fcc*100:.0f}% rule confidence ≥ 70%) overrides Decision Tree's '{top_dt}'."
            conflict_winner = "rules"
        elif top_dtc < KG_BOOST_THRESHOLD:
            conflict_note   = f"Low DT confidence ({top_dtc*100:.0f}%) — Knowledge Graph Boost applied as tie-breaker."
            conflict_winner = "kg"; kg_boost_applied = True
        else:
            conflict_note   = f"Decision Tree preferred '{top_dt}' over Rules suggestion '{top_fc}'."
            conflict_winner = "dt"
    else:
        conflict_note   = f"✅ Both engines agree: '{top_dt}'."
        conflict_winner = "agree"

    creative_interests = {"art","design","writing","music","drawing"}
    creative_traits    = {"creative","expressive","imaginative"}
    art_signal = len(creative_interests & set(interests)) + len(creative_traits & set(traits))
    dt_w = 0.50 if kg_boost_applied else 0.70
    fc_w = 0.50 if kg_boost_applied else 0.30

    final = {}
    for career in CAREER_KNOWLEDGE_BASE:
        s = dt_probs.get(career,0)*dt_w + fc_scores.get(career,0)*fc_w
        if career in ARTISTIC_CAREERS and kg_boost_applied and art_signal > 0:
            s += art_signal * 0.04
        final[career] = s

    trace = {
        "dt_top": top_dt, "dt_confidence": round(top_dtc*100,1),
        "fc_top": top_fc, "fc_confidence": round(top_fcc*100,1),
        "kg_boost": kg_boost_applied, "conflict_winner": conflict_winner,
        "conflict_note": conflict_note,
        "blend_weights": f"DT {int(dt_w*100)}% + Rules {int(fc_w*100)}%",
    }
    return final, conflict_note, kg_boost_applied, trace

def get_recommendations(interests, skills, traits, math, history, physics, chemistry, biology, english, geography, study_hours, gender=0, part_time=0, absence=0, extracurricular=0):
    dt_probs, dt_top3 = decision_tree_predict(gender,part_time,absence,extracurricular,study_hours,math,history,physics,chemistry,biology,english,geography)
    fc_scores, fc_reasons = forward_chaining_score(interests, skills, traits)
    final_scores, conflict_note, kg_boost, trace = knowledge_graph_boost(dt_probs, fc_scores, interests, traits)
    sorted_careers = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    top_val = sorted_careers[0][1] if sorted_careers[0][1] > 0 else 1
    results = []
    for career, score in sorted_careers:
        pct = min(round((score/top_val)*85+10), 95)
        kb  = CAREER_KNOWLEDGE_BASE.get(career,{})
        riasec = RIASEC_MAP.get(career,"R")
        results.append({
            "name": career, "score": pct, "riasec": riasec,
            "riasec_name": RIASEC_NAMES.get(riasec, riasec),
            "color": CAREER_COLORS.get(career,"#555"),
            "related": kb.get("related_careers",[]),
            "description": kb.get("description",""),
            "kg_boost": kg_boost, "conflict_note": conflict_note,
            "fc_reasons": fc_reasons.get(career,[])[:4],
        })
    results[0]["pipeline_trace"] = trace
    results[0]["dt_top3"] = [{"name":c,"pct":round(p*100,1)} for c,p in dt_top3]
    results[0]["fc_top3"] = sorted([{"name":c,"pct":round(s*100,1)} for c,s in fc_scores.items()], key=lambda x:x["pct"], reverse=True)[:3]
    return results

@app.route("/")
def index(): return render_template("index.html")

@app.route("/students")
def students_page(): return render_template("students.html")

@app.route("/status")
def status(): return jsonify({"mongo": MONGO_AVAILABLE, "model": "loaded"})

@app.route("/recommend", methods=["POST"])
def recommend():
    d = request.json
    name=d.get("name","Unknown"); interests=d.get("interests",[]); skills=d.get("skills",[]); traits=d.get("traits",[])
    math=int(d.get("math",70)); history=int(d.get("history",70)); physics=int(d.get("physics",70))
    chemistry=int(d.get("chemistry",70)); biology=int(d.get("biology",70)); english=int(d.get("english",70))
    geography=int(d.get("geography",70)); study_hours=int(d.get("study_hours",10))
    gender=int(d.get("gender",0)); part_time=int(d.get("part_time",0))
    absence=int(d.get("absence",2)); extracurricular=int(d.get("extracurricular",0))
    results = get_recommendations(interests,skills,traits,math,history,physics,chemistry,biology,english,geography,study_hours,gender,part_time,absence,extracurricular)
    doc = {"name":name,"interests":interests,"skills":skills,"traits":traits,"math":math,"history":history,"physics":physics,"chemistry":chemistry,"biology":biology,"english":english,"geography":geography,"study_hours":study_hours,"created_at":datetime.utcnow().isoformat()}
    if MONGO_AVAILABLE:
        profiles_col.insert_one({**doc}); recommend_col.insert_one({"name":name,"recommendations":results,"created_at":datetime.utcnow().isoformat()})
    else:
        _mem_profiles.insert(0, doc)
        if len(_mem_profiles) > 100: _mem_profiles.pop()
    return jsonify(results)

@app.route("/import-csv", methods=["POST"])
def import_csv():
    csv_path = request.json.get("path","data/student-scores-6k.csv")
    if not os.path.exists(csv_path): return jsonify({"error":f"File not found: {csv_path}"}), 404
    records = []
    with open(csv_path,newline="",encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append({"id":int(row["id"]),"first_name":row["first_name"],"last_name":row["last_name"],"email":row["email"],"gender":row["gender"],"part_time_job":row["part_time_job"]=="True","absence_days":int(row["absence_days"]),"extracurricular":row["extracurricular_activities"]=="True","study_hours":int(row["weekly_self_study_hours"]),"career_aspiration":row["career_aspiration"],"scores":{"math":int(row["math_score"]),"history":int(row["history_score"]),"physics":int(row["physics_score"]),"chemistry":int(row["chemistry_score"]),"biology":int(row["biology_score"]),"english":int(row["english_score"]),"geography":int(row["geography_score"])}})
    if MONGO_AVAILABLE: students_col.delete_many({}); students_col.insert_many(records) if records else None
    else: _mem_students.clear(); _mem_students.extend(records)
    return jsonify({"imported":len(records),"message":"CSV imported successfully!"})

@app.route("/api/profiles")
def get_profiles():
    if MONGO_AVAILABLE: data = list(profiles_col.find({},{"_id":0}).sort("created_at",-1).limit(50))
    else: data = _mem_profiles[:50]
    return jsonify(data)

@app.route("/api/students")
def get_students():
    page=int(request.args.get("page",1)); per_page=int(request.args.get("per_page",20)); search=request.args.get("search","").lower()
    if MONGO_AVAILABLE:
        query = {"$or":[{"first_name":{"$regex":search,"$options":"i"}},{"last_name":{"$regex":search,"$options":"i"}},{"career_aspiration":{"$regex":search,"$options":"i"}}]} if search else {}
        total=students_col.count_documents(query); students=list(students_col.find(query,{"_id":0}).skip((page-1)*per_page).limit(per_page))
    else:
        filtered=[s for s in _mem_students if not search or search in s["first_name"].lower() or search in s["last_name"].lower() or search in s["career_aspiration"].lower()]
        total=len(filtered); students=filtered[(page-1)*per_page:(page-1)*per_page+per_page]
    return jsonify({"students":students,"total":total,"page":page})

if __name__ == "__main__":
    app.run(debug=True)
