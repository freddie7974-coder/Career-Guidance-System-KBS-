// ════════════════════════════════════════════════════════════
//  CAREER GUIDANCE SYSTEM — main.js
//  Handles: chip selection, recommendations, results display,
//           pipeline trace rendering, CSV student search, modals
// ════════════════════════════════════════════════════════════

// ── Chip toggle ──────────────────────────────────────────────
document.querySelectorAll('.chip').forEach(chip =>
  chip.addEventListener('click', () => chip.classList.toggle('selected'))
);

// ── MongoDB status badge ─────────────────────────────────────
fetch('/status').then(r => r.json()).then(d => {
  const b = document.getElementById('mongo-badge');
  if (!b) return;
  if (d.mongo) { b.textContent = '● MongoDB'; b.className = 'status-pill mongo-on'; }
  else         { b.textContent = '● In-Memory Mode'; b.className = 'status-pill mongo-off'; }
}).catch(() => {});

// ── Career metadata ──────────────────────────────────────────
const CAREER_META = {
  "Software Engineer":      { icon:"💻", skills:"Math, Programming, Logic",           degree:"BSc Computer Science / Software Engineering" },
  "Doctor":                 { icon:"🩺", skills:"Biology, Chemistry, Communication",   degree:"MBChB / Bachelor of Medicine" },
  "Lawyer":                 { icon:"⚖️", skills:"Critical Thinking, Communication",    degree:"LLB Bachelor of Laws" },
  "Teacher":                { icon:"📚", skills:"Communication, Patience, Empathy",    degree:"BEd Bachelor of Education" },
  "Scientist":              { icon:"🔬", skills:"Research, Analytical Thinking",       degree:"BSc in relevant science field" },
  "Business Owner":         { icon:"🏢", skills:"Leadership, Financial Planning",      degree:"BBA Business Administration" },
  "Accountant":             { icon:"🧮", skills:"Mathematics, Attention to Detail",    degree:"BCom Accounting" },
  "Banker":                 { icon:"🏦", skills:"Mathematics, Communication",          degree:"BCom Finance / Economics" },
  "Designer":               { icon:"🎨", skills:"Creativity, Visual Thinking",         degree:"BDes Bachelor of Design" },
  "Artist":                 { icon:"🖌️", skills:"Creativity, Fine Arts",               degree:"BFA Bachelor of Fine Arts" },
  "Writer":                 { icon:"✍️", skills:"Writing, Research, Creativity",       degree:"BA Journalism / English Literature" },
  "Game Developer":         { icon:"🎮", skills:"Programming, Creativity, Math",       degree:"BSc Computer Science / Game Design" },
  "Construction Engineer":  { icon:"🏗️", skills:"Physics, Mathematics, Problem Solving",degree:"BEng Civil / Construction Engineering" },
  "Stock Investor":         { icon:"📈", skills:"Mathematics, Analysis, Economics",   degree:"BCom Finance / Economics" },
  "Real Estate Developer":  { icon:"🏘️", skills:"Geography, Business, Negotiation",   degree:"BCom Property Studies" },
  "Government Officer":     { icon:"🏛️", skills:"Communication, History, Leadership", degree:"BPA Public Administration" },
  "Social Network Studies": { icon:"🌐", skills:"Communication, Technology, Research",degree:"BA Media & Communications" },
};

const RIASEC_FULL = {
  R:{ name:"Realistic",     color:"#4f46e5", desc:"Practical, hands-on problem solving" },
  I:{ name:"Investigative", color:"#0891b2", desc:"Analytical, intellectual curiosity" },
  A:{ name:"Artistic",      color:"#be185d", desc:"Creative, expressive, imaginative" },
  S:{ name:"Social",        color:"#0369a1", desc:"Helping, teaching, serving others" },
  E:{ name:"Enterprising",  color:"#b45309", desc:"Leading, persuading, managing" },
  C:{ name:"Conventional",  color:"#065f46", desc:"Organising, data, detail-oriented" },
};

// Subject-based rules explanation
const SUBJECT_RULES = {
  "Software Engineer":      [["math",70,"Strong Math score supports technical computing"],["physics",65,"Physics background supports algorithm thinking"]],
  "Doctor":                 [["biology",70,"High Biology aligns with medical knowledge"],["chemistry",70,"Chemistry essential for pharmacology"]],
  "Lawyer":                 [["english",70,"High English supports legal writing"],["history",65,"History knowledge supports legal precedents"]],
  "Teacher":                [["english",70,"Strong English supports classroom communication"],["history",60,"History broadens teaching capability"]],
  "Scientist":              [["biology",68,"Biology supports scientific methodology"],["chemistry",68,"Chemistry is core to laboratory sciences"]],
  "Accountant":             [["math",75,"Excellent Math is the foundation of accounting"],["english",65,"English supports report writing"]],
  "Banker":                 [["math",70,"Strong Math supports financial calculations"],["english",65,"Communication essential for client banking"]],
  "Business Owner":         [["math",65,"Math supports financial planning"],["english",65,"English supports business communication"]],
  "Designer":               [["english",60,"Creative expression supported by English"],["math",60,"Spatial thinking linked to Math"]],
  "Artist":                 [["english",62,"Artistic expression linked to communication"],["history",60,"Art history provides cultural context"]],
  "Writer":                 [["english",75,"Excellent English essential for writing careers"],["history",65,"History enriches writing depth"]],
  "Game Developer":         [["math",70,"Math critical for game physics and logic"],["physics",65,"Physics supports realistic game simulation"]],
  "Construction Engineer":  [["physics",72,"Physics is the foundation of structural engineering"],["math",70,"Mathematics essential for calculations"]],
  "Stock Investor":         [["math",75,"Strong Math supports financial analysis"],["history",65,"Economic history helps predict market trends"]],
  "Real Estate Developer":  [["geography",65,"Geography supports site selection"],["math",65,"Math supports property valuation"]],
  "Government Officer":     [["history",70,"History aligns with policy and governance"],["english",68,"English essential for public service"]],
  "Social Network Studies": [["english",68,"Communication core to social media management"],["geography",60,"Geography supports global social trends"]],
};


// ════════════════════════════════════════════════════════════
//  GET RECOMMENDATIONS
// ════════════════════════════════════════════════════════════
function getRecommendations() {
  const name        = document.getElementById('name').value.trim() || 'Student';
  const interests   = [...document.querySelectorAll('.chip.selected')].map(c => c.dataset.val);
  const scores      = {
    math:      +document.getElementById('math').value,
    history:   +document.getElementById('history').value,
    physics:   +document.getElementById('physics').value,
    chemistry: +document.getElementById('chemistry').value,
    biology:   +document.getElementById('biology').value,
    english:   +document.getElementById('english').value,
    geography: +document.getElementById('geography').value,
    study_hours: +document.getElementById('study_hours').value,
  };

  // Derive skills from scores
  const skills = [];
  if (scores.math >= 75)     skills.push('math');
  if (scores.math >= 80)     skills.push('logic');
  if (scores.physics >= 75)  skills.push('science');
  if (scores.biology >= 75)  skills.push('science');
  if (scores.chemistry >= 75) skills.push('science');
  if (scores.english >= 75)  skills.push('communication', 'writing');
  if (scores.study_hours >= 20) skills.push('organization');

  // Derive traits from interests
  const traitMap = { computers:'analytical', math:'analytical', research:'curious', biology:'detail_oriented', art:'creative', communication:'expressive', helping_people:'empathetic', education:'patient', law:'persistent', problem_solving:'analytical', technology:'detail_oriented' };
  const traits = [...new Set(interests.map(i => traitMap[i]).filter(Boolean))];

  // Show loading
  document.getElementById('rec-placeholder').style.display = 'none';
  document.getElementById('rec-result').style.display      = 'none';
  document.getElementById('trace-panel').style.display     = 'none';
  document.getElementById('rec-loading').style.display     = 'block';

  fetch('/recommend', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, interests, skills, traits, ...scores })
  })
  .then(r => r.json())
  .then(results => {
    document.getElementById('rec-loading').style.display = 'none';
    showResults(name, results, interests, scores);
  })
  .catch(() => {
    document.getElementById('rec-loading').style.display = 'none';
    document.getElementById('rec-placeholder').style.display = 'block';
    alert('Error connecting to the server. Make sure app.py is running.');
  });
}


// ════════════════════════════════════════════════════════════
//  SHOW RESULTS
// ════════════════════════════════════════════════════════════
function showResults(name, results, interests, scores) {
  window._lastResults = results;
  window._lastProfile = { name, interests, ...scores };

  const top    = results[0];
  const meta   = CAREER_META[top.name] || { icon:'🎯', skills:'Various', degree:'Relevant degree' };
  const riasec = RIASEC_FULL[top.riasec] || { name: top.riasec, color:'#555', desc:'' };

  // Hero section
  document.getElementById('rec-icon').textContent    = meta.icon;
  document.getElementById('rec-title').textContent   = top.name;
  document.getElementById('rec-desc').textContent    = top.description || meta.desc || '';
  document.getElementById('rec-skills').textContent  = meta.skills;
  document.getElementById('rec-degrees').textContent = meta.degree;

  // Top match tag
  const tag = document.getElementById('rec-tag');
  if (tag) { tag.style.display = ''; tag.className = 'card-tag tag-green'; tag.textContent = `${top.score}% Match`; }

  // RIASEC badge
  document.getElementById('rec-riasec').innerHTML = `
    <div class="riasec-badge" style="background:${riasec.color}">
      ${top.riasec} · ${riasec.name}
    </div>
    <span class="riasec-desc">${riasec.desc}</span>`;

  // Confidence bar
  const confLevel = top.score >= 75 ? 'High' : top.score >= 55 ? 'Medium' : 'Low';
  const confColor = top.score >= 75 ? '#16a34a' : top.score >= 55 ? '#d97706' : '#dc2626';
  document.getElementById('rec-confidence').innerHTML = `
    <span class="conf-label">Confidence</span>
    <span class="conf-value" style="color:${confColor}">${confLevel} (${top.score}%)</span>
    <div class="conf-bar-wrap" style="margin-left:8px">
      <div class="conf-bar" style="width:0%;background:${confColor}" data-w="${top.score}%"></div>
    </div>`;

  // AI Reasoning explanation
  const topCareer = results[0];
  if (topCareer && topCareer.fc_reasons) {
    document.getElementById('explanation-text').innerText = topCareer.fc_reasons.join(", ");
  }
  
  if (topCareer && topCareer.conflict_note) {
    const badge = document.getElementById('conflict-badge');
    badge.innerText = `⚖️ Logic: ${topCareer.conflict_note}`;
    badge.style.display = 'block';
  }

  // Rule-based explanation
  const rules = buildRules(top.name, interests, scores);
  document.getElementById('rec-rules').innerHTML = `
    <div class="rules-title">📋 Why this career was recommended</div>
    ${rules.map(r => `<div class="rule-item"><span class="rule-icon">✓</span>${r}</div>`).join('')}`;

  // Match cards (top 3)
  const othersEl = document.getElementById('rec-others');
  othersEl.innerHTML = '';
  results.slice(0, 3).forEach((career, i) => {
    const m = CAREER_META[career.name] || { icon:'🎯' };
    const r = RIASEC_FULL[career.riasec] || { name: career.riasec, color:'#555' };
    const card = document.createElement('div');
    card.className = 'match-card';
    card.innerHTML = `
      <div class="match-rank">#${i+1} Match</div>
      <div class="match-name">${m.icon} ${career.name}</div>
      <div><span class="match-riasec-tag" style="background:${r.color}">${career.riasec} · ${r.name}</span></div>
      <div class="match-score">${career.score}% match</div>
      <div class="match-bar-wrap"><div class="match-bar" style="width:0%;background:${r.color}" data-w="${career.score}%"></div></div>`;
    othersEl.appendChild(card);
  });

  // Pipeline trace
  if (top.pipeline_trace) {
    renderPipelineTrace(top.pipeline_trace, top.dt_top3, top.fc_top3);
  }

  // Animate all bars
  setTimeout(() => {
    document.querySelectorAll('[data-w]').forEach(b => b.style.width = b.dataset.w);
  }, 120);

  // Saved note
  const panel = document.getElementById('rec-card');
  let note = panel.querySelector('.saved-note');
  if (!note) { note = document.createElement('p'); note.className = 'saved-note'; panel.appendChild(note); }
  note.textContent = `✓ ${name}'s profile saved`;

  document.getElementById('rec-result').style.display = 'block';
}


// ════════════════════════════════════════════════════════════
//  BUILD RULE EXPLANATIONS
// ════════════════════════════════════════════════════════════
function buildRules(careerName, interests, scores) {
  const rules = [];
  const sr = SUBJECT_RULES[careerName] || [];
  sr.forEach(([subj, threshold, reason]) => { if (scores[subj] >= threshold) rules.push(reason); });
  if (scores.study_hours >= 30) rules.push(`High study hours (${scores.study_hours}hrs/wk) reflect strong academic commitment`);
  else if (scores.study_hours >= 15) rules.push(`Consistent study hours (${scores.study_hours}hrs/wk) support career readiness`);
  if (window._lastResults?.[0]?.kg_boost)      rules.push('⚡ Knowledge Graph boost applied — interest & trait signals used as tie-breaker');
  if (window._lastResults?.[0]?.conflict_note) rules.push(window._lastResults[0].conflict_note);
  return rules.slice(0, 4);
}


// ════════════════════════════════════════════════════════════
//  PIPELINE TRACE RENDERER
// ════════════════════════════════════════════════════════════
function renderPipelineTrace(trace, dtTop3, fcTop3) {
  const panel = document.getElementById('trace-panel');
  const steps = document.getElementById('trace-steps');
  panel.style.display = 'block';
  steps.innerHTML = '';

  function miniList(items, color) {
    if (!items || !items.length) return '';
    const max = items[0].pct || 1;
    return `<div class="trace-mini">${items.map(r => `
      <div class="trace-mini-row">
        <span class="trace-mini-name">${r.name}</span>
        <div class="trace-mini-bar-wrap"><div class="trace-mini-bar" style="background:${color};width:0%" data-w="${(r.pct/max*100).toFixed(0)}%"></div></div>
        <span class="trace-mini-pct">${r.pct}%</span>
      </div>`).join('')}</div>`;
  }

  const dtColor = trace.dt_confidence >= 60 ? '#2563eb' : trace.dt_confidence >= 35 ? '#d97706' : '#dc2626';

  // Step 1 — Decision Tree
  steps.innerHTML += `
    <div class="trace-step step-dt">
      <div class="trace-icon">🌳</div>
      <div class="trace-body">
        <div class="trace-week">Step 1 · Week 2</div>
        <div class="trace-label">Decision Tree — <strong>${trace.dt_top}</strong></div>
        <div style="display:flex;align-items:center;margin-top:5px">
          <div class="trace-bar-wrap"><div class="trace-bar" style="background:${dtColor};width:0%" data-w="${trace.dt_confidence}%"></div></div>
          <span class="trace-pill ${trace.dt_confidence>=35?'pill-blue':'pill-red'}">${trace.dt_confidence}% confidence</span>
        </div>
        <div class="trace-sub">Trained on 6,000 student records · max_depth=8 · Gini impurity · ${trace.blend_weights}</div>
        ${dtTop3 ? miniList(dtTop3, dtColor) : ''}
      </div>
    </div>`;

  // Step 2 — KG Boost
  if (trace.kg_boost) {
    steps.innerHTML += `
      <div class="trace-step step-kg">
        <div class="trace-icon">⚡</div>
        <div class="trace-body">
          <div class="trace-week">Step 2 · Week 3</div>
          <div class="trace-label">Knowledge Graph Boost — <strong>Activated</strong></div>
          <span class="trace-pill pill-yellow">DT confidence &lt; 35% threshold</span>
          <div class="trace-sub">Interest &amp; trait signals used as tie-breaker. Artistic careers amplified if creative signals detected. Blend shifted to 50 / 50.</div>
        </div>
      </div>`;
  } else {
    steps.innerHTML += `
      <div class="trace-step step-kg">
        <div class="trace-icon">✅</div>
        <div class="trace-body">
          <div class="trace-week">Step 2 · Week 3</div>
          <div class="trace-label">Knowledge Graph Boost — <strong>Not needed</strong></div>
          <span class="trace-pill pill-green">DT confidence ≥ 35%</span>
          <div class="trace-sub">Standard blend applied: ${trace.blend_weights}.</div>
        </div>
      </div>`;
  }

  // Step 3 — Forward Chaining
  steps.innerHTML += `
    <div class="trace-step step-fc">
      <div class="trace-icon">⚙️</div>
      <div class="trace-body">
        <div class="trace-week">Step 3 · Week 1</div>
        <div class="trace-label">Forward Chaining Engine — <strong>${trace.fc_top}</strong></div>
        <span class="trace-pill pill-green">${trace.fc_confidence}% rule match</span>
        <div class="trace-sub">IF-THEN rules applied across interests, skills, and traits from Knowledge Base.</div>
        ${fcTop3 ? miniList(fcTop3, '#16a34a') : ''}
      </div>
    </div>`;

  // Step 4 — Conflict Resolution
  const winnerStyles = { agree:['pill-green','✅ Both engines agree'], rules:['pill-blue','⚖️ Rules Engine wins'], kg:['pill-yellow','⚡ KG Boost overrides'], dt:['pill-purple','🌳 Decision Tree wins'] };
  const [pillClass, winnerLabel] = winnerStyles[trace.conflict_winner] || ['pill-white', trace.conflict_winner];
  steps.innerHTML += `
    <div class="trace-step step-conf">
      <div class="trace-icon">⚖️</div>
      <div class="trace-body">
        <div class="trace-week">Step 4 · Conflict Resolution</div>
        <div class="trace-label">Final Decision</div>
        <span class="trace-pill ${pillClass}">${winnerLabel}</span>
        <div class="trace-sub">${trace.conflict_note}</div>
      </div>
    </div>`;

  // Animate mini bars
  setTimeout(() => {
    steps.querySelectorAll('[data-w]').forEach(b => b.style.width = b.dataset.w);
  }, 150);
}


// ════════════════════════════════════════════════════════════
//  CSV STUDENT SEARCH MODAL
// ════════════════════════════════════════════════════════════
function loadCSVStudent() {
  openModal('Test With CSV Student');
  document.getElementById('modal-body').innerHTML = `
    <p style="font-size:12px;color:#64748b;margin-bottom:12px">Search the 6,000-student dataset and load a student's scores automatically.</p>
    <div style="display:flex;gap:8px;margin-bottom:12px">
      <input id="csv-search-input" type="text" class="field-input" placeholder="Search by name or career…" style="flex:1"/>
      <button onclick="searchCSVStudents()" class="btn-primary" style="width:auto;padding:8px 16px">Search</button>
    </div>
    <div id="csv-results"><p style="color:#94a3b8;font-size:12px;text-align:center;padding:16px">Enter a name or career to search.</p></div>`;
  document.getElementById('csv-search-input').addEventListener('keydown', e => { if (e.key==='Enter') searchCSVStudents(); });
}

function searchCSVStudents() {
  const q  = document.getElementById('csv-search-input').value.trim();
  if (!q) return;
  const el = document.getElementById('csv-results');
  el.innerHTML = '<p style="color:#94a3b8;font-size:12px;text-align:center;padding:12px">Searching…</p>';
  fetch(`/api/students?search=${encodeURIComponent(q)}&per_page=6`)
    .then(r => r.json())
    .then(data => {
      if (!data.students?.length) { el.innerHTML = '<p style="color:#94a3b8;font-size:12px;text-align:center;padding:12px">No students found.</p>'; return; }
      el.innerHTML = data.students.map(s => `
        <div class="csv-row" onclick='loadStudentProfile(${JSON.stringify(s).replace(/'/g,"&#39;")})'>
          <div class="csv-row-name">${s.first_name} ${s.last_name}</div>
          <div class="csv-row-info">
            <span>${s.career_aspiration}</span>
            <span>Math: ${s.scores?.math??'—'}</span>
            <span>Bio: ${s.scores?.biology??'—'}</span>
            <span>Eng: ${s.scores?.english??'—'}</span>
          </div>
        </div>`).join('');
    });
}

const CAREER_INTERESTS_MAP = {
  "Software Engineer":["computers","technology","problem_solving","math"],
  "Doctor":["biology","helping_people","research"],
  "Lawyer":["law","communication","helping_people"],
  "Teacher":["education","helping_people","communication"],
  "Scientist":["research","biology","math"],
  "Business Owner":["communication","problem_solving","law"],
  "Accountant":["math","problem_solving"],
  "Banker":["math","communication"],
  "Designer":["art","technology","problem_solving"],
  "Artist":["art","education","communication"],
  "Writer":["communication","education","art"],
  "Game Developer":["computers","technology","art","problem_solving"],
  "Construction Engineer":["problem_solving","math","technology"],
  "Stock Investor":["math","problem_solving","research"],
  "Real Estate Developer":["law","math","communication"],
  "Government Officer":["law","communication","helping_people","education"],
  "Social Network Studies":["communication","technology","education"],
};

function loadStudentProfile(student) {
  closeModal();
  document.getElementById('name').value = `${student.first_name} ${student.last_name}`;
  const fieldMap = { math:'math-val', history:'hist-val', physics:'phys-val', chemistry:'chem-val', biology:'bio-val', english:'eng-val', geography:'geo-val' };
  Object.entries(fieldMap).forEach(([key, valId]) => {
    const el = document.getElementById(key);
    if (el && student.scores?.[key] !== undefined) {
      el.value = student.scores[key];
      document.getElementById(valId).textContent = student.scores[key];
    }
  });
  const sh = document.getElementById('study_hours');
  if (sh) { sh.value = student.study_hours||20; document.getElementById('study-val').textContent = student.study_hours||20; }
  document.querySelectorAll('.chip').forEach(c => c.classList.remove('selected'));
  getRecommendations();
}


// ════════════════════════════════════════════════════════════
//  VIEW DETAILS MODAL
// ════════════════════════════════════════════════════════════
function viewDetails() {
  openModal('Saved Profile');
  document.getElementById('modal-body').innerHTML = '<p style="color:#94a3b8;text-align:center;padding:20px;font-size:12px">Loading…</p>';
  fetch('/api/profiles')
    .then(r => r.json())
    .then(profiles => {
      const name    = document.getElementById('name').value.trim() || 'Student';
      const profile = profiles.find(p => p.name === name) || profiles[0];
      if (!profile) { document.getElementById('modal-body').innerHTML = '<p style="color:#94a3b8;text-align:center;padding:20px;font-size:12px">No saved profile found.</p>'; return; }
      const interests = Array.isArray(profile.interests) ? profile.interests : [];
      const badges = interests.length ? interests.map(i => `<span class="modal-badge">${i}</span>`).join('') : '—';
      const recCards = [...document.querySelectorAll('.match-card')];
      const recsHtml = recCards.map((c, i) => {
        const n = c.querySelector('.match-name')?.textContent || '';
        const s = c.querySelector('.match-score')?.textContent || '';
        return `<div class="modal-row"><span class="label">#${i+1}</span><span class="value">${n} — ${s}</span></div>`;
      }).join('');
      document.getElementById('modal-body').innerHTML = `
        <div class="modal-section-title">📋 Student Details</div>
        <div class="modal-row"><span class="label">Name</span><span class="value">${profile.name}</span></div>
        <div class="modal-row"><span class="label">Math</span><span class="value">${profile.math}</span></div>
        <div class="modal-row"><span class="label">Biology</span><span class="value">${profile.biology}</span></div>
        <div class="modal-row"><span class="label">English</span><span class="value">${profile.english}</span></div>
        <div class="modal-row"><span class="label">Study Hours</span><span class="value">${profile.study_hours}/week</span></div>
        <div class="modal-row"><span class="label">Interests</span><span class="value" style="text-align:right">${badges}</span></div>
        <div class="modal-section-title">🎯 Career Recommendations</div>
        ${recsHtml || '<p style="color:#94a3b8;font-size:12px">No recommendations yet.</p>'}`;
    })
    .catch(() => {
      document.getElementById('modal-body').innerHTML = '<p style="color:#dc2626;text-align:center;padding:20px;font-size:12px">⚠️ Could not connect to storage.</p>';
    });
}


// ════════════════════════════════════════════════════════════
//  MODAL HELPERS
// ════════════════════════════════════════════════════════════
function openModal(title) {
  document.getElementById('modal-title').textContent = title;
  document.getElementById('modal-overlay').classList.add('active');
}
function closeModal() {
  document.getElementById('modal-overlay').classList.remove('active');
}
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
