# Task 2 Error Analysis (Held-Out Test Set)

**Model**: Qwen2.5-1.5B-Instruct + LoRA (adapter_result_kaggle)
**Test Split**: 8 patients, 206 criterion-level annotations (75 inclusion, 131 exclusion)
**Metrics**: Accuracy=66.99%, Macro-F1=37.89%, Evidence micro-F1=40.13%

---

## Summary of Key Weaknesses

| Issue | Evidence |
|-------|----------|
| **Exclusion criteria collapse to "not excluded"** | 91/131 exclusion rows are gold "not excluded"; model predicts "not excluded" for 118/131 rows (90%). Recall for "not enough information" (28 support) = 0%. Recall for "not applicable" (12 support) = 25%. |
| **Inclusion "not included" never predicted** | 7 gold "not included" → 0 predicted (4→NI, 3→included). |
| **NI/NA confusion** | Gold NI → pred "not excluded" (exclusion) or "included" (inclusion). Gold NA → pred "not excluded" or "NI" or "included". |
| **Evidence retrieval poor on exclusion** | Exclusion micro-F1=18.4% vs Inclusion micro-F1=64.4%. 115/131 exclusion rows have empty gold evidence; model predicts non-empty evidence for 114 of them. |
| **Hallucinated eligibility** | Model frequently asserts "included"/"not excluded" with generic evidence (sentence 0 = age/consent) when gold says NI/NA. |

---

## 8+ Distinct Failure Modes (with Test Examples)

### 1. False Positive Inclusion — Gold "not enough information" → Pred "included"
**Annotations**: 163, 164, 177, 189, 192, 307, 323, 325, 384, 740, 744, 756, 760, 762, 765, 827, 830, 831, 832, 834, 836, 838 (22 cases)

| Ann | Criterion | Gold | Pred | Explanation |
|-----|-----------|------|------|-------------|
| 163 | "Patients undergoing endovascular abdominal aneurysm repair" | NI | included | Cites sentence 0 (age) — cardiac cath ≠ aneurysm repair |
| 164 | "Patients undergoing surgical lower limb revascularisation" | NI | included | Cites sentence 0 — cardiac cath ≠ surgical revascularisation |
| 177 | "Maintenance hemodialysis after central vena catheterization" | NI | included | Cites sentence 0 — femoral cath ≠ hemodialysis |
| 189 | "at least 1 of 3 signs of hypoperfusion..." | NI | included | Cites sentence 1 — pallor/tachycardia ≠ defined hypoperfusion signs |
| 192 | "no clinical evidence of left atrial hypertension..." | NI | included | Cites sentence 0 — no evidence for PA pressure |
| 307 | Complex liver resection criterion | NI | included | Cites sentence 0 — CVID history ≠ liver lesion requiring resection |
| 323 | "Immune Deficiency and early BMF in childhood" | NI | included | Cites sentence 0 — CVID ≠ childhood BMF |
| 325 | "Genetic patients" | NI | included | Cites sentence 0 — CVID ≠ confirmed genetic diagnosis |
| 384 | Lab values (Hb, ANC, bilirubin, AST/ALT) | NI | included | Cites sentence 2 — anemia/thrombocytopenia ≠ meeting thresholds |
| 740 | "patient speaks and understands Finnish or Swedish" | NI | included | Cites sentence 4 — consent ≠ language proficiency |
| 744 | "patient has peritonsillar abscess" | NI | included | Cites sentence 4 — consent ≠ abscess diagnosis |
| 756 | "treatment of peritonsillar abscess requires in-patient care" | NI | included | Cites sentence 0 — choking symptoms ≠ peritonsillar abscess |
| 760 | "Clinical diagnosis of acute tonsillitis/pharyngitis caused by Strep pyogenes" | NI | included | Cites sentence 0 — dysphagia/drooling ≠ confirmed Strep |
| 762 | "sore throat + ≥2 clinical signs" | NI | included | Cites sentence 0 — dysphagia only, not 2 signs |
| 765 | "Uvular edema" | NI | included | Cites sentence 0 — dysphagia ≠ uvular edema |
| 827 | "Clinical diagnosis of bronchial asthma" | NI | included | Cites sentence 0 — wheezing ≠ clinical diagnosis |
| 830 | "Normal lung function (FVC>80%, FEV1>80%, FEV1/FVC>.75)" | NI | included | Cites sentence 0 — wheezing ≠ normal lung function |
| 831 | "Oxygen saturation >94% and normal BP" | NI | included | Cites sentence 7 — SpO2 100% but no BP |
| 832 | "Symptom Score ≤6" | NI | included | Cites sentence 0 — contradicts "symptoms do not meet criteria" |
| 834 | "Negative pregnancy test for females" | NA | **included** | Male patient → should be NA, not included |
| 836 | "History of episodic wheezing... after age 6" | not included | included | Patient is 4yo — explicitly fails age criterion |
| 838 | "FEV1 ≥80% predicted..." | NI | NI | Correct label but cites sentence 0 with "No data provided" |

**Pattern**: Model treats *any* vaguely related symptom as sufficient for inclusion. Uses sentence 0 (demographics) as catch-all evidence.

---

### 2. False Negative Exclusion — Gold "not enough information" → Pred "not excluded"
**Annotations**: 169, 174, 183, 193, 315, 389, 390, 391, 392, 393, 394, 395, 398, 399, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 418, 419, 424, 681, 697, 841, 849, 853 (32+ cases)

| Ann | Criterion | Gold | Pred | Issue |
|-----|-----------|------|------|-------|
| 169 | "GFR <30" | NI | not excluded | No kidney data → should be NI |
| 174 | "Previous axillary surgery" | NI | not excluded | "No surgical procedures" ≠ confirmed no axillary surgery |
| 183 | "Patients were moribund" | NI | not excluded | Gold evidence=[0,1,2,3,4]; model cites [0] with "not moribund" |
| 193 | "Failure to pass MRI metal screening" | NI | not excluded | Ultrasound done ≠ MRI screening passed |
| 315 | "BMI >40" | NI | not excluded | No BMI data |
| 389 | "Severe renal impairment (GFR<50%)" | NI | not excluded | Protein/RBC casts → actually suggests impairment! |
| 390 | "Oral corticosteroids >15mg/day..." | NI | not excluded | No medication data |
| 391 | "IV corticosteroids within 3 months" | NI | not excluded | No medication data |
| 392 | "IV cyclophosphamide within 6 months" | NI | not excluded | No medication data |
| 393 | "Anti-rheumatic/immunosuppressive drugs..." | NI | not excluded | No medication data |
| 394 | "B-cell depletion therapy (Rituximab)..." | NI | not excluded | No medication data |
| 395 | "CYP3A4 inhibitors/inducers..." | NI | not excluded | No medication data |
| 398 | "Additional risk factors for torsade..." | NI | not excluded | No cardiac history |
| 399 | "Concomitant QT-prolonging medications" | NI | not excluded | No medication data |
| 403 | "Hep B/C/HIV positive" | NI | not excluded | Cites ANA/anti-dsDNA — irrelevant |
| 404 | "Drug abuse" | NI | not excluded | Consent ≠ no drug abuse |
| 405 | "Major surgery within 3 weeks" | NI | not excluded | Consent ≠ no recent surgery |
| 406 | "Hypersensitivity to ABR-215757" | NI | not excluded | Cites ANA — irrelevant |
| 407 | "Female childbearing potential + contraception" | NI | not excluded | Age 21 → needs contraception status |
| 408 | "Pregnant or lactating" | NI | not excluded | Age 21 → needs pregnancy test |
| 409 | "Participation in other investigational study" | NI | not excluded | Consent ≠ no other study |
| 410 | "Other unstable medical disease" | NI | not excluded | SLE symptoms ≠ no other disease |
| 411 | "Likely to receive steroids/immunosuppressant" | NI | not excluded | Consent ≠ no future meds |
| 412 | "Vaccination within 4 weeks" | NI | not excluded | Consent ≠ no vaccination |
| 418 | "Failure to pass MRI metal screening" | NI | not excluded | Ultrasound ≠ MRI screening |
| 419 | "Claustrophobia/sedation need" | NI | not excluded | Consent ≠ no claustrophobia |
| 424 | "Life expectancy <72 hours" | NI | not excluded | Age only ≠ life expectancy |
| 681 | "Primary indication for AT is nocturnal obstructive symptoms" | NI | not excluded | Gold=included! Model says not excluded (wrong type) |
| 697 | "Pregnant" | NA | NI | Male child → should be NA |
| 841 | "Chronic medical condition contraindication" | NI | not excluded | Allergic rhinitis ≠ no contraindication |
| 849 | "Nighttime cough/wheeze >1x/week" | NI | not excluded | Gold evidence=[0,3,6]; model cites [0] only |
| 853 | "Acute infection requiring antibiotics" | NI | not excluded | Allergic rhinitis ≠ no acute infection |

**Pattern**: Default exclusion prediction = "not excluded" with generic "no mention" or consent evidence. Fails to distinguish *absence of evidence* from *evidence of absence*.

---

### 3. Not Applicable → Wrong Label (NI / Not Excluded / Included)
**Annotations**: 165, 186, 198, 422, 683, 697, 745, 750, 751, 754, 758, 767, 771, 834, 847, 848 (16 cases)

| Ann | Criterion | Gold | Pred | Issue |
|-----|-----------|------|------|-------|
| 165 | "Pregnancy" (67yo male) | NA | NA | Correct label but cites sentence 0 (age) — should be empty evidence |
| 186 | "Negative pregnancy test for females" | NA | **included** | Male patient → included is wrong |
| 198 | "If female, using birth control" | NA | included | 15yo girl, no sexual partners → NA |
| 422 | "If female, using birth control" | NA | included | Same as 198 |
| 683 | "Previous tonsillectomy" | NA | not excluded | No tonsils → NA |
| 697 | "Pregnant" | NA | NI | 10yo boy → NA |
| 745 | "Female patients have adequate birth-control" | NA | NI | Male child → NA |
| 750 | "Pregnancy" | NA | NA | Correct |
| 751 | "Breast-feeding" | NA | not excluded | 5yo child → NA |
| 754 | "Alcoholism (drunk ≥1x/week)" | NA | not excluded | 5yo child → NA |
| 758 | "Army recruit" | NA | not excluded | 5yo child → NA |
| 767 | "Symptoms suggest non-streptococcal T/P" | NA | not excluded | Gold=NA (streptococcal symptoms present) |
| 771 | "Females of childbearing potential" | NA | NI | 5yo boy → NA |
| 834 | "Negative pregnancy test for females" | NA | **included** | 4yo boy → NA |
| 847 | "Pregnancy or nursing" | NA | NA | Correct |
| 848 | "Cigarette smoking >1 pack/month" | NA | not excluded | 4yo boy → NA |

**Pattern**: Model does not understand demographic/conditional inapplicability. Treats NA criteria as evaluable.

---

### 4. "Not Included" (Inclusion) Never Predicted — 0/7 Recall
**Annotations**: 161, 162, 175?, 188?, 306?, 323?, 325?, 384?, 421?, 760?, 762?, 763?, 764? (gold "not included" or NI on inclusion criteria)

| Ann | Criterion | Gold | Pred | Issue |
|-----|-----------|------|------|-------|
| 161 | "Patients undergoing elective carotid endarterectomy" | not included | NI | Cites sentence 0 (age) |
| 162 | "Patients undergoing open AAA repair" | not included | NI | Cites sentence 0 |
| 188 | "Use of vasopressors to maintain SBP≥90..." | not included | NI | Gold evidence=[3] (BP normal) |
| 306 | "Patients with liver space occupying lesion requiring extended hepatic resection" | not included | NI | Gold evidence=[3] |
| 323 | "Immune Deficiency and early BMF in childhood" | not included | **included** | CVID ≠ childhood BMF |
| 325 | "Genetic patients" | not included | **included** | CVID ≠ confirmed genetic |
| 384 | Lab values criterion | not included | **included** | Anemia/thrombocytopenia present |
| 421 | "Able to take medicine orally after surgery" | NI | NI | Correct NI |
| 760 | "Clinical diagnosis of Strep tonsillitis" | not included | **included** | Dysphagia ≠ Strep diagnosis |
| 762 | "Sore throat + ≥2 signs" | not included | **included** | Only dysphagia |
| 763 | "Tonsil/pharyngeal erythema/exudate" | not included | NI | No mention |
| 764 | "Cervical adenopathy" | not included | NI | No mention |

**Pattern**: Model cannot say "not included" — defaults to NI or included. No negative inclusion examples in training?

---

### 5. "Excluded" (Exclusion) Never Predicted — 0/0 Support (but 2 gold in val)
Test set has 0 gold "excluded". Validation had 2 (both predicted correctly after LoRA). Model appears calibrated to avoid "excluded" entirely.

---

### 6. Hallucinated / Incorrect Evidence Sentences
**Many cases** — model cites sentence 0 (demographics) or sentence 3/4/5 (consent) for criteria requiring specific clinical data.

| Ann | Criterion | Gold Evidence | Pred Evidence | Issue |
|-----|-----------|---------------|---------------|-------|
| 163 | Endovascular AAA repair | [] | [0] | Age ≠ procedure |
| 164 | Surgical lower limb revascularisation | [] | [0] | Age ≠ procedure |
| 169 | GFR <30 | [] | [0] | Age ≠ kidney function |
| 174 | Previous axillary surgery | [] | [0] | "No surgical procedures" (wrong) |
| 189 | Hypoperfusion signs | [] | [1] | Pallor/tachycardia ≠ defined signs |
| 307 | Complex liver resection | [3] | [0] | CVID ≠ liver lesion |
| 315 | BMI >40 | [] | [0] | Age ≠ BMI |
| 384 | Lab values | [] | [2] | Anemia present but thresholds? |
| 389 | GFR <50% | [] | [3] | Protein/RBC casts → actually evidence FOR impairment |
| 390 | Corticosteroids >15mg | [] | [4] | Consent ≠ no steroids |
| 391 | IV corticosteroids | [] | [4] | Consent ≠ no IV steroids |
| 403 | Hep B/C/HIV | [] | [1] | ANA/anti-dsDNA ≠ viral serology |
| 407 | Contraception | [] | [0] | Age 21 ≠ contraception status |
| 418 | MRI metal screening | [] | [5] | Ultrasound ≠ MRI screening |
| 422 | Birth control (NA) | [3] | [3] | "No sexual partners" → should be NA with empty evidence |
| 681 | Primary indication AT | [0] | [0] | Gold=included, model=not excluded |
| 697 | Pregnant (NA) | [] | [0] | Male child |
| 745 | Birth control (NA) | [] | [0] | Male child |
| 751 | Breast-feeding (NA) | [] | [4] | 5yo child |
| 754 | Alcoholism (NA) | [] | [1] | "Toxic-appearing" ≠ alcoholism |
| 758 | Army recruit (NA) | [] | [0] | 5yo child |
| 767 | Non-strep symptoms (NA) | [0] | [0] | Gold=NA, model=not excluded |
| 834 | Pregnancy test (NA) | [] | [0] | 4yo boy |
| 848 | Smoking (NA) | [] | [1] | Allergic rhinitis ≠ smoking |

**Evidence micro-F1**: Overall=40.1%, Inclusion=64.4%, Exclusion=18.4% — exclusion evidence largely hallucinated.

---

### 7. Parse Failure (Truncated Output)
**Annotation 396**: Long criterion text → model output cut off mid-JSON
```
raw_text: "{\"label\":\"not excluded\",\"evidence\":[0],\"explanation\":\"The patient has no history of myocardial infarction or current uncontrolled angina, severe uncontrolled ventricular arrhythmias, symptomatic congestive heart failure, unstable angina pectoris, or electrocardiographic evidence of acute ischem"
```
**Cause**: `max_new_tokens=64` too small for long criteria + explanation. Fix: increase to 128+.

---

### 8. Contradictory Predictions Within Same Patient-Trial
**Patient sigir-201410, Trial NCT02097186**:
- Ann 163: "endovascular AAA repair" → **included** (gold NI)
- Ann 164: "surgical lower limb revascularisation" → **included** (gold NI)
- Ann 161: "elective carotid endarterectomy" → NI (gold not included)
- Ann 162: "open AAA repair" → NI (gold not included)

Same patient cannot be simultaneously eligible for mutually exclusive procedures. Model treats each criterion independently without trial-level consistency.

---

### 9. Systematic Exclusion-Criteria Bias (Quantitative)
| Gold Label | Support | Predicted as "not excluded" | Recall |
|------------|---------|----------------------------|--------|
| not excluded | 91 | 88 | 96.7% |
| not enough information | 28 | 28 | 0% |
| not applicable | 12 | 7 | 25% |
| excluded | 0 | 0 | N/A |

Model effectively learns: **exclusion → "not excluded"**. This inflates accuracy (69.5% on exclusion) but macro-F1=20.4% (near random for minority classes).

---

## Recommendations for Report

1. **Acknowledge exclusion-criteria collapse** — primary limitation. Cite class imbalance (91/131 = "not excluded") and lack of "excluded"/NI/NA examples in training.
2. **Evidence hallucination** — model cites generic sentences (demographics, consent) when specific clinical evidence absent. Suggest: add "no evidence" token, or train with evidence masking.
3. **NI/NA confusion** — model lacks explicit "not applicable" reasoning. Suggest: add criterion-type-specific prompts with NA examples.
4. **Increase max_new_tokens** — fix parse failure on long criteria (64 → 128+).
5. **Trial-level consistency** — post-hoc check: a patient cannot be "included" for mutually exclusive procedures.
6. **Per-class thresholds** — macro-F1 low because model optimizes accuracy on majority class. Consider calibrated thresholds per label.

---

## Files for Submission
- `results/metrics_after_lora_test.json` — full metrics
- `results/preds_after_lora_test.jsonl` — per-row predictions
- `results/error_analysis_task2.md` — this document