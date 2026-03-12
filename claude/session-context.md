# Contexte du projet — Energy Load Prediction pour OIKEN

## Vue d'ensemble

Projet d'Energy Informatics II (HES-SO) visant à construire des modèles de prédiction des **charges nettes** (consommation - production PV) sur la zone de desserte d'OIKEN.

Étudiant : Alexandre Jewell (alexandre.jewell@me.com)

## Page Notion du projet

https://www.notion.so/alexandre-jewell/Energy-load-and-production-forecast-for-OIKEN-312c3942dace804982cbef88a35fbbac

Table de to-dos : section `#31ec3942dace801ab6dff0e3e180d331`

## Trois modèles

| Modèle | Objectif | Approche | Début | Fin |
|---|---|---|---|---|
| Day-ahead | Charges nettes quart-horaires pour J+1 | Directe (multi-output) | D-1 à 00h | D-1 à 11h |
| Intraday | Charges nettes quart-horaires H → fin J | Récursive (pas-à-pas) | D-1 à 11h | D à H-45min |
| Réglage (optionnel) | Charges nettes H → H+45min | Récursive | D à H-45min | D à H |

## Sources de données

### OIKEN (CSV)
- `standardised load [-]` : consommation normalisée z-score, 15 min, disponible J+1 01h
- `standardised forecast load [-]` : prévision OIKEN normalisée z-score, 15 min
- Production PV sites (Sion, Sierre, Central Valais) en kWh, 15 min, temps réel (à confirmer)
- Production PV smart meters (remote) en kWh, 15 min, J+1 01h
- Fichier : `data/oiken-data(in).csv` (105 120 lignes, oct 2022 – sept 2025)

### Météo Suisse (InfluxDB)
- Mesures historiques : 10 min, disponibles H+1
- Prévisions : 1h, actualisées toutes les 3h depuis H-33
- Variables : T_2M, GLOB, DURSUN, TOT_PREC (moyenne, D10, D90, std pour prévisions)
- On utilise dans un premier temps uniquement la prévision la plus récente (pas les déciles/std)

## Architecture globale — 4 couches

1. **Acquisition** : InfluxDB API + CSV OIKEN → DataFrames à schéma unifié
2. **Feature engineering** : features calendaires, météo normalisées, capacité PV installée
3. **Modèle** : interface abstraite commune (fit/predict/save/load)
4. **Évaluation** : walk-forward causal, MAE/RMSE

## Feature engineering

### Capacité PV installée
- ratio(t) = P_pv(t) / (G(t)/1000 × 0.25), filtré GLOB > 100 W/m²
- Rolling median 30 jours, causale (center=False dans Polars)
- Résultat : courbe lentement croissante de la capacité effective du parc PV

### Normalisation z-score
- Scaler fitté sur portion entraînement de chaque fenêtre walk-forward
- Scaler final fitté sur intégralité du training set
- Même scaler final pour test set et production
- **Important** : la charge OIKEN est déjà pré-normalisée par z-score → on n'a PAS les paramètres μ, σ originaux

### Cohérence entraînement / inférence (training-serving skew)
- Day-ahead : entraîner sur **prévisions historiques** (archivées dans InfluxDB), PAS sur mesures réelles
- Intraday : hybride (mesures réelles jusqu'à H + prévisions pour H → fin J)
- Réglage : quasi-exclusivement mesures récentes (nowcasting, lag features dominants)

## Modèles candidats

### Baselines
- **Principale** : naïf saisonnier J-7 même heure (persistance pour réglage)
- **Secondaire** : prévision OIKEN (standardised forecast load) — référence opérationnelle

### Candidats
| Modèle | Horizons | Notes |
|---|---|---|
| Régression linéaire Ridge | Tous | Baseline sophistiquée, interprétable |
| SARIMA(X) | Intraday, réglage | Récursif par nature, inadapté au direct multi-output |
| LightGBM | Tous | Modèle principal attendu, gradient boosting (≠ random forest) |

### Non retenus
- **Prophet** : support limité des variables exogènes
- **LSTM** : perspective future, complexité hors cadre du projet

## Optimisation des hyperparamètres

| Modèle | Méthode | Hyperparamètres clés |
|---|---|---|
| Ridge | Grid search (grille log) | λ : {0.001, 0.01, 0.1, 1, 10, 100} |
| SARIMA(X) | AIC/BIC + test ADF pour d, D | p, q ∈ {0,1,2}, P, Q ∈ {0,1}, s=96 |
| LightGBM | Random search 30-50 iter | num_leaves, learning_rate, n_estimators, min_child_samples, subsample, colsample_bytree, reg_lambda |

Option : optimisation bayésienne via Optuna pour LightGBM si le temps le permet.

## Évaluation

### Métriques
- **MAE** (principale) : robuste, interprétable
- **RMSE** (secondaire) : sensible aux pics d'erreur
- **MAPE** : écartée (instable quand charges nettes → 0)
- Diagnostic biais-variance : comparaison erreurs train vs validation par fenêtre walk-forward
- Métriques calculées en **espace normalisé [-]** (charge pré-normalisée par OIKEN)
- Demander μ, σ aux professeurs si interprétation en kW requise

### Protocole
- **Training set** : octobre 2022 – septembre 2024 (2 ans)
- **Test set** : octobre 2024 – septembre 2025 (1 an, représentatif de toutes les saisons)
- Walk-forward opère exclusivement dans le training set
- Test set évalué une seule fois après toutes décisions figées
- Analyse par granularité : tranche horaire, type de jour, saison
- Analyse par horizon pour modèles récursifs (accumulation d'erreurs)

## Planning des tâches

| Phase | Période | Tâches |
|---|---|---|
| 1 — Acquisition et préparation | 11.03 – 25.03 | Connexion InfluxDB, import CSVs, feature engineering, normalisation, validation pipeline |
| 2 — Développement et entraînement | 26.03 – 16.04 | Baselines, modèles candidats, walk-forward, hyperparamètres, entraînement final, évaluation test set |
| 3 — Pipeline E2E et visualisation | 22.04 – 29.04 | Intégration pipeline complète, visualisations, tests E2E |
| 4 — Démonstration | 30.04.2025 | Démo finale à OIKEN |

### Deadlines d'évaluation
- 05.03 : Workplan conceptuel (2 pages) + présentation orale (25%)
- 25.03 : Pipeline acquisition & normalisation + live code review (25%)
- 16.04 : Modèle entraîné et évalué + live code review (25%)
- 29.04 : Pipeline E2E avec visualisation + live code review (25%)
- 30.04 : Démo à OIKEN (BONUS)

## Fichiers générés

- `others/Workplan.docx` : document de travail principal (mis à jour manuellement par l'étudiant)
- `others/Architecture_timeline.pptx` : timeline des modèles day-ahead et intraday
- `others/Architecture_globale.png` : diagramme 4 couches de l'architecture

## Décisions techniques importantes

1. La charge OIKEN est **pré-normalisée** → pas d'accès aux valeurs brutes en kW
2. LightGBM est du **gradient boosting**, pas du random forest (bagging)
3. Walk-forward (pas k-fold) pour respecter la causalité temporelle
4. Entraîner sur **prévisions historiques** (pas mesures réelles) pour éviter le training-serving skew
5. PV sites disponible "temps réel" vs smart meters J+1 01h → contraintes différentes sur lag features
6. Médiane glissante causale (center=False) pour capacité PV → pas de fuite de données futures

## État du workplan (dernière lecture : 09.03.2026)

Sections complètes : Introduction, Architecture globale, Acquisition des données, Feature engineering (3 sous-sections), Entraînement des modèles (4 sous-sections), Évaluation (2 sous-sections), Planning des tâches.

**Prochaine étape** : Rédaction de la version synthétique 2 pages + planning pour le rendu du 05.03.

## Outils et stack technique

- Polars (DataFrames)
- LightGBM
- uv (package manager)
- pptxgenjs (génération PPTX)
- matplotlib (diagrammes)
- marimo (notebooks, dossier `notebooks/`)
