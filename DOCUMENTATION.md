# SignalGuard — Complete Documentation

**Fake profile identification across online social networks**

Repository: https://github.com/Saisriya2003/fake-profile-detector
Author: Pettem Sai Sriya · saisriyavarma@gmail.com

Resume project: *Identification of Fake Profiles Across Online Social Networks — Built a deep learning model using artificial neural networks to assess the authenticity of friend requests. Trained on data from Facebook and other platforms, applying sigmoid activation and backpropagation to refine weights and biases.*

---

## Contents

1. [Overview](#1-overview)
2. [Tech stack](#2-tech-stack)
3. [Repository layout](#3-repository-layout)
4. [System architecture](#4-system-architecture)
5. [Data: features and synthetic corpus](#5-data-features-and-synthetic-corpus)
6. [The neural network](#6-the-neural-network)
7. [Training pipeline and results](#7-training-pipeline-and-results)
8. [Inference and explainability](#8-inference-and-explainability)
9. [REST API reference](#9-rest-api-reference)
10. [Frontend: screens and user workflow](#10-frontend-screens-and-user-workflow)
11. [Running the application](#11-running-the-application)
12. [Configuration](#12-configuration)
13. [Testing, CI, and verification](#13-testing-ci-and-verification)
14. [Retraining and extending](#14-retraining-and-extending)
15. [Limitations and production path](#15-limitations-and-production-path)
16. [Glossary](#16-glossary)

---

## 1. Overview

SignalGuard is a trust-and-safety tool. Given the public signals of a social-network profile that sent a friend request (followers, following, account age, photo presence, bio length, username shape, mutual connections, posting cadence), it returns an **authenticity score**, a **verdict**, and an **explanation** of which signals pushed the decision.

The model is a **from-scratch artificial neural network written in NumPy**. There is no scikit-learn estimator, no PyTorch, no autograd. Sigmoid activations, binary cross-entropy loss, and mini-batch backpropagation are implemented explicitly in `backend/app/ann.py`, so every claim in the resume line is visible in code.

Key properties:

- Runs fully offline; no API keys, no external database.
- Trained weights, scaler, and metrics are committed, so a fresh clone scores profiles immediately.
- Multi-platform corpus (Facebook, Instagram, Twitter distributions) generated synthetically — no scraped identities.
- Three-screen React UI: Inspect (interactive scoring), Gallery (seeded profiles), Model lab (training diagnostics).

## 2. Tech stack

| Layer | Technology | Version (pinned) |
| --- | --- | --- |
| Language (backend) | Python | 3.11+ (CI uses 3.12) |
| Numerical core | NumPy | 2.5.3 |
| Data handling | pandas | 3.0.5 |
| Evaluation metrics only | scikit-learn (`sklearn.metrics`) | 1.9.1 |
| Web framework | FastAPI | 0.141.1 |
| ASGI server | Uvicorn (`[standard]`) | 0.52.4 |
| Multipart support | python-multipart | 0.0.32 |
| Language (frontend) | JavaScript (ES modules, JSX) | — |
| UI library | React | 18 |
| Build tool / dev server | Vite | 5 |
| Styling | Hand-written CSS with design tokens; Google Fonts Outfit + Fraunces | — |
| Charts | Custom SVG (`Gauge.jsx`, `Sparkline.jsx`) | — |
| Runtime (frontend) | Node.js | 18+ |
| CI | GitHub Actions (Ubuntu) | — |

## 3. Repository layout

```
fake-profile-detector/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── ann.py             NeuralNetwork: forward, backward, train, save/load, contributions
│   │   ├── features.py        FEATURE_NAMES, labels, UI bounds, completeness, min-max scaler
│   │   ├── generate_data.py   synthetic multi-platform profile generator → data/profiles.csv
│   │   ├── train.py           stratified split, training loop, metrics → models/*
│   │   └── main.py            FastAPI app and endpoints
│   ├── data/profiles.csv      4,000 synthetic profiles (committed)
│   ├── models/
│   │   ├── ann_weights.npz    trained weights and biases (committed)
│   │   ├── scaler.json        per-feature min/max from the training split
│   │   └── metrics.json       architecture, hyper-parameters, metrics, loss history
│   └── requirements.txt
├── frontend/
│   ├── index.html             inline SVG favicon, font links
│   ├── package.json
│   ├── vite.config.js         port 5173, strictPort, /api → http://127.0.0.1:8001
│   └── src/
│       ├── main.jsx, App.jsx  shell, tab navigation, health polling
│       ├── api.js             fetch helpers (VITE_API_URL or same-origin /api)
│       ├── features.js        feature metadata mirrored for the UI (labels, ranges, steps)
│       ├── index.css          design tokens and components
│       ├── components/Gauge.jsx, Sparkline.jsx
│       └── views/Inspect.jsx, Gallery.jsx, ModelLab.jsx
├── .github/workflows/ci.yml
├── start.ps1 / start.sh       one-command local start
├── .gitattributes             LF for .sh, CRLF for .ps1, binaries marked
├── .gitignore
└── README.md
```

## 4. System architecture

```
┌──────────────────────────────┐          ┌─────────────────────────────────────────┐
│ React UI  (Vite, :5173)      │          │ FastAPI  (Uvicorn, :8001)               │
│                              │  /api/*  │                                         │
│  Inspect   sliders/toggles ──┼─────────▶│ POST /api/predict                       │
│  Gallery   sample cards      │  proxy   │   PredictBody → derive username stats,  │
│  Model lab metrics/loss      │          │   completeness → vector_from_mapping    │
│                              │◀─────────┤   → normalize(scaler) → ANN forward     │
│  Gauge, Sparkline (SVG)      │   JSON   │   → P(fake) → score/verdict/confidence  │
└──────────────────────────────┘          │   → feature_contributions → explanation │
                                          │ GET /api/metrics  ← models/metrics.json │
                                          │ GET /api/samples  ← data/profiles.csv   │
                                          │ GET /api/platforms                      │
                                          │ GET /api/health                         │
                                          └─────────────────────────────────────────┘
                     offline, once:  generate_data.py ─▶ profiles.csv ─▶ train.py ─▶ models/*
```

Two processes. The UI never talks to the API cross-origin in development: Vite proxies `/api` to `127.0.0.1:8001`. CORS is additionally allowed for `localhost:5173/5174` in case the UI is served elsewhere.

## 5. Data: features and synthetic corpus

### 5.1 The 14 input features (`features.py`)

| Feature | Label in UI | UI range | Meaning |
| --- | --- | --- | --- |
| `follower_count` | Followers | 0–80,000 | People following the account |
| `following_count` | Following | 0–12,000 | Accounts it follows |
| `friends_count` | Friends | 0–6,000 | Mutual-consent connections (Facebook-style) |
| `posts_count` | Posts | 0–8,000 | Lifetime posts |
| `account_age_days` | Account age (days) | 1–5,200 | Days since creation |
| `has_profile_photo` | Profile photo | 0/1 | Has an avatar |
| `has_cover_photo` | Cover photo | 0/1 | Has a banner |
| `bio_length` | Bio length | 0–320 | Characters in the bio |
| `username_digit_ratio` | Username digit ratio | 0–1 | Fraction of digits in the handle |
| `username_length` | Username length | 1–32 | Characters in the handle |
| `friend_request_mutuals` | Mutual connections | 0–250 | Mutual friends with the recipient |
| `profile_completeness` | Profile completeness | 0–1 | Aggregate derived score (below) |
| `avg_likes_per_post` | Avg. likes per post | 0–8,000 | Engagement per post |
| `posts_per_week` | Posts per week | 0–40 | Cadence = posts / (age in weeks) |

**Profile completeness** is the mean of six sub-scores: has photo, has cover, `min(bio/80, 1)`, `min(posts/40, 1)`, `1.0` if username length is 4–18 else `0.35`, `min(age/365, 1)`. It is computed identically at data-generation time and at inference time so there is no train/serve skew.

### 5.2 Scaling

Min-max scaling to \[0, 1\] using per-feature min/max **fitted on the training split only** and persisted to `scaler.json`. At inference, `normalize()` clips to \[0, 1\] so out-of-range inputs cannot explode the sigmoid.

### 5.3 Synthetic corpus (`generate_data.py`)

- 4,000 profiles, seed 2025; platforms drawn as Facebook 42% / Instagram 36% / Twitter 22%; 44% labelled `fake`.
- Each platform has its own log-normal distributions for followers, following, friends, posts, likes, mutuals, bio length, and photo probabilities so the model learns platform-relative cues rather than one global threshold.
- **Overlapping sub-populations** make the problem non-trivial: *sockpuppets* (older fakes with photos and a few posts), *new genuine* accounts (thin but real), *quiet real* (low activity), *groomed fakes* (boosted followers, longer bios). Fakes under 21 days may "burst" post.
- Usernames: real accounts are `first[.|_]last[yy]`; most fakes are `stem[_.]stem?digits` from stems like `user`, `official`, `promo`, `bot`, giving a high digit ratio; 18% of fakes mimic real names with a trailing digit.
- Multiplicative noise `N(1, 0.14)` on all count features.

No real people, handles, or scraped data are involved.

## 6. The neural network

File: `backend/app/ann.py`, class `NeuralNetwork`.

### 6.1 Architecture

`[14, 16, 8, 1]` — 14 inputs → 16 hidden → 8 hidden → 1 output. Every layer uses the **sigmoid** activation. Output is P(fake).

### 6.2 Initialisation

Xavier/LeCun-normal: `W ~ N(0, 1/fan_in)`, biases zero, seeded RNG (2025) for reproducibility.

### 6.3 Forward pass

For each layer `l`: `z_l = a_{l-1} W_l + b_l`, `a_l = σ(z_l)` with `σ(z) = 1 / (1 + e^{-z})`, `z` clipped to ±60. Activations are cached for backprop.

### 6.4 Loss

Binary cross-entropy, mean over the batch, with ε = 1e-9 clipping:

\[
L = -\frac{1}{N}\sum_i \big[y_i \log p_i + (1-y_i)\log(1-p_i)\big]
\]

### 6.5 Backpropagation (`_backward`)

Because the output activation is sigmoid and the loss is BCE, the output delta simplifies to
\( \delta_{out} = (a_{out} - y) / N \).
Then for each layer from last to first:

- `grad_W = a_{prev}ᵀ · δ`
- `grad_b = Σ δ` (over the batch)
- propagate: `δ_prev = (δ · W_lᵀ) ⊙ a_prev(1 − a_prev)` — the sigmoid derivative expressed from the cached activation
- update: `W_l -= lr · grad_W`, `b_l -= lr · grad_b`

This is the "refine weights and biases" step of the resume line, done by hand.

### 6.6 Training loop (`train`)

Mini-batch gradient descent: shuffle each epoch (seeded), iterate in batches of 64, forward + backward per batch, then record the full-dataset BCE for the epoch's loss history. Defaults: learning rate 0.18, 180 epochs.

### 6.7 Persistence

`save()` writes layer sizes, learning rate, `W{i}`, `b{i}` to a `.npz`; `load()` restores an identical network.

### 6.8 Explainability helpers

- `effective_input_weights()` collapses `W₀ · W₁ · W₂` into one signed weight per input.
- `feature_contributions(x_norm)` = normalised input × effective weight. Positive pushes toward *fake*, negative toward *authentic*. This is a linearised approximation used for explanation only, not for prediction.

## 7. Training pipeline and results

File: `backend/app/train.py`.

1. If `profiles.csv` is missing, generate it.
2. Load features `X` (14 columns) and target `y = 1 if label == "fake"`.
3. **Stratified 80/20 split** (seeded) so both classes keep their ratio in train and test.
4. Fit min-max scaler on train; normalise both splits.
5. Train `NeuralNetwork([14,16,8,1], lr=0.18)` for 180 epochs, batch 64.
6. Evaluate on test at threshold 0.5.
7. Write `ann_weights.npz`, `scaler.json`, `metrics.json`.

### Results (held-out test, 800 profiles)

| Metric | Value |
| --- | --- |
| Accuracy | **0.9925** |
| Precision (fake class) | 0.9942 |
| Recall (fake class) | 0.9885 |
| F1 | 0.9914 |
| Confusion matrix | TN 449 · FP 2 · FN 4 · TP 345 |
| Train / test size | 3,200 / 800 |
| Loss | 0.6806 (epoch 1) → 0.0359 (epoch 180) |

Because the corpus is synthetic, these numbers show the pipeline learns the encoded cues cleanly; they are not a claim about real-world performance (see §15).

## 8. Inference and explainability

`POST /api/predict` flow (`main.py`):

1. Validate `PredictBody` (all features optional with defaults; extra keys ignored).
2. Derive missing values: `username_length` and `username_digit_ratio` from `username` if supplied; `profile_completeness` if not given.
3. Build the 14-vector in canonical order → normalise with `scaler.json`.
4. `p_fake = model.predict_proba(x)`; `score = 1 − p_fake` (authenticity).
5. **Verdict thresholds:** score ≥ 0.65 → `Likely authentic`; score ≤ 0.35 → `Likely fake`; otherwise `Suspicious` (label follows the 0.5 side).
6. `confidence = max(score, 1 − score)`.
7. Top 6 feature contributions by magnitude, each with `direction` (`fake`/`authentic`), raw `value`, and signed `contribution`.

Model artifacts are loaded per request from disk (small files, keeps hot-reload simple); if they are missing the API returns 503 with the retrain instruction.

## 9. REST API reference

Base URL (dev): `http://127.0.0.1:8001`. All responses JSON.

| Method | Path | Query / body | Response |
| --- | --- | --- | --- |
| GET | `/api/health` | — | `{status: "ok"|"degraded", service: "signalguard", model_ready}` |
| GET | `/api/metrics` | — | Contents of `metrics.json` plus `feature_labels` |
| GET | `/api/platforms` | — | `{by_platform: {facebook, instagram, twitter}, by_label: {real, fake}, total}` |
| GET | `/api/samples` | `label=fake|real` (optional), `limit=1..24` (default 8) | `{samples: [{username, platform, label, …14 features}], label}` — random draw |
| POST | `/api/predict` | JSON `PredictBody` (any subset of the 14 features, optional `username`, `platform`) | `PredictResponse` below |

`PredictResponse`

```json
{
  "score": 0.9312,
  "p_fake": 0.0688,
  "label": "authentic",
  "verdict": "Likely authentic",
  "confidence": 0.9312,
  "explanation": [
    {"feature": "friend_request_mutuals", "label": "Mutual connections", "value": 38, "contribution": -1.92, "direction": "authentic"},
    {"feature": "username_digit_ratio", "label": "Username digit ratio", "value": 0.0, "contribution": -0.41, "direction": "authentic"}
  ]
}
```

Interactive docs: `http://127.0.0.1:8001/docs` (Swagger UI generated by FastAPI).

## 10. Frontend: screens and user workflow

### 10.1 Shell (`App.jsx`)

Header with product name, three tabs (Inspect / Gallery / Model lab), and a health pill that polls `/api/health` and shows *Model ready* or *API offline*.

### 10.2 Inspect (`views/Inspect.jsx`)

- Opens pre-filled with a typical authentic profile and scores it immediately.
- Left: grouped controls — numeric sliders with live values, two toggles (profile photo, cover photo), username text field (drives digit ratio and length), platform select.
- Right: authenticity **Gauge** (SVG arc), verdict badge, confidence, and **explanation chips** (top 6 contributions coloured by direction).
- Any change triggers a debounced (~220 ms) `POST /api/predict`.
- Presets: *Typical authentic* and *Suspicious pattern* set all controls at once.
- Error state shows the API message and keeps the last result.

### 10.3 Gallery (`views/Gallery.jsx`)

- Loads 8 real + 8 fake samples via `/api/samples`.
- Filters by label and platform; cards show username, platform, key stats.
- *Score request* on a card calls predict and displays the verdict inline.
- *Inspect signals* loads the card's values into the Inspect view.

### 10.4 Model lab (`views/ModelLab.jsx`)

- Layer diagram rendered from `metrics.architecture`.
- Hyper-parameters (activation, loss, optimiser, lr, epochs, batch).
- Four metric tiles, confusion matrix grid, and the loss curve as a **Sparkline** over 180 epochs.
- Feature list with human labels.

### 10.5 Responsive behaviour

Verified at 1360 px and 390 px. Controls stack vertically on narrow screens; gauge stays above the fold.

## 11. Running the application

### Prerequisites

- Python 3.11 or newer (`python --version`)
- Node.js 18 or newer (`node --version`)
- Git

### One command

```powershell
# Windows PowerShell
git clone https://github.com/Saisriya2003/fake-profile-detector.git
cd fake-profile-detector
.\start.ps1
# if scripts are blocked: powershell -ExecutionPolicy Bypass -File .\start.ps1
```

```bash
# macOS / Linux
git clone https://github.com/Saisriya2003/fake-profile-detector.git
cd fake-profile-detector
./start.sh
```

First run creates `backend/.venv`, installs `requirements.txt`, runs `npm install`, then starts both servers and opens `http://localhost:5173`.

### Manual

```bash
# API
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload

# UI (second terminal)
cd frontend
npm install
npm run dev          # http://localhost:5173
```

### Production build

`cd frontend && npm run build` → `frontend/dist/`. Serve statically with `/api` proxied to the FastAPI process, or build with `VITE_API_URL=https://your-api` to call it directly (then add that origin to CORS in `main.py`).

## 12. Configuration

| Variable | Where | Effect |
| --- | --- | --- |
| `VITE_API_URL` | frontend build/dev env | Absolute API base; default empty → same-origin `/api` via Vite proxy |
| Port 8001 | `start.*`, `vite.config.js` proxy | Change both if you move the API |

The backend has no environment variables; all paths are relative to `backend/`.

## 13. Testing, CI, and verification

### GitHub Actions (`.github/workflows/ci.yml`)

Runs on every push and pull request to `main`, on Ubuntu:

- **backend** job: Python 3.12 → `pip install -r requirements.txt` → import `app.main` and load the weights → start Uvicorn on 8001 → `curl /api/health`, `/api/metrics`, `POST /api/predict` → assert 200.
- **frontend** job: Node 20 → `npm ci` → `npm run build`.

Status badge in `README.md`; last run green.

### End-to-end verification performed

Playwright (Edge) at 1360 px and 390 px, 10 steps: health pill, pre-scored Inspect, slider re-score, both presets, Gallery load and filters, *Score request*, *Inspect signals* hand-off, Model lab metrics and loss curve render, mobile layout. Zero console errors and zero failed network requests. Fresh clone from GitHub set up strictly by the README succeeded on Windows.

## 14. Retraining and extending

```bash
cd backend
python -m app.generate_data      # optional: regenerate profiles.csv (edit n/seed in the file)
python -m app.train              # retrain; overwrites models/*
```

- Change architecture: edit `ARCHITECTURE` in `train.py` (e.g. `[14, 32, 16, 1]`). The API reads layer sizes from the `.npz`, no other change needed.
- Change hyper-parameters: `train(epochs=, batch_size=, learning_rate=)`.
- Add a feature: append to `FEATURE_NAMES`, `FEATURE_LABELS`, `FEATURE_BOUNDS` in `features.py`, generate it in `generate_data.py`, mirror in `frontend/src/features.js`, retrain.
- Use a real dataset: produce a CSV with the 14 columns plus `label` (`real`/`fake`), `username`, `platform`, and point `DATA_PATH` at it.

## 15. Limitations and production path

- **Synthetic data.** Metrics reflect the generator's encoded cues; a real labelled dataset (with consent and platform policy compliance) is needed before any operational use.
- **No persistence of predictions.** The API is stateless by design; add a database if audit logging is required.
- **Threshold policy** (0.35 / 0.65) is a product decision; expose it in configuration for real deployments.
- **Model reload per request** is fine for a 3 KB `.npz` but should be cached in a production server.
- Production path: real data → retrain → containerise API (`uvicorn` behind a reverse proxy) → static UI on a CDN → authentication in front of `/api/predict`.

## 16. Glossary

- **ANN / MLP** — artificial neural network / multilayer perceptron: stacked fully connected layers.
- **Sigmoid** — squashing activation mapping any real number to (0, 1); used here on every layer.
- **BCE** — binary cross-entropy, the loss for two-class probabilities.
- **Backpropagation** — computing gradients layer by layer from the output back to the input using the chain rule.
- **Mini-batch gradient descent** — updating weights after each small batch rather than the whole dataset.
- **Xavier initialisation** — scaling initial weights by 1/√fan_in to keep activations well-conditioned.
- **Min-max scaling** — mapping each feature to \[0, 1\] using training minima and maxima.
- **Stratified split** — a train/test split preserving class proportions.
- **Precision / recall / F1** — of predicted fakes, how many were fake / of true fakes, how many were caught / their harmonic mean.
