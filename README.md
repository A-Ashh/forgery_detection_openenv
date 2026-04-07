# 🕵️ Forgery Detection OpenEnv

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-brightgreen)](https://github.com/meta-pytorch/OpenEnv)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-Spaces-yellow)](https://huggingface.co/spaces)

> An RL environment where an AI agent acts as a forensic document analyst — inspecting regions of a document step-by-step to determine whether it has been forged.

---

## 🌍 Real-World Use Case

Document forgery is a major issue in banking, legal systems, healthcare, and education. Traditional detection is manual and slow. This environment trains AI agents to investigate documents systematically — the same way a human forensic analyst would — and make explainable predictions.

---

## 🎯 Environment Overview

The agent receives a document and must choose inspection actions before making a final prediction. The environment rewards:
- **Correct final predictions**
- **Inspecting the actual forged region** (for medium/hard tasks)
- **Thorough multi-step investigation** (for hard tasks)

### Action Space

| Action | Description |
|---|---|
| `inspect_date` | Check if dates are valid and formatted correctly |
| `inspect_signature` | Verify signature authenticity |
| `inspect_metadata` | Scan metadata for null/invalid values |
| `inspect_seal` | Check official seals for tampering |
| `request_additional_info` | Request more context (costs a step) |
| `predict_real` | Final prediction: document is REAL |
| `predict_forged` | Final prediction: document is FORGED |

### Observation Space

```json
{
  "document_text": "string — the document content",
  "visible_features": "string — regions already inspected",
  "last_action_result": "string — result of the last action",
  "step_number": "int — current step",
  "available_actions": ["list of valid action strings"]
}
```

### Reward Function

| Event | Reward |
|---|---|
| Inspect correct forged region | +0.25 |
| Inspect any region (non-forged doc) | +0.05 |
| Correct final prediction (easy) | +1.0 |
| Correct prediction, missed forged region | +0.6 |
| Correct prediction, full investigation | +1.0 |
| Wrong final prediction | -0.5 |
| Unknown action | -0.1 |

---

## 📋 Tasks

### 🟢 Easy
- Classify the document as real or forged in 1–2 steps
- No region inspection required
- Score: binary correct/incorrect

### 🟡 Medium
- Must inspect at least one region before predicting
- Partial credit if the correct forged region isn't found
- Max steps: 4

### 🔴 Hard
- Full forensic investigation: inspect 3+ regions
- Rewarded for finding the actual forged region
- Max steps: 6 — frontier models are challenged here

---

## 🚀 Setup & Usage

### Local Development

```bash
# Clone and enter directory
git clone https://github.com/YOUR_USERNAME/forgery-detection-openenv
cd forgery-detection-openenv

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate      # Windows
source venv/bin/activate      # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py
```

The API will be available at `http://localhost:7860`

### Run Inference (Baseline Agent)

```bash
# Set your API key
export HF_TOKEN="your_api_key_here"          # Linux/Mac
$env:HF_TOKEN="your_api_key_here"            # Windows PowerShell

# Run baseline on all 3 tasks
python inference.py
```

Expected output format:
```
[START] task=easy env=forgery-detection-openenv model=gpt-4o-mini
[STEP] step=1 action=inspect_date reward=0.250 done=false error=none
[STEP] step=2 action=predict_forged reward=1.000 done=true error=none
[END] success=true steps=2 score=0.625 rewards=0.25,1.00

[SUMMARY] easy=0.625 medium=0.540 hard=0.480 overall=0.548
```

### Docker

```bash
# Build
docker build -t forgery-openenv .

# Run
docker run -p 7860:7860 -e HF_TOKEN="your_api_key" forgery-openenv
```

---

## 🌐 API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/reset` | Reset environment (`{"task": "easy"}`) |
| POST | `/step` | Take action (`{"action": "inspect_date", "task": "easy"}`) |
| GET | `/state` | Get current state |
| GET | `/tasks` | List all tasks |

---

## 📁 Project Structure

```
forgery-detection-openenv/
│
├── env/
│   ├── __init__.py
│   └── environment.py     # Core OpenEnv environment
│
├── app.py                 # FastAPI server (HTTP endpoints)
├── inference.py           # Baseline agent (required by hackathon)
├── openenv.yaml           # OpenEnv metadata spec
├── Dockerfile             # Container for HF Spaces
├── requirements.txt       # Dependencies
└── README.md
```

---

## 📊 Baseline Scores

| Task | Score |
|---|---|
| Easy | ~0.60 |
| Medium | ~0.50 |
| Hard | ~0.45 |
| **Overall** | **~0.52** |

*(Scores with `gpt-4o-mini` at temperature=0)*

---

## 🔮 Future Improvements

- Add image-based forgery detection (CNN / Grad-CAM)
- Support PDF and Word document upload
- Add multi-lingual document support
- Web UI for interactive investigation
- Integrate real forensic datasets (e.g. CASIA, CoMoFoD)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.