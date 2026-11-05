# ChestVision AI

Explainable chest X-ray analysis — DenseNet121 classifier + U-Net lung segmenter + Grad-CAM visualizations.

## Architecture

```
┌────────────────────────────────────────────────────┐
│                  Docker Compose                    │
│                                                    │
│  ┌─────────────────┐   ┌──────────────────────┐   │
│  │ frontend        │   │ backend              │   │
│  │ React + Vite    │──▶│ FastAPI (JSON API)   │   │
│  │ nginx :3000→80  │   │ uvicorn :8000        │   │
│  └─────────────────┘   └──────────────────────┘   │
└────────────────────────────────────────────────────┘
```

## Quick Start (Docker Compose)

```bash
# Build and run both services
docker compose up --build

# Access the app
open http://localhost:3000
```

## Development

### Backend
```bash
cd "Chest X Ray"
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

### Frontend
```bash
cd "Chest X Ray/frontend"
npm install
npm run dev        # http://localhost:3000
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/api/health` | Health check |
| GET  | `/api/ready` | Model readiness |
| GET  | `/api/info` | Model info & class definitions |
| POST | `/api/analyze` | Single X-ray analysis |
| POST | `/api/compare` | Side-by-side X-ray comparison |
| GET  | `/api/export/{fmt}` | Export report (`txt` or `json`) |

## Classification Classes

- **COVID-19** — Bilateral ground-glass opacities
- **Lung Opacity** — Airspace opacity (fluid, pus, tissue)
- **Normal** — Clear lung fields
- **Viral Pneumonia** — Interstitial infiltrates

## Disclaimer

ChestVision AI is an educational and research prototype. Not for clinical use.
