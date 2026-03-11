# 🚦 bongo-traffic-ai

**Real YOLO · best.pt · Free · UAUT Hackathon 2026**

## Muundo

```
bongo-traffic-ai/          ← GitHub repo (jina hili hili)
├── backend/               → deploy Railway  (best.pt + Flask)
│   ├── app.py
│   ├── requirements.txt
│   ├── Procfile
│   ├── railway.json
│   ├── Dockerfile
│   └── best.pt            ← weka hapa!
└── frontend/              → deploy HuggingFace Spaces (UI)
    ├── app.py
    ├── requirements.txt
    └── README.md
```

## Deploy

### 1. Weka best.pt
```bash
cp /path/to/best.pt backend/best.pt
```

### 2. Push GitHub
```bash
git add .
git commit -m "ready"
git push
```

### 3. Railway — backend/
- New Project → GitHub repo → **Root Directory: `backend`**
- Variable: `MODEL_PATH = ./best.pt`
- Copy URL: `https://bongo-traffic-ai-xxx.up.railway.app`

### 4. HuggingFace Spaces — frontend/
- New Space → SDK: Gradio → upload `frontend/` files
- Secret: `BACKEND_URL = https://bongo-traffic-ai-xxx.up.railway.app`

### 5. ✅ Done
Mtumiaji anaenda HuggingFace link → apakia picha → aona matokeo.
