"""
bongo-traffic-ai — Railway Backend
Real YOLO inference using best.pt only. No external AI APIs.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os, uuid, time, base64, traceback
import cv2
import numpy as np

app = Flask(__name__)
CORS(app, origins="*")

# ── Model Loading ────────────────────────────────────────────
HERE       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(HERE, "best.pt"))
PORT       = int(os.environ.get("PORT", 5000))

model  = None
ENGINE = None

CLASS_COLORS = {
    "Bajaj":      "#FF6B2B",
    "Bodaboda":   "#00C9B1",
    "Daladala":   "#F5C518",
    "Car":        "#7C6FFF",
    "Truck":      "#FF4D8D",
    "Motorcycle": "#40C4FF",
    "Bus":        "#F5C518",
}

def color_for(name: str) -> str:
    return CLASS_COLORS.get(name, "#9CA3AF")

def load_model():
    global model, ENGINE
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"best.pt not found at: {MODEL_PATH}\n"
            "Copy best.pt into the backend/ directory before deploying."
        )
    try:
        from ultralytics import YOLO
        model  = YOLO(MODEL_PATH)
        ENGINE = "ultralytics"
        if hasattr(model, "names") and isinstance(model.names, dict):
            for k, v in model.names.items():
                if v not in CLASS_COLORS:
                    CLASS_COLORS[v] = color_for(v)
        print(f"[bongo-traffic-ai] ✅ ultralytics · classes={list(set(CLASS_COLORS.keys()))}")
    except Exception as e:
        if "YOLOv5" in str(e) or "models" in str(e):
            import torch
            model  = torch.hub.load("ultralytics/yolov5", "custom",
                                    path=MODEL_PATH, force_reload=False)
            model.eval()
            ENGINE = "yolov5"
            print("[bongo-traffic-ai] ✅ yolov5-hub loaded")
        else:
            raise

load_model()

UPLOAD_DIR = os.path.join(HERE, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────
def class_name(cid: int) -> str:
    try:
        names = getattr(model, "names", {})
        if isinstance(names, dict) and cid in names:
            return str(names[cid])
    except Exception:
        pass
    return f"class_{cid}"

def run_inference(img_bgr):
    dets = []
    if ENGINE == "ultralytics":
        results = model(img_bgr, verbose=False)
        for r in results:
            for box in r.boxes:
                cid  = int(box.cls[0])
                conf = float(box.conf[0]) * 100
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                name = class_name(cid)
                dets.append({
                    "class":      name,
                    "class_id":   cid,
                    "confidence": round(conf, 1),
                    "bbox":       [round(x1,1), round(y1,1), round(x2,1), round(y2,1)],
                    "color":      color_for(name),
                })
    else:
        res   = model(img_bgr)
        preds = res.xyxy[0].cpu().numpy()
        for row in preds:
            x1,y1,x2,y2,conf,cid = row[:6]
            name = class_name(int(cid))
            dets.append({
                "class":      name,
                "class_id":   int(cid),
                "confidence": round(float(conf)*100, 1),
                "bbox":       [round(float(v),1) for v in (x1,y1,x2,y2)],
                "color":      color_for(name),
            })
    dets.sort(key=lambda d: d["confidence"], reverse=True)
    return dets

def annotate(img_bgr, dets):
    out  = img_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    lw   = max(2, int(img_bgr.shape[1] * 0.003))
    for d in dets:
        x1,y1,x2,y2 = [int(v) for v in d["bbox"]]
        hex_c = d["color"].lstrip("#")
        r,g,b = int(hex_c[0:2],16), int(hex_c[2:4],16), int(hex_c[4:6],16)
        col   = (b, g, r)
        cv2.rectangle(out, (x1,y1), (x2,y2), col, lw)
        cs = int(min(x2-x1, y2-y1) * 0.18)
        for pts in [((x1,y1+cs),(x1,y1),(x1+cs,y1)),
                    ((x2-cs,y1),(x2,y1),(x2,y1+cs)),
                    ((x1,y2-cs),(x1,y2),(x1+cs,y2)),
                    ((x2-cs,y2),(x2,y2),(x2,y2-cs))]:
            for i in range(len(pts)-1):
                cv2.line(out, pts[i], pts[i+1], col, lw*2)
        fs   = max(0.45, img_bgr.shape[1] / 2000)
        text = f"{d['class']}  {d['confidence']:.0f}%"
        (tw, th), bl = cv2.getTextSize(text, font, fs, 1)
        pad = 4
        cv2.rectangle(out, (x1, y1-th-bl-pad*2), (x1+tw+pad*2, y1), col, -1)
        cv2.putText(out, text, (x1+pad, y1-bl-pad), font, fs, (0,0,0), 1, cv2.LINE_AA)
    return out

def to_b64(img_bgr) -> str:
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


# ── Routes ───────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "name":    "bongo-traffic-ai",
        "version": "3.0.0",
        "engine":  ENGINE,
        "classes": list(set(CLASS_COLORS.keys())),
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":  "ok",
        "engine":  ENGINE,
        "model":   os.path.basename(MODEL_PATH),
        "classes": list(set(CLASS_COLORS.keys())),
    })

@app.route("/detect", methods=["POST", "OPTIONS"])
def detect():
    if request.method == "OPTIONS":
        return "", 204
    if "file" not in request.files:
        return jsonify({"error": "No file. Use multipart/form-data key 'file'."}), 400

    f   = request.files["file"]
    ext = os.path.splitext(f.filename)[1].lower() or ".jpg"
    tmp = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{ext}")
    f.save(tmp)

    try:
        is_video = ext in {".mp4",".mov",".avi",".mkv",".webm",".m4v"}

        if not is_video:
            img = cv2.imread(tmp)
            if img is None:
                return jsonify({"error": "Cannot decode image."}), 400
            t0   = time.perf_counter()
            dets = run_inference(img)
            ms   = round((time.perf_counter()-t0)*1000, 1)
            ann  = annotate(img, dets)
            summary = {}
            for d in dets:
                summary[d["class"]] = summary.get(d["class"], 0) + 1
            top = dets[0] if dets else None
            return jsonify({
                "type":             "image",
                "engine":           ENGINE,
                "inference_ms":     ms,
                "detections":       dets,
                "annotated_image":  to_b64(ann),
                "summary":          summary,
                "top_class":        top["class"] if top else "—",
                "top_confidence":   top["confidence"] if top else 0,
                "total_detections": len(dets),
            })

        max_frames = int(request.form.get("max_frames", 40))
        stride     = int(request.form.get("stride", 5))
        cap = cv2.VideoCapture(tmp)
        if not cap.isOpened():
            return jsonify({"error": "Cannot open video."}), 400
        fps      = cap.get(cv2.CAP_PROP_FPS) or 30
        total_f  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        all_dets = []
        summary  = {}
        preview  = None
        fi = fp  = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok: break
                if fi % max(1, stride) != 0:
                    fi += 1; continue
                if fp >= max_frames: break
                dets = run_inference(frame)
                for d in dets:
                    d["frame_index"] = fi
                    d["timestamp_s"] = round(fi/fps, 2)
                    all_dets.append(d)
                    summary[d["class"]] = summary.get(d["class"],0)+1
                if preview is None and dets:
                    preview = to_b64(annotate(frame, dets))
                fi += 1; fp += 1
        finally:
            cap.release()
        all_dets.sort(key=lambda d: d["confidence"], reverse=True)
        top = all_dets[0] if all_dets else None
        return jsonify({
            "type":             "video",
            "engine":           ENGINE,
            "total_frames":     total_f,
            "frames_processed": fp,
            "detections":       all_dets[:500],
            "preview_image":    preview,
            "summary":          summary,
            "top_class":        top["class"] if top else "—",
            "top_confidence":   top["confidence"] if top else 0,
            "total_detections": len(all_dets),
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc(5)}), 500
    finally:
        try: os.remove(tmp)
        except Exception: pass


if __name__ == "__main__":
    print(f"[bongo-traffic-ai] Starting on http://0.0.0.0:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
