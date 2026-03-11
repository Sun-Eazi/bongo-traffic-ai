"""
bongo-traffic-ai — HuggingFace Spaces Frontend
Calls Railway backend (Flask + best.pt YOLO).
Set BACKEND_URL in HF Spaces → Settings → Repository Secrets.
"""

import gradio as gr
import requests
import base64
import os
from PIL import Image
import io

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:5000")


def detect(image, location, team_number):
    if image is None:
        return None, "⚠️ Pakia picha kwanza!"

    try:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        buf.seek(0)

        resp = requests.post(
            f"{BACKEND_URL}/detect",
            files={"file": ("image.jpg", buf, "image/jpeg")},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

    except requests.exceptions.ConnectionError:
        return None, (
            f"❌ **Backend haifiki!**\n\n"
            f"URL iliyowekwa: `{BACKEND_URL}`\n\n"
            "**Suluhisho:**\n"
            "1. Nenda HuggingFace Spaces → Settings → Repository Secrets\n"
            "2. Ongeza: `BACKEND_URL = https://your-app.up.railway.app`\n"
            "3. Restart Space"
        )
    except Exception as e:
        return None, f"❌ Hitilafu: {str(e)}"

    # Decode annotated image
    ann_b64 = data.get("annotated_image", "")
    ann_img = None
    if ann_b64 and "," in ann_b64:
        img_bytes = base64.b64decode(ann_b64.split(",")[1])
        ann_img   = Image.open(io.BytesIO(img_bytes))

    dets     = data.get("detections", [])
    summary  = data.get("summary", {})
    top_cls  = data.get("top_class", "—")
    top_conf = data.get("top_confidence", 0)
    total    = data.get("total_detections", 0)
    ms       = data.get("inference_ms", "—")
    engine   = data.get("engine", "—")

    lines = [
        "## 🎯 Matokeo ya Detection\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Total Detected** | {total} |",
        f"| **Top Class** | {top_cls} |",
        f"| **Top Confidence** | {top_conf:.1f}% |",
        f"| **Inference Time** | {ms}ms |",
        f"| **Engine** | {engine} |",
    ]
    if location:
        lines.append(f"| **Location** | {location} |")
    if team_number:
        lines.append(f"| **Team** | {team_number} |")

    if summary:
        lines.append("\n### 🚗 Vehicles Found")
        for cls, cnt in sorted(summary.items(), key=lambda x: -x[1]):
            dot = ("🟠" if "bajaj" in cls.lower() else
                   "🟢" if "boda"  in cls.lower() else
                   "🟡" if "dala"  in cls.lower() else "🔵")
            lines.append(f"- {dot} **{cls}**: {cnt} detected")

    if dets:
        lines.append("\n### 📋 Top 10 Detections")
        lines.append("| # | Class | Confidence | BBox |")
        lines.append("|---|-------|-----------|------|")
        for i, d in enumerate(dets[:10], 1):
            bbox = d.get("bbox", [])
            bbox_str = f"[{', '.join(str(round(v)) for v in bbox)}]"
            lines.append(f"| {i} | **{d['class']}** | {d['confidence']:.1f}% | {bbox_str} |")

    if total == 0:
        lines.append(
            "\n> ℹ️ Hakuna gari iliyopatikana. "
            "Jaribu picha nyingine yenye magari wazi zaidi."
        )

    return ann_img, "\n".join(lines)


def health_check():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=8)
        d = r.json()
        classes = ", ".join(d.get("classes", []))
        return (
            f"✅ **Backend Online**\n"
            f"- Engine: `{d.get('engine')}`\n"
            f"- Model: `{d.get('model')}`\n"
            f"- Classes: {classes}"
        )
    except Exception as e:
        return (
            f"❌ **Backend Offline** — `{BACKEND_URL}`\n\n"
            f"Error: `{str(e)}`\n\n"
            "Weka `BACKEND_URL` katika HF Spaces → Settings → Repository Secrets"
        )


# ── UI ────────────────────────────────────────────────────────
with gr.Blocks(
    title="bongo-traffic-ai",
    theme=gr.themes.Base(
        primary_hue="amber",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Outfit"), "sans-serif"],
    ),
    css="""
    .gradio-container { max-width: 1100px !important; }
    footer { display: none !important; }
    """
) as demo:

    gr.HTML("""
    <div style="text-align:center;padding:28px 0 12px">
      <h1 style="font-size:2.2rem;font-weight:800;margin:0;letter-spacing:-1px">
        🚦 bongo-traffic-ai
      </h1>
      <p style="color:#94a3b8;margin:8px 0 0;font-size:1rem">
        YOLO Detection · Dar es Salaam · UAUT Hackathon 2026
      </p>
    </div>
    """)

    with gr.Row():

        # ── LEFT ──────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Pakia Picha")
            img_input = gr.Image(type="pil", label="Picha ya Dashcam", height=280)

            with gr.Row():
                location = gr.Dropdown(
                    choices=["Mbezi","Buguruni","Makumbusho",
                             "Kariakoo","Ubungo","Kimara","Tegeta"],
                    label="📍 Location",
                    value="Mbezi",
                )
                team = gr.Textbox(label="👥 Team", placeholder="T01", max_lines=1)

            detect_btn = gr.Button("▶  RUN DETECTION", variant="primary", size="lg")

            gr.Markdown("---")
            gr.Markdown("### 🔌 Backend Status")
            health_md  = gr.Markdown("*Inaangalia...*")
            health_btn = gr.Button("Refresh Status", size="sm", variant="secondary")

        # ── RIGHT ─────────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 🎯 Picha na Bounding Boxes")
            img_output  = gr.Image(type="pil", label="Annotated Result",
                                   height=340, interactive=False)
            gr.Markdown("### 📊 Summary")
            summary_out = gr.Markdown("*Pakia picha na bonyeza Run Detection...*")

    gr.Markdown(
        "---\n"
        "**Jinsi ya kutumia:** Pakia picha ya barabara → Chagua location → "
        "**RUN DETECTION** → Uone magari yaliyopatikana na bounding boxes."
    )

    detect_btn.click(
        fn=detect,
        inputs=[img_input, location, team],
        outputs=[img_output, summary_out],
        show_progress=True,
    )
    health_btn.click(fn=health_check, outputs=health_md)
    demo.load(fn=health_check, outputs=health_md)


if __name__ == "__main__":
    demo.launch()
