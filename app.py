import os, uuid, logging
from datetime import datetime
from flask import Flask, request, jsonify, render_template, session, send_file
import torch
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from prompts import INITIAL_PROMPT, INITIAL_STYLES, QUESTIONS_OPTIONS
import openai

app = Flask(__name__)
app.secret_key = "your_secret_key"
openai_api_key = os.environ.get("OPENAI_API_KEY")

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = get_device()
print("Using device:", device)
torch_dtype = torch.float16 if device == "mps" else torch.float32
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch_dtype
).to(device)
pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
pipeline.enable_attention_slicing()
lora_path = os.path.join("models", "lora", "SDXL-tattoo-Lora.safetensors")
if PeftModel and os.path.exists(lora_path):
    try:
        pipeline.unet.load_lora_adapter(lora_path)
        print("Loaded LoRA adapter from", lora_path)
    except Exception as e:
        logging.warning(f"載入 LoRA adapter 失敗: {e}, 使用未調整模型")
elif not PeftModel:
    logging.warning("未安裝 peft，請 `pip install peft` 以啟用 LoRA")
elif not os.path.exists(lora_path):
    logging.warning(f"LoRA 權重未找到：{lora_path}")

def generate_prompt_sdxl(style, user_input, color_preference, line_style, composition_preference):
    prompt = "clean tattoo outline drawing, "
    prompt += f"abstract representation of {user_input}, {color_preference} (primarily black lines on a white background), "
    prompt += "plain white background or transparent background, focus on clear precise lines."
    return prompt

def generate_prompt_dalle(style, user_input, color_preference, line_style, composition_preference):
    prompt = "Generate a clean tattoo outline drawing, "
    prompt += f"representing an abstract concept of {user_input}, "
    prompt += f"using {color_preference} (primarily black lines on a white background). "
    prompt += "Focus on clear, precise lines. The background should be plain white or transparent."
    return prompt

@app.route("/")
def index():
    session.clear()
    styles = {
        "S1": {"text": "日式傳統", "image": "/static/image/japense.jpeg"},
        "S2": {"text": "寫實", "image": "/static/image/real.jpeg"},
        "S3": {"text": "水彩", "image": "/static/image/watercolor.jpeg"},
        # ... 更多風格
    }
    return render_template(
        "index.html",
        initial_prompt=INITIAL_PROMPT,
        styles=styles
    )

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    msg = data.get("message", "").strip()

    if "phase" not in session:
        key = msg.upper()
        if key in INITIAL_STYLES:
            session.update(phase="asking", style_key=key, answers={}, q_idx=0, model="dalle")
            q = QUESTIONS_OPTIONS[0]
            follow_ups = [{"key": k, "text": v} for k, v in q["options"].items()] if q.get("type") == "choice" else []
            return jsonify({"reply": q["text"], "follow_ups": follow_ups})
        return jsonify({"reply": "請先選擇風格。", "follow_ups": []})

    elif session.get("phase") == "asking":
        idx = session["q_idx"]
        q = QUESTIONS_OPTIONS[idx]
        if q.get("type") == "choice":
            choice = msg.upper()
            if choice in q["options"]:
                session["answers"][q["key"]] = q["options"][choice]
            else:
                follow_ups = [{"key": k, "text": v} for k, v in q["options"].items()]
                return jsonify({"reply": "請從下列選項中選擇：", "follow_ups": follow_ups})
        else:
            session["answers"][q["key"]] = msg

        if idx + 1 < len(QUESTIONS_OPTIONS):
            session["q_idx"] = idx + 1
            q = QUESTIONS_OPTIONS[idx + 1]
            follow_ups = [{"key": k, "text": v} for k, v in q["options"].items()] if q.get("type") == "choice" else []
            return jsonify({"reply": q["text"], "follow_ups": follow_ups})
        else:
            session["phase"] = "generating"
            return jsonify({"reply": "好的，正在為你生成紋身設計...", "follow_ups": []})

    elif session.get("phase") == "generating":
        model_to_use = session.get("model", "sdxl")
        style_key = session.get("style_key")
        answers = session.get("answers")
        theme = answers.get("Q1", "")
        line_style = answers.get("Q2", "")
        composition_preference = answers.get("Q3", "")
        color_preference = answers.get("Q5", "")

        if model_to_use == "dalle":
            if not openai_api_key:
                return jsonify({"reply": "圖像生成服務未設定。", "follow_ups": []}), 500
            prompt = generate_prompt_dalle(style_key, theme, color_preference, line_style, composition_preference)
            try:
                client = openai.Client(api_key=openai_api_key)
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    n=1,
                    size="1024x1024",
                    quality="standard"
                )
                image_url = response.data[0].url
                session['generated_image_url'] = image_url
                return jsonify({"redirect": "/download"})
            except Exception as e:
                return jsonify({"reply": f"圖像生成失敗：{e}", "follow_ups": []}), 500
        else: # SDXL
            prompt = generate_prompt_sdxl(style_key, theme, color_preference, line_style, composition_preference)
            gen_device = device
            pipeline.to(gen_device)
            pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
            pipeline.disable_attention_slicing()
            try:
                out = pipeline(
                    prompt=prompt,
                    num_inference_steps=75,
                    guidance_scale=8.0
                )
                img = out.images[0]
                folder = os.path.join("static", "image")
                os.makedirs(folder, exist_ok=True)
                fn = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
                filepath = os.path.join(folder, fn)
                img.save(filepath)
                image_url = f"/static/image/{fn}"
                session['generated_image_path'] = filepath
                session['generated_image_url'] = image_url
                return jsonify({"redirect": "/download"})
            except Exception as e:
                return jsonify({"reply": f"圖像生成失敗：{e}", "follow_ups": []}), 500

    session.clear()
    return jsonify({"reply": "發生錯誤，請重試。", "follow_ups": []}), 400

@app.route("/download")
def download_page():
    image_url = session.get('generated_image_url')
    if image_url:
        return render_template("download.html", image_url=image_url)
    else:
        return "圖片生成失敗或已過期，請重新開始。", 400

@app.route("/download_file")
def download_file():
    image_path = session.get('generated_image_path')
    if image_path and os.path.exists(image_path):
        return send_file(image_path, as_attachment=True)
    else:
        return "圖片檔案不存在或已過期，無法下載。", 404

if __name__ == "__main__":
    app.run(debug=True)