import sys
import subprocess
import gc
import os
import shutil

# ১. প্রয়োজনীয় লাইব্রেরি ইনস্টল (imageio-ffmpeg সহ)
print("[Driver 0/5] Installing dependencies...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
    "diffusers>=0.31.0",
    "transformers",
    "accelerate",
    "scipy",
    "sentencepiece",
    "numpy",
    "imageio",
    "imageio-ffmpeg"
], check=True)

import requests
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from diffusers import LTXImageToVideoPipeline, AudioLDM2Pipeline
from diffusers.utils import export_to_video
import scipy.io.wavfile

# ২. প্রম্পট রিড করা
prompt = "Cinematic OLED aesthetic, dark cyberpunk neon interior, dramatic lighting, 8k"
if os.path.exists("prompt.txt"):
    with open("prompt.txt", "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content: prompt = content

print(f"[Driver] Target Prompt: {prompt}")

# ৩. বেস ফ্রেম জেনারেশন
print("[Driver 1/4] Generating base visual frame...")
flux_prompt = f"masterpiece, photorealistic 8k, raw film footage, cinematic lighting, {prompt}"
img_url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(flux_prompt)}?model=flux&width=704&height=480&nologo=true"

try:
    res = requests.get(img_url, timeout=60)
    init_img = Image.open(BytesIO(res.content)).convert("RGB")
except Exception as e:
    print(f"[Driver] Image fallback triggered: {e}")
    init_img = Image.new("RGB", (704, 480), color=(10, 15, 20))

init_img.save("base.jpg")

# ৪. LTX-Video রেন্ডারিং (T4 GPU Safe)
print("[Driver 2/4] Rendering video frames (Uncensored on T4)...")
v_pipe = LTXImageToVideoPipeline.from_pretrained(
    "Lightricks/LTX-Video",
    torch_dtype=torch.float16
)
v_pipe.enable_model_cpu_offload()

if hasattr(v_pipe, "vae") and hasattr(v_pipe.vae, "enable_tiling"):
    v_pipe.vae.enable_tiling()

if hasattr(v_pipe, "safety_checker"):
    v_pipe.safety_checker = None

frames = v_pipe(
    image=init_img,
    prompt=prompt,
    negative_prompt="blurry, distorted, low resolution, watermark, deformed",
    width=704,
    height=480,
    num_frames=81,              # ৮১ ফ্রেম @ ৮ এফপিএস = ১০.১ সেকেন্ড
    num_inference_steps=25,
    guidance_scale=3.0
).frames[0]

temp_raw_video = "temp_raw.mp4"
export_to_video(frames, temp_raw_video, fps=8)

# ৫. ডিস্ক ও VRAM সম্পূর্ণ মুক্ত করা (যাতে ডিস্ক ফুল না হয়)
print("[Driver] Clearing VRAM and purging LTX cache to free disk space...")
del v_pipe
del frames
gc.collect()
torch.cuda.empty_cache()
shutil.rmtree(os.path.expanduser("~/.cache/huggingface"), ignore_errors=True)

# ৬. সাউন্ড সিন্থেসিস (Float32 স্ট্যাবিলিটি সহ)
print("[Driver 3/4] Generating cinematic atmospheric audio...")
a_pipe = AudioLDM2Pipeline.from_pretrained(
    "cvssp/audioldm2",
    torch_dtype=torch.float32
).to("cuda")

audio_prompt = f"Cinematic atmospheric drones, dark ambient soundtrack, deep bass impact, {prompt}"
audio_data = a_pipe(
    audio_prompt,
    num_inference_steps=20,
    audio_length_in_s=10.2
).audios[0]

audio_data_int16 = (audio_data * 32767).clip(-32768, 32767).astype(np.int16)
temp_audio = "temp_audio.wav"
scipy.io.wavfile.write(temp_audio, rate=16000, data=audio_data_int16)

del a_pipe
gc.collect()
torch.cuda.empty_cache()
shutil.rmtree(os.path.expanduser("~/.cache/huggingface"), ignore_errors=True)

# ৭. FFmpeg মাস্টার এনকোডিং
print("[Driver 4/4] Master video rendering via FFmpeg...")
final_output = "argent_render_output.mp4"
subprocess.run([
    "ffmpeg", "-y",
    "-i", temp_raw_video,
    "-i", temp_audio,
    "-c:v", "libx264",
    "-crf", "19",
    "-preset", "fast",
    "-pix_fmt", "yuv420p",
    "-c:a", "aac",
    "-b:a", "192k",
    "-shortest",
    final_output
], check=True)

if os.path.exists(temp_raw_video): os.remove(temp_raw_video)
if os.path.exists(temp_audio): os.remove(temp_audio)
print("[Driver] Pipeline complete: argent_render_output.mp4 is ready!")
