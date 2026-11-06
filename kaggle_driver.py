import sys
import subprocess
import gc
import os
import shutil

print("[Driver 0/5] Installing core libraries...")
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
from diffusers import LTXImageToVideoPipeline, AudioLDMPipeline
from diffusers.utils import export_to_video
import scipy.io.wavfile

prompt = "Cinematic OLED aesthetic, dark cyberpunk neon interior, dramatic lighting, 8k"
if os.path.exists("prompt.txt"):
    with open("prompt.txt", "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content: prompt = content

print(f"[Driver] Target Prompt: {prompt}")

# ১. বেস ইমেজ তৈরি
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

# ২. আনসেন্সরড LTX-Video রেন্ডারিং
print("[Driver 2/4] Rendering 81 video frames on T4 GPU...")
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
    num_frames=81,
    num_inference_steps=25,
    guidance_scale=3.0
).frames[0]

temp_raw_video = "temp_raw.mp4"
export_to_video(frames, temp_raw_video, fps=8)

# VRAM ও ক্যাশ সম্পূর্ণ মুক্ত করা
del v_pipe
del frames
gc.collect()
torch.cuda.empty_cache()
shutil.rmtree(os.path.expanduser("~/.cache/huggingface"), ignore_errors=True)

# ৩. সিনেমাটিক অডিও সিন্থেসিস (বাগ-মুক্ত ও দ্রুতগতির)
print("[Driver 3/4] Synthesizing atmospheric soundtrack...")
temp_audio = "temp_audio.wav"
audio_rendered = False

try:
    a_pipe = AudioLDMPipeline.from_pretrained(
        "cvssp/audioldm-s-full-v2",
        torch_dtype=torch.float32
    ).to("cuda")

    audio_prompt = f"Cinematic atmospheric sub-bass drone, dark ambient film score, {prompt}"
    audio_data = a_pipe(
        audio_prompt,
        num_inference_steps=15,
        audio_length_in_s=10.2
    ).audios[0]

    audio_data_int16 = (audio_data * 32767).clip(-32768, 32767).astype(np.int16)
    scipy.io.wavfile.write(temp_audio, rate=16000, data=audio_data_int16)
    del a_pipe
    audio_rendered = True
except Exception as a_err:
    print(f"[Driver] Audio model warning ({a_err}), generating synthetic sub-bass atmosphere...")

# যদি কোনো কারণে অডিও লাইব্রেরিতে সমস্যা হয়, তবে সিন্থেসাইজার সক্রিয় হবে (ক্র্যাশ করবে না)
if not audio_rendered or not os.path.exists(temp_audio):
    sr = 16000
    duration = 10.2
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    drone = 0.5 * np.sin(2 * np.pi * 55 * t) + 0.3 * np.sin(2 * np.pi * 110 * t) + 0.2 * np.sin(2 * np.pi * 73.4 * t)
    drone = (drone / np.max(np.abs(drone)) * 26000).astype(np.int16)
    scipy.io.wavfile.write(temp_audio, rate=sr, data=drone)

gc.collect()
torch.cuda.empty_cache()

# ৪. মাস্টার এনকোডিং
print("[Driver 4/4] Encoding master .MP4 container...")
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
print("[Driver] Completed! Video ready for GitHub release publish.")
