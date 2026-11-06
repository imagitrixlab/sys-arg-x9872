import sys
import subprocess
import gc
import os

# ১. প্রয়োজনীয় ডিপেন্ডেন্সি ইনস্টলেশন নিশ্চিতকরণ
print("[Driver 0/5] Setting up dependencies for CUDA 12.x...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "diffusers>=0.31.0",
    "transformers",
    "accelerate",
    "scipy",
    "sentencepiece",
    "numpy"
], check=True)

import requests
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from diffusers import LTXImageToVideoPipeline, AudioLDM2Pipeline
from diffusers.utils import export_to_video
import scipy.io.wavfile

# ২. ইনপুট প্রম্পট রিড করা
prompt = "Cinematic OLED aesthetic, dark cyberpunk neon interior, dramatic moody light, photorealistic 8k"
if os.path.exists("prompt.txt"):
    with open("prompt.txt", "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content:
            prompt = content

print(f"[Driver] Target Prompt: {prompt}")

# ৩. আল্ট্রা-ডিটেইলড Flux বেস ফ্রেম জেনারেশন (768x512 Widescreen - Divisible by 32)
print("[Driver 1/4] Generating ultra-high resolution base visual frame...")
flux_prompt = f"masterpiece, photorealistic 8k, raw film footage, cinematic lighting, ultra-detailed, {prompt}"
img_url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(flux_prompt)}?model=flux&width=768&height=512&nologo=true"
res = requests.get(img_url, timeout=90)
init_img = Image.open(BytesIO(res.content)).convert("RGB")
init_img.save("base.jpg")

# ৪. আনসেন্সরড LTX-Video রেন্ডারিং (Float16 + VAE Tiling)
print("[Driver 2/4] Initializing LTX-Video with VAE Tiling (Uncensored)...")
v_pipe = LTXImageToVideoPipeline.from_pretrained(
    "Lightricks/LTX-Video",
    torch_dtype=torch.float16
)

# VRAM অপটিমাইজেশন
v_pipe.enable_model_cpu_offload()
if hasattr(v_pipe, "vae") and hasattr(v_pipe.vae, "enable_tiling"):
    v_pipe.vae.enable_tiling()

# আনসেন্সরড: সেফটি ফিল্টার নিষ্ক্রিয়করণ
if hasattr(v_pipe, "safety_checker"):
    v_pipe.safety_checker = None

# ১২.১২৫ সেকেন্ডের মোশন সিন্থেসিস (৯৭ ফ্রেম @ ৮ এফপিএস - LTX কমপ্লায়েন্ট)
frames = v_pipe(
    image=init_img,
    prompt=prompt,
    negative_prompt="blurry, distorted, low resolution, glitch, watermark, deformed, bad anatomy",
    width=768,
    height=512,
    num_frames=97,
    num_inference_steps=32,
    guidance_scale=3.0
).frames[0]

temp_raw_video = "temp_raw.mp4"
export_to_video(frames, temp_raw_video, fps=8)

# মেমোরি সম্পূর্ণ খালি করা (Garbage Collection সহ)
del v_pipe
gc.collect()
torch.cuda.empty_cache()

# ৫. থিয়েট্রিকাল অ্যাম্বিয়েন্ট সাউন্ড সিন্থেসিস (১২.২ সেকেন্ড সিঙ্ক)
print("[Driver 3/4] Generating 12.2s cinematic atmospheric audio...")
a_pipe = AudioLDM2Pipeline.from_pretrained(
    "cvssp/audioldm2-large",
    torch_dtype=torch.float16
).to("cuda")

audio_prompt = f"Cinematic atmospheric drones, deep sub-bass cinematic impact, dark soundtrack, {prompt}"
audio_data = a_pipe(
    audio_prompt,
    num_inference_steps=25,
    audio_length_in_s=12.2
).audios[0]

# স্ট্যান্ডার্ড 16-bit PCM অডিও ফরম্যাটে রূপান্তর
audio_data_int16 = (audio_data * 32767).clip(-32768, 32767).astype(np.int16)
temp_audio = "temp_audio.wav"
scipy.io.wavfile.write(temp_audio, rate=16000, data=audio_data_int16)

del a_pipe
gc.collect()
torch.cuda.empty_cache()

# ৬. FFmpeg হাই-বিটরেট মাস্টার এনকোডিং ও মার্জ
print("[Driver 4/4] Master video rendering via FFmpeg (CRF 18)...")
final_output = "argent_render_output.mp4"
subprocess.run([
    "ffmpeg", "-y",
    "-i", temp_raw_video,
    "-i", temp_audio,
    "-c:v", "libx264",
    "-crf", "18",
    "-preset", "fast",
    "-pix_fmt", "yuv420p",
    "-c:a", "aac",
    "-b:a", "192k",
    "-shortest",
    final_output
], check=True)

if os.path.exists(temp_raw_video): os.remove(temp_raw_video)
if os.path.exists(temp_audio): os.remove(temp_audio)
print("[Driver] Pipeline complete: 12.2s cinematic video generated!")
