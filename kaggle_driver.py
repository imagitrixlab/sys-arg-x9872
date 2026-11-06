import sys
import subprocess
import gc
import os
import shutil

print("[System 0/5] Initializing 70GB Memory Pool & Fast Engines...")

# 1. Linux Swap Buffer Allocation
try:
    subprocess.run("fallocate -l 8G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile", shell=True, check=False)
except Exception:
    pass

# 2. Dependency Installation
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
    "diffusers>=0.33.0",
    "transformers>=4.48.0",
    "accelerate>=0.34.0",
    "sentencepiece",
    "scipy",
    "numpy",
    "imageio",
    "imageio-ffmpeg"
], check=True)

import torch
import numpy as np
import scipy.io.wavfile
from diffusers import WanPipeline, AudioLDMPipeline
from diffusers.utils import export_to_video

prompt = "Cinematic slow motion, dark cyberpunk neon city warrior, 8k photorealistic, raw film footage"
if os.path.exists("prompt.txt"):
    with open("prompt.txt", "r", encoding="utf-8") as f:
        c = f.read().strip()
        if c: prompt = c

print(f"[Driver] Executing cinematic master pipeline for: {prompt}")

# 3. Wan 2.1 Engine (Optimized for Tesla T4 FP16 + VAE Tiling/Slicing)
print("[Driver 1/4] Loading Wan 2.1 Engine with CPU Offload & VAE Memory Optimization...")
pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()

# Maliklikan ti OOM iti VAE decoding ti Tesla T4
if hasattr(pipe, "vae"):
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

if hasattr(pipe, "safety_checker"):
    pipe.safety_checker = None

# 81 frames @ 16 fps = 5.1s smooth motion nga addaan 20 steps tapno napartak
video_frames = pipe(
    prompt=prompt,
    negative_prompt="blurry, distorted, low quality, static, deformed, artifacts, watermark, cartoon",
    height=480,
    width=832,
    num_frames=81,
    num_inference_steps=20,
    guidance_scale=6.0
).frames[0]

temp_raw_video = "temp_raw_video.mp4"
export_to_video(video_frames, temp_raw_video, fps=16)

del pipe
del video_frames
gc.collect()
torch.cuda.empty_cache()

# 4. Cinematic Audio Synthesis
print("[Driver 2/4] Generating Hollywood-grade atmospheric audio...")
temp_audio = "temp_audio.wav"
audio_done = False

try:
    a_pipe = AudioLDMPipeline.from_pretrained("cvssp/audioldm-s-full-v2", torch_dtype=torch.float32).to("cuda")
    a_data = a_pipe(
        f"Cinematic dark ambient sub bass drone, dramatic movie score, {prompt}",
        num_inference_steps=10,
        audio_length_in_s=5.2
    ).audios[0]
    scipy.io.wavfile.write(temp_audio, rate=16000, data=(a_data * 32767).clip(-32768, 32767).astype(np.int16))
    del a_pipe
    audio_done = True
except Exception as e:
    print(f"[Driver] Primary audio warning: {e}")

if not audio_done or not os.path.exists(temp_audio):
    sr = 16000
    dur = 5.2
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    sub_bass = 0.6 * np.sin(2 * np.pi * 42 * t) + 0.3 * np.sin(2 * np.pi * 84 * t)
    sub_bass = (sub_bass / np.max(np.abs(sub_bass)) * 26000).astype(np.int16)
    scipy.io.wavfile.write(temp_audio, rate=sr, data=sub_bass)

gc.collect()
torch.cuda.empty_cache()

# 5. DaVinci Color Grading & Fast H.264 Web Mastering
print("[Driver 3/4] Applying 35mm film grading & audio mastering...")
final_output = "argent_render_output.mp4"

video_filter = "eq=contrast=1.12:brightness=0.01:saturation=1.15,unsharp=3:3:0.8:3:3:0.0"
audio_filter = "bass=g=6:f=60,volume=1.2"

subprocess.run([
    "ffmpeg", "-y",
    "-i", temp_raw_video,
    "-i", temp_audio,
    "-vf", video_filter,
    "-af", audio_filter,
    "-c:v", "libx264",
    "-crf", "18",
    "-preset", "faster",
    "-pix_fmt", "yuv420p",
    "-c:a", "aac",
    "-b:a", "256k",
    "-movflags", "+faststart",
    "-shortest",
    final_output
], check=True)

if os.path.exists(temp_raw_video): os.remove(temp_raw_video)
if os.path.exists(temp_audio): os.remove(temp_audio)
print("[Driver 4/4] Master video rendered successfully!")
