import sys
import subprocess
import gc
import os
import shutil

print("[System 0/4] Initializing Universal Fast Turbo Engine...")

# 1. Dependency installation
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
from diffusers import WanPipeline
from diffusers.utils import export_to_video

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

raw_prompt = "Cinematic slow motion, dark cyberpunk neon city warrior, 8k photorealistic"
if os.path.exists("prompt.txt"):
    with open("prompt.txt", "r", encoding="utf-8") as f:
        c = f.read().strip()
        if c: raw_prompt = c

# 2. Universal cinematic prompt enhancer
enhanced_prompt = (
    f"Cinematic film footage, highly detailed, sharp focus, 8k raw footage, "
    f"natural lighting, smooth motion, authentic environment, {raw_prompt}"
)

print(f"[Driver] Generating scene for: {enhanced_prompt}")

# 3. Wan 2.1 Engine (22 steps = 5-7 min rendering)
print("[Driver 1/3] Loading Wan 2.1 Engine (Tesla T4 Turbo)...")
pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()

# Memory crash protection
if hasattr(pipe, "vae"):
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

# Complete unfilter & safety filter removal
if hasattr(pipe, "safety_checker"):
    pipe.safety_checker = None

# 81 frames @ 16 fps = 5.1s cinematic motion
video_frames = pipe(
    prompt=enhanced_prompt,
    negative_prompt="abstract, dark void, floating blob, distorted, blurry, cartoon, low quality, static, deformed, watermark",
    height=480,
    width=832,
    num_frames=81,
    num_inference_steps=22,
    guidance_scale=6.0
).frames[0]

temp_raw_video = "temp_raw_video.mp4"
export_to_video(video_frames, temp_raw_video, fps=16)

del pipe
del video_frames
gc.collect()
torch.cuda.empty_cache()

# 4. Context-aware dynamic audio synthesizer
print("[Driver 2/3] Synthesizing Context-Matched Dynamic Audio...")
temp_audio = "temp_audio.wav"
sr = 44100
dur = 5.2
total_samples = int(sr * dur)
t = np.linspace(0, dur, total_samples, endpoint=False)
p_lower = raw_prompt.lower()

# Keyword detection
is_sports = any(k in p_lower for k in ["football", "soccer", "player", "ball", "match", "kick", "sports", "খেলা", "ফুটবল"])
is_vehicle = any(k in p_lower for k in ["car", "vehicle", "drive", "racing", "speed", "bike", "highway", "গাড়ি"])

audio_track = np.zeros(total_samples)

if is_sports:
    # Sports track: Stadium crowd + kick impact
    crowd_noise = np.random.normal(0, 0.22, total_samples) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.4 * t)**2)
    kick_times = [0.4, 1.7, 3.1, 4.4]
    kicks = np.zeros(total_samples)
    for kt in kick_times:
        idx = int(kt * sr)
        k_len = min(int(sr * 0.18), total_samples - idx)
        if k_len > 0:
            tk = np.linspace(0, 0.18, k_len, endpoint=False)
            freq = 60 + 120 * np.exp(-tk * 35)
            kicks[idx:idx + k_len] += 0.9 * np.sin(2 * np.pi * np.cumsum(freq) / sr) * np.exp(-tk * 20)
    audio_track = (crowd_noise * 0.45) + (kicks * 0.75)

elif is_vehicle:
    # Vehicle track: Engine roar + 4/4 driving kick beat
    rpm = 75 + 28 * t + 8 * (t ** 1.4)
    engine = np.tanh(0.45 * np.sin(2 * np.pi * np.cumsum(rpm) / sr) * 2.0) * 0.55
    beat_step = int(sr * (60.0 / 128.0))
    beat = np.zeros(total_samples)
    for b in range(0, total_samples, beat_step):
        b_len = min(int(sr * 0.2), total_samples - b)
        if b_len > 0:
            tb = np.linspace(0, 0.2, b_len, endpoint=False)
            beat[b:b + b_len] += 0.8 * np.sin(2 * np.pi * np.cumsum(50 + 90 * np.exp(-tb * 25)) / sr) * np.exp(-tb * 14)
    audio_track = (engine * 0.6) + (beat * 0.7)

else:
    # Default: Deep cinematic sub-drone + rhythmic risers
    sub_drone = 0.5 * np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 100 * t)
    pulse_step = int(sr * 0.5)
    pulse = np.zeros(total_samples)
    for p in range(0, total_samples, pulse_step):
        p_len = min(int(sr * 0.15), total_samples - p)
        if p_len > 0:
            tp = np.linspace(0, 0.15, p_len, endpoint=False)
            pulse[p:p + p_len] += 0.6 * np.sin(2 * np.pi * 65 * tp) * np.exp(-tp * 18)
    audio_track = (sub_drone * 0.5) + (pulse * 0.65)

# Normalization
audio_track = audio_track / (np.max(np.abs(audio_track)) + 1e-6)
final_audio_data = (audio_track * 28000).astype(np.int16)
scipy.io.wavfile.write(temp_audio, rate=sr, data=final_audio_data)

# 5. FFmpeg Web Mastering
print("[Driver 3/3] FFmpeg Final Mastering...")
final_output = "argent_render_output.mp4"

subprocess.run([
    "ffmpeg", "-y",
    "-i", temp_raw_video,
    "-i", temp_audio,
    "-vf", "eq=contrast=1.08:saturation=1.12,unsharp=3:3:0.5:3:3:0.0",
    "-c:v", "libx264",
    "-crf", "18",
    "-preset", "ultrafast",
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
