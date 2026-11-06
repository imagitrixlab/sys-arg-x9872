import sys
import subprocess
import gc
import os
import shutil

print("[System 0/4] Fast Engine Initialization & Setup...")

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

prompt = "Cinematic slow motion, dark cyberpunk neon city warrior, 8k photorealistic, raw film footage"
if os.path.exists("prompt.txt"):
    with open("prompt.txt", "r", encoding="utf-8") as f:
        c = f.read().strip()
        if c: prompt = c

print(f"[Driver] Generating video for: {prompt}")

# Wan 2.1 Ultra-Fast Turbo Configuration
print("[Driver 1/3] Loading Wan 2.1 Engine (FP16 Turbo)...")
pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()

if hasattr(pipe, "vae"):
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

if hasattr(pipe, "safety_checker"):
    pipe.safety_checker = None

# 15 steps @ 16 fps = Super Fast Rendering
video_frames = pipe(
    prompt=prompt,
    negative_prompt="blurry, distorted, low quality, static, deformed, artifacts, watermark",
    height=480,
    width=832,
    num_frames=81,
    num_inference_steps=15,
    guidance_scale=5.5
).frames[0]

temp_raw_video = "temp_raw_video.mp4"
export_to_video(video_frames, temp_raw_video, fps=16)

del pipe
del video_frames
gc.collect()
torch.cuda.empty_cache()

# Instant Hollywood Cinema Audio Synthesizer (0.1 Second)
print("[Driver 2/3] Generating Cinema Dynamic Sub-Bass Audio...")
temp_audio = "temp_audio.wav"
sr = 16000
dur = 5.2
t = np.linspace(0, dur, int(sr * dur), endpoint=False)
sub_bass = 0.6 * np.sin(2 * np.pi * 42 * t) + 0.3 * np.sin(2 * np.pi * 84 * t)
sub_bass = (sub_bass / np.max(np.abs(sub_bass)) * 26000).astype(np.int16)
scipy.io.wavfile.write(temp_audio, rate=sr, data=sub_bass)

# Fast Master Encoding with Hardware Friendly Web MP4
print("[Driver 3/3] FFmpeg Fast Web Mastering...")
final_output = "argent_render_output.mp4"

subprocess.run([
    "ffmpeg", "-y",
    "-i", temp_raw_video,
    "-i", temp_audio,
    "-vf", "eq=contrast=1.10:brightness=0.01:saturation=1.12,unsharp=3:3:0.6:3:3:0.0",
    "-af", "bass=g=5:f=60,volume=1.15",
    "-c:v", "libx264",
    "-crf", "18",
    "-preset", "ultrafast",
    "-pix_fmt", "yuv420p",
    "-c:a", "aac",
    "-b:a", "192k",
    "-movflags", "+faststart",
    "-shortest",
    final_output
], check=True)

if os.path.exists(temp_raw_video): os.remove(temp_raw_video)
if os.path.exists(temp_audio): os.remove(temp_audio)
print("[Driver] Master video rendered successfully!")
