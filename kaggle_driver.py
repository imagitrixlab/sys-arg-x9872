import sys
import subprocess
import gc
import os
import shutil

print("[Driver 0/4] Installing core libraries...")
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

import torch
import numpy as np
import scipy.io.wavfile
from diffusers import LTXPipeline, AudioLDMPipeline
from diffusers.utils import export_to_video

# প্রম্পট রিড করা
prompt = "Cinematic OLED aesthetic, dark cyberpunk neon warrior running, dramatic lighting, 8k"
if os.path.exists("prompt.txt"):
    with open("prompt.txt", "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content: prompt = content

print(f"[Driver] Target Prompt: {prompt}")

# ১. সরাসরি পিওর LTX Text-to-Video (কোনো ইমেজ বা পলিনেশন ওয়াটারমার্ক ছাড়া)
print("[Driver 1/3] Generating pure cinematic video via LTX Text-to-Video...")
pipe = LTXPipeline.from_pretrained(
    "Lightricks/LTX-Video",
    torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()

# গ্লিচ বক্স প্রতিরোধের জন্য টাইলিং সম্পূর্ণ বন্ধ, শুধু সেফ স্লাইসিং সচল
if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
    pipe.vae.enable_slicing()

if hasattr(pipe, "safety_checker"):
    pipe.safety_checker = None

# ৮১ ফ্রেম @ ৮ এফপিএস = ১০.১ সেকেন্ড পিওর সিনেমাটিক রেন্ডার
frames = pipe(
    prompt=prompt,
    negative_prompt="blurry, distorted, low resolution, watermark, logo, text, deformed, scanlines, artifacts",
    width=704,
    height=480,
    num_frames=81,
    num_inference_steps=28,
    guidance_scale=3.5
).frames[0]

temp_raw_video = "temp_raw.mp4"
export_to_video(frames, temp_raw_video, fps=8)

# মেমোরি ও ক্যাশ ক্লিয়ার
del pipe
del frames
gc.collect()
torch.cuda.empty_cache()
shutil.rmtree(os.path.expanduser("~/.cache/huggingface"), ignore_errors=True)

# ২. সাউন্ড সিন্থেসিস
print("[Driver 2/3] Synthesizing atmospheric soundtrack...")
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
    print(f"[Driver] Audio fallback triggered: {a_err}")

if not audio_rendered or not os.path.exists(temp_audio):
    sr = 16000
    duration = 10.2
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    drone = 0.5 * np.sin(2 * np.pi * 55 * t) + 0.3 * np.sin(2 * np.pi * 110 * t) + 0.2 * np.sin(2 * np.pi * 73.4 * t)
    drone = (drone / np.max(np.abs(drone)) * 26000).astype(np.int16)
    scipy.io.wavfile.write(temp_audio, rate=sr, data=drone)

gc.collect()
torch.cuda.empty_cache()

# ৩. মাস্টার এনকোডিং
print("[Driver 3/3] Finalizing master MP4...")
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
print("[Driver] Master video ready!")
