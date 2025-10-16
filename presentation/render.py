import os
import sys
import math
from typing import List, Tuple, Optional, Union
import numpy as np

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# MoviePy setup compatible with v2.x (no editor module)
from moviepy import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    concatenate_videoclips,
)
from moviepy.audio.AudioClip import CompositeAudioClip
from moviepy.video.VideoClip import VideoClip

# Audio helpers
try:
    from gtts import gTTS  # type: ignore
except Exception:
    gTTS = None  # type: ignore

import imageio_ffmpeg  # type: ignore

import argparse

# --- Constants ---
WIDTH = 1080
HEIGHT = 1080
FPS = 30
BG_COLOR = (248, 248, 248)  # modern white
PASTEL_COLORS = [
    (255, 228, 225),  # misty rose
    (255, 239, 213),  # papaya whip
    (224, 255, 255),  # light cyan
    (230, 230, 250),  # lavender
    (240, 255, 240),  # honeydew
]

SLIDE_DURATIONS = [10, 10, 10, 10, 10, 10, 15, 15]
TOTAL_DURATION = sum(SLIDE_DURATIONS)

VOICE_LINES = [
    "Hi, I’m Héctor Vega, from Colombia.",
    "I have experience in international telecommunications and technology.",
    "I’m passionate about sharing knowledge and helping others grow.",
    "I believe learning English should be practical, fun, and inspiring.",
    "I’m currently teaching Angie, a law student, to master the A1 level.",
    "In my class, your progress matters most — let’s learn together!",
    "Thank you for watching! Let’s keep growing in English.",
]

SLIDE_TEXTS = [
    ("Welcome Presentation – English with Teacher Héctor Vega",),
    ("Hi, I’m Héctor Vega, from Colombia.",),
    ("I have experience in international telecommunications and technology.",),
    ("I’m passionate about sharing knowledge and helping others grow.",),
    ("I believe learning English should be practical, fun, and inspiring.",),
    ("I’m currently teaching Angie, a law student, to master the A1 level.",),
    ("In my class, your progress matters most — let’s learn together!",),
    ("Thank you for watching! Let’s keep growing in English.", "English with Teacher Héctor Vega"),
]

ASSETS_DIR = os.path.join("presentation", "assets")
VOICE_DIR = os.path.join(ASSETS_DIR, "voice")
IMAGES_DIR = os.path.join(ASSETS_DIR, "images")
MUSIC_DIR = os.path.join(ASSETS_DIR, "music")
OUTPUT_DIR = os.path.join("presentation", "output")

os.makedirs(VOICE_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MUSIC_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def ensure_ffmpeg_available() -> str:
    """Ensure ffmpeg binary is discoverable for moviepy and subprocess calls."""
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path
    return ffmpeg_path


def load_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        # Try to use DejaVuSans (commonly available)
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        # Fallback to a default font
        return ImageFont.load_default()


def create_gradient_background(width: int, height: int) -> Image.Image:
    rng = np.random.default_rng(42)
    color_a = np.array(rng.choice(PASTEL_COLORS))
    color_b = np.array(rng.choice(PASTEL_COLORS))
    while np.all(color_a == color_b):
        color_b = np.array(rng.choice(PASTEL_COLORS))

    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        alpha = y / (height - 1)
        row = (1 - alpha) * color_a + alpha * color_b
        gradient[y, :, :] = row
    return Image.fromarray(gradient)


def blur_background_from_photo(photo_path: str) -> Image.Image:
    try:
        img = Image.open(photo_path).convert("RGB")
        img = img.resize((WIDTH, HEIGHT), Image.LANCZOS)
        img = img.filter(ImageFilter.GaussianBlur(radius=16))
        overlay = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
        return Image.blend(img, overlay, alpha=0.25)
    except Exception:
        return create_gradient_background(WIDTH, HEIGHT)


def draw_text_image(text: str, max_width: int, font_size: int, color=(20, 20, 20)) -> Image.Image:
    font = load_font(font_size)
    # simple wrap
    draw_test = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    words = text.split()
    lines: List[str] = []
    current = ""
    for w in words:
        trial = (current + " " + w).strip()
        w_size = draw_test.textlength(trial, font=font)
        if w_size <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = w
    if current:
        lines.append(current)

    line_height = int(font_size * 1.35)
    text_height = line_height * len(lines)
    canvas = Image.new("RGBA", (max_width, text_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    y = 0
    for line in lines:
        draw.text((0, y), line, font=font, fill=color)
        y += line_height
    return canvas


def image_to_clip(img: Image.Image, duration: float, pos: Union[Tuple[str, str], Tuple[int, int]] = ("center", "center")) -> ImageClip:
    arr = np.array(img)
    clip = ImageClip(arr, duration=duration)
    try:
        return clip.set_position(pos)
    except AttributeError:
        # MoviePy v2 uses with_position
        return clip.with_position(pos)


def apply_fade_in_out(clip: VideoClip, fadein_s: float = 0.0, fadeout_s: float = 0.0) -> VideoClip:
    """Apply fade-in/out compatible with MoviePy v1 and v2.

    For v2 where helpers are missing, emulate by scaling frames' alpha.
    """
    # Try classic fade methods first
    try:
        if fadein_s > 0:
            clip = clip.fadein(fadein_s)
        if fadeout_s > 0:
            clip = clip.fadeout(fadeout_s)
        return clip
    except AttributeError:
        pass

    # Fallback to manual frame scaling (works for RGB clips)
    total = float(clip.duration or 0)
    if total <= 0:
        return clip

    def frame_with_fades(get_frame):
        def new_frame(t):
            f = get_frame(t).astype("float32")
            if fadein_s > 0:
                fi_alpha = min(1.0, max(0.0, t / fadein_s))
            else:
                fi_alpha = 1.0
            if fadeout_s > 0:
                fo_alpha = min(1.0, max(0.0, (total - t) / fadeout_s))
            else:
                fo_alpha = 1.0
            alpha = min(fi_alpha, fo_alpha)
            return np.clip(f * alpha, 0, 255).astype("uint8")
        return new_frame

    return clip.with_updated_frame_function(frame_with_fades(clip.get_frame))


def make_slide_clip(index: int, bg_photo: Optional[str]) -> VideoClip:
    duration = SLIDE_DURATIONS[index]

    # Background
    if index in (0, 7) and bg_photo and os.path.exists(bg_photo):
        bg_img = blur_background_from_photo(bg_photo)
    else:
        bg_img = create_gradient_background(WIDTH, HEIGHT)

    base = image_to_clip(bg_img, duration)

    # Subtle zoom
    def zoom_factor(t: float) -> float:
        return 1.0 + 0.02 * (t / duration)

    try:
        base = base.resize(lambda t: zoom_factor(t))
    except AttributeError:
        # MoviePy v2 method name
        base = base.resized(lambda t: zoom_factor(t))

    # Text blocks
    slide_lines = SLIDE_TEXTS[index]

    elements: List[VideoClip] = [base]

    # Positions per slide
    if index == 0:
        text_width = int(WIDTH * 0.8)
        text_img = draw_text_image(slide_lines[0], text_width, 70)
        text_clip = image_to_clip(text_img, duration).with_position(("center", "center"))
        elements.append(apply_fade_in_out(text_clip, 0.8, 0.0))
    elif index == 1:
        # Side layout
        text_width = int(WIDTH * 0.55)
        text_img = draw_text_image(slide_lines[0], text_width, 60)
        text_clip = image_to_clip(text_img, duration).with_position((int(WIDTH * 0.08), int(HEIGHT * 0.35)))

        # Portrait circle from photo if present
        if bg_photo and os.path.exists(bg_photo):
            try:
                p = Image.open(bg_photo).convert("RGB").resize((420, 420), Image.LANCZOS)
                mask = Image.new("L", p.size, 0)
                md = ImageDraw.Draw(mask)
                md.ellipse((0, 0, p.size[0], p.size[1]), fill=255)
                circle = Image.new("RGBA", p.size)
                circle.paste(p, (0, 0))
                circle.putalpha(mask)
                portrait = image_to_clip(circle, duration, pos=(int(WIDTH * 0.64), int(HEIGHT * 0.28)))
                elements.extend([apply_fade_in_out(text_clip, 0.8, 0.0), apply_fade_in_out(portrait, 0.8, 0.0)])
            except Exception:
                elements.append(apply_fade_in_out(text_clip, 0.8, 0.0))
        else:
            elements.append(apply_fade_in_out(text_clip, 0.8, 0.0))
    elif index == 2:
        text_width = int(WIDTH * 0.8)
        text_img = draw_text_image("🌐 " + slide_lines[0], text_width, 60)
        text_clip = image_to_clip(text_img, duration).with_position(("center", int(HEIGHT * 0.45)))
        elements.append(apply_fade_in_out(text_clip, 0.8, 0.0))
    elif index == 3:
        text_width = int(WIDTH * 0.8)
        text_img = draw_text_image("📚 " + slide_lines[0], text_width, 60)
        text_clip = image_to_clip(text_img, duration).with_position(("center", int(HEIGHT * 0.45)))
        elements.append(apply_fade_in_out(text_clip, 0.8, 0.0))
    elif index == 4:
        text_width = int(WIDTH * 0.85)
        text_img = draw_text_image("✨ " + slide_lines[0], text_width, 60)
        text_clip = image_to_clip(text_img, duration).with_position(("center", int(HEIGHT * 0.45)))
        elements.append(apply_fade_in_out(text_clip, 0.8, 0.0))
    elif index == 5:
        text_width = int(WIDTH * 0.85)
        text_img = draw_text_image("📝 " + slide_lines[0], text_width, 60)
        text_clip = image_to_clip(text_img, duration).with_position(("center", int(HEIGHT * 0.45)))
        elements.append(apply_fade_in_out(text_clip, 0.8, 0.0))
    elif index == 6:
        text_width = int(WIDTH * 0.9)
        text_img = draw_text_image("🚀 " + slide_lines[0], text_width, 64)
        text_clip = image_to_clip(text_img, duration).with_position(("center", int(HEIGHT * 0.45)))
        elements.append(apply_fade_in_out(text_clip, 0.8, 0.0))
    elif index == 7:
        line1 = draw_text_image(slide_lines[0], int(WIDTH * 0.9), 64)
        line2 = draw_text_image(slide_lines[1], int(WIDTH * 0.9), 54)
        c1 = image_to_clip(line1, duration).with_position(("center", int(HEIGHT * 0.40)))
        c2 = image_to_clip(line2, duration).with_position(("center", int(HEIGHT * 0.56)))
        elements.extend([apply_fade_in_out(c1, 0.8, 0.0), apply_fade_in_out(c2, 0.8, 0.0)])

    return CompositeVideoClip(elements, size=(WIDTH, HEIGHT))


def tempo_adjust_ffmpeg(input_path: str, output_path: str, atempo: float) -> None:
    ffmpeg_exe = ensure_ffmpeg_available()
    import subprocess
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        input_path,
        "-filter:a",
        f"atempo={atempo}",
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def ensure_voice_clips(generate: bool) -> List[str]:
    paths: List[str] = []
    for i, line in enumerate(VOICE_LINES, start=1):
        path = os.path.join(VOICE_DIR, f"voice{i}.mp3")
        paths.append(path)

    missing = [p for p in paths if not os.path.exists(p)]
    if not missing:
        return paths

    if not generate:
        return paths  # will be missing; handled later as silence

    if gTTS is None:
        print("gTTS not available, skipping TTS generation.")
        return paths

    print("Generating TTS voice clips with gTTS…")
    for i, line in enumerate(VOICE_LINES, start=1):
        out = os.path.join(VOICE_DIR, f"voice{i}.mp3")
        if os.path.exists(out):
            continue
        try:
            tts = gTTS(text=line, lang="en", tld="com")
            tmp = os.path.join(VOICE_DIR, f"voice{i}.raw.mp3")
            tts.save(tmp)
            # Adjust tempo to ~0.95 (slightly slower)
            try:
                tempo_adjust_ffmpeg(tmp, out, 0.95)
                os.remove(tmp)
            except Exception:
                os.replace(tmp, out)
        except Exception as e:
            print(f"Failed to generate TTS for line {i}: {e}")
    return paths


def build_audio_timeline(voice_paths: List[str], music_path: Optional[str]) -> CompositeAudioClip:
    starts = []
    t = 0
    for d in SLIDE_DURATIONS:
        starts.append(t)
        t += d

    voice_clips: List[AudioFileClip] = []
    for i, p in enumerate(voice_paths[1:], start=2):
        pass

    # Map slide -> voice index: slides 2..8 have voices 1..7
    audio_parts: List = []
    current_time = 0
    for slide_idx, duration in enumerate(SLIDE_DURATIONS):
        if slide_idx == 0:
            current_time += duration
            continue
        voice_index = slide_idx  # 1..7
        path = voice_paths[voice_index]
        if os.path.exists(path):
            try:
                clip = AudioFileClip(path).set_start(sum(SLIDE_DURATIONS[:slide_idx]))
                audio_parts.append(clip.volumex(1.0))
            except Exception:
                pass
        current_time += duration

    # Music on slides 1 and 8 only
    if music_path and os.path.exists(music_path):
        try:
            music_full = AudioFileClip(music_path)
            # Slide 1 music segment
            m1 = (music_full
                  .subclip(0, SLIDE_DURATIONS[0])
                  .audio_fadein(1.5)
                  .audio_fadeout(1.2)
                  .volumex(0.25)
                  .set_start(0))
            audio_parts.append(m1)
            # Slide 8 music segment
            start8 = sum(SLIDE_DURATIONS[:-1])
            # pick a middle segment if music shorter; otherwise loop
            m8 = (music_full
                  .subclip(0, min(SLIDE_DURATIONS[-1], music_full.duration))
                  .audio_fadein(1.5)
                  .audio_fadeout(1.8)
                  .volumex(0.25)
                  .set_start(start8))
            audio_parts.append(m8)
        except Exception:
            pass

    if not audio_parts:
        # No audio parts; return None-like to signal absence
        return None  # type: ignore

    return CompositeAudioClip(audio_parts)


def build_video(bg_photo: Optional[str]) -> VideoClip:
    clips: List[VideoClip] = []
    for idx in range(len(SLIDE_DURATIONS)):
        c = make_slide_clip(idx, bg_photo)
        # Per-slide fade at boundaries for smoothness
        c = apply_fade_in_out(c, 0.5, 0.5)
        clips.append(c)
    final = concatenate_videoclips(clips, method="compose")
    try:
        return final.set_duration(TOTAL_DURATION)
    except AttributeError:
        return final.with_duration(TOTAL_DURATION)


def main() -> None:
    ensure_ffmpeg_available()

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=os.path.join(IMAGES_DIR, "photo.jpg"), help="Background portrait/photo path (optional)")
    parser.add_argument("--music", type=str, default=os.path.join(MUSIC_DIR, "music.mp3"), help="Background music path (optional)")
    parser.add_argument("--no-tts", action="store_true", help="Skip TTS generation and use existing files if present")
    parser.add_argument("--output", type=str, default=os.path.join(OUTPUT_DIR, "welcome_presentation.mp4"))

    args = parser.parse_args()

    # Voice generation or discovery
    voice_paths = [None]  # 0th unused
    voice_paths.extend(ensure_voice_clips(generate=not args.no_tts))

    # Build video
    video = build_video(bg_photo=args.image if os.path.exists(args.image) else None)

    # Build audio
    audio = build_audio_timeline(voice_paths, args.music if os.path.exists(args.music) else None)
    if audio is not None:
        try:
            video = video.set_audio(audio)
        except AttributeError:
            video = video.with_audio(audio)

    # Render
    out = args.output
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"Rendering video to: {out}")
    video.write_videofile(
        out,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        threads=os.cpu_count() or 4,
        bitrate="5000k",
    )


if __name__ == "__main__":
    main()
