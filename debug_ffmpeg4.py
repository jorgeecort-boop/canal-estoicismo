from src.video import VideoComposer
from config import get_settings
from pathlib import Path
import subprocess

s = get_settings(draft_mode=True)
c = VideoComposer(s)

# Test _create_placeholder_image
test_path = s.paths.temp_dir / 'test_placeholder5.jpg'
c._create_placeholder_image(test_path)

# Test using font name instead of fontfile
text = "Seneca: 'Ensayemos la muerte"

# Escape for FFmpeg
escaped = text.replace("'", r"'\''").replace(":", r"\:").replace("%", r"\%")

# Use font name (Georgia) instead of fontfile path
text_filter = "drawtext=text='" + escaped + "':font='Georgia':fontsize=72:fontcolor=0xF5F0E1:bordercolor=0x1A1A1A:borderw=3:x=(w-text_w)/2:y=h-text_h-120:line_spacing=1.3:enable='between(t,0,2.0)'"

print('Text filter:', text_filter)

filter_complex = "[0:v]" + text_filter + "[v]"

# Run FFmpeg
audio_path = Path('test_audio.mp3')
out_path = Path('test_scene5.mp4')
cmd = [
    'ffmpeg', '-y',
    '-loop', '1', '-i', str(test_path),
    '-i', str(audio_path),
    '-filter_complex', filter_complex,
    '-map', '[v]', '-map', '1:a',
    '-c:v', 'libx264', '-preset', 'medium', '-b:v', '8000k',
    '-c:a', 'aac', '-b:a', '192k',
    '-r', '30', '-t', '2', '-shortest', '-pix_fmt', 'yuv420p',
    str(out_path),
]
print('CMD:', ' '.join(cmd))
result = subprocess.run(cmd, capture_output=True, text=True)
print('Return code:', result.returncode)
if result.stderr:
    print('STDERR:', result.stderr[-3000:])