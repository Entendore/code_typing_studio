"""Export configuration, Audio Worker, and Video Exporter."""
from __future__ import annotations
from typing import Optional, List, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future
from PySide6.QtCore import QThread, Signal, Slot, QObject

from .core import (
    EXPORT_FORMATS, ENCODERS, RenderingError, FFmpegError, 
    log, _subprocess_startup_info, temp_directory, THEMES
)
from .text import Tokenizer, TypingProfile, TypingAnimator, KeyboardOverlay
from .rendering import CodeRenderer, MultiPaneRenderer
from .audio import SimpleSoundGen, _pcm_to_wav_bytes
import os, math, threading, subprocess, shutil, numpy as np
from PySide6.QtGui import QImage

@dataclass(slots=True)
class ExportConfig:
    junior_code: str
    senior_code: str
    name: str
    fmt: str
    res: str
    theme: str
    j_font: str
    s_font: str
    font_size_auto: bool
    font_size: int
    chrome: bool
    ln: bool
    cursor_glow: bool
    watermark: bool
    bg_image_path: str
    show_kb: bool
    kb_layout: str
    kb_pos: str
    kb_op: float
    j_wpm: int
    j_pause: float
    s_wpm: int
    s_pause: float
    start_pause: float
    end_pause: float
    lang: str
    pane_layout: str
    pad: int
    gap: int
    target_duration: float = 0.0

class ExportConfigBuilder:
    def __init__(self):
        self._fields: dict = {}

    def junior_code(self, code: str) -> "ExportConfigBuilder": self._fields["junior_code"] = code; return self
    def senior_code(self, code: str) -> "ExportConfigBuilder": self._fields["senior_code"] = code; return self
    def name(self, n: str) -> "ExportConfigBuilder": self._fields["name"] = n; return self
    def fmt(self, f: str) -> "ExportConfigBuilder": self._fields["fmt"] = f; return self
    def res(self, r: str) -> "ExportConfigBuilder": self._fields["res"] = r; return self
    def theme(self, t: str) -> "ExportConfigBuilder": self._fields["theme"] = t; return self
    def j_font(self, f: str) -> "ExportConfigBuilder": self._fields["j_font"] = f; return self
    def s_font(self, f: str) -> "ExportConfigBuilder": self._fields["s_font"] = f; return self
    def font_size_auto(self, v: bool) -> "ExportConfigBuilder": self._fields["font_size_auto"] = v; return self
    def font_size(self, s: int) -> "ExportConfigBuilder": self._fields["font_size"] = s; return self
    def chrome(self, c: bool) -> "ExportConfigBuilder": self._fields["chrome"] = c; return self
    def ln(self, l: bool) -> "ExportConfigBuilder": self._fields["ln"] = l; return self
    def cursor_glow(self, g: bool) -> "ExportConfigBuilder": self._fields["cursor_glow"] = g; return self
    def watermark(self, w: bool) -> "ExportConfigBuilder": self._fields["watermark"] = w; return self
    def bg_image_path(self, p: str) -> "ExportConfigBuilder": self._fields["bg_image_path"] = p; return self
    def show_kb(self, s: bool) -> "ExportConfigBuilder": self._fields["show_kb"] = s; return self
    def kb_layout(self, l: str) -> "ExportConfigBuilder": self._fields["kb_layout"] = l; return self
    def kb_pos(self, p: str) -> "ExportConfigBuilder": self._fields["kb_pos"] = p; return self
    def kb_op(self, o: float) -> "ExportConfigBuilder": self._fields["kb_op"] = o; return self
    def j_wpm(self, w: int) -> "ExportConfigBuilder": self._fields["j_wpm"] = w; return self
    def j_pause(self, p: float) -> "ExportConfigBuilder": self._fields["j_pause"] = p; return self
    def s_wpm(self, w: int) -> "ExportConfigBuilder": self._fields["s_wpm"] = w; return self
    def s_pause(self, p: float) -> "ExportConfigBuilder": self._fields["s_pause"] = p; return self
    def start_pause(self, s: float) -> "ExportConfigBuilder": self._fields["start_pause"] = s; return self
    def end_pause(self, e: float) -> "ExportConfigBuilder": self._fields["end_pause"] = e; return self
    def lang(self, l: str) -> "ExportConfigBuilder": self._fields["lang"] = l; return self
    def pane_layout(self, p: str) -> "ExportConfigBuilder": self._fields["pane_layout"] = p; return self
    def pad(self, p: int) -> "ExportConfigBuilder": self._fields["pad"] = p; return self
    def gap(self, g: int) -> "ExportConfigBuilder": self._fields["gap"] = g; return self
    def target_duration(self, t: float) -> "ExportConfigBuilder": self._fields["target_duration"] = t; return self

    def build(self) -> ExportConfig:
        return ExportConfig(**self._fields)

def _build_panes_from_config(cfg: ExportConfig) -> MultiPaneRenderer:
    fmt_data = EXPORT_FORMATS.get(cfg.fmt, EXPORT_FORMATS["YouTube (16:9)"])
    w, h = fmt_data["resolutions"].get(cfg.res, (1920, 1080))
    theme = THEMES[cfg.theme]
    
    jp = TypingProfile(name="Junior Dev", wpm=cfg.j_wpm, pause_freq=cfg.j_pause,
                       start_pause=cfg.start_pause, end_pause=cfg.end_pause, error_chance=0.04)
    sp = TypingProfile(name="Senior Dev", wpm=cfg.s_wpm, pause_freq=cfg.s_pause,
                       start_pause=cfg.start_pause, end_pause=cfg.end_pause, error_chance=0.01)
                       
    if cfg.target_duration > 0:
        for profile, code in [(jp, cfg.junior_code), (sp, cfg.senior_code)]:
            dummy = TypingProfile(name=profile.name, wpm=100, pause_freq=profile.pause_freq,
                                  start_pause=profile.start_pause, end_pause=profile.end_pause,
                                  auto_indent=profile.auto_indent, error_chance=profile.error_chance)
            dummy_anim = TypingAnimator(code, dummy, seed=123)
            base_dur = dummy_anim.duration()
            if base_dur > 0:
                req_wpm = 100.0 * (base_dur / cfg.target_duration)
                profile.wpm = max(10, min(500, int(req_wpm)))

    panes: List[Tuple[CodeRenderer, TypingAnimator]] = []
    for i, (code, profile, title, accent_key) in enumerate([
        (cfg.junior_code, jp, f"👶 Junior Dev — {cfg.name}", "junior_accent"),
        (cfg.senior_code, sp, f"🧓 Senior Dev — {cfg.name}", "senior_accent"),
    ]):
        match cfg.pane_layout:
            case "Side by Side (50/50)": pw, ph = (w - cfg.gap) // 2, h
            case "Stacked (50/50)": pw, ph = w, (h - cfg.gap) // 2
            case "Junior Big (70/30)":
                pw = int((w - cfg.gap) * 0.7) if i == 0 else (w - cfg.gap) - int((w - cfg.gap) * 0.7)
                ph = h
            case "Senior Big (70/30)":
                pw = int((w - cfg.gap) * 0.3) if i == 0 else (w - cfg.gap) - int((w - cfg.gap) * 0.3)
                ph = h
            case "PIP - Senior Big (Junior Small)":
                pw = int(w * 0.4) if i == 0 else w
                ph = int(h * 0.4) if i == 0 else h
            case "PIP - Junior Big (Senior Small)":
                pw = w if i == 0 else int(w * 0.4)
                ph = h if i == 0 else int(h * 0.4)
            case _: pw, ph = (w - cfg.gap) // 2, h
        language = cfg.lang
        if language == "Auto":
            ext = os.path.splitext(cfg.name)[1].lower()
            ext_map = {".py": "Python", ".js": "JavaScript", ".cpp": "CFamily", ".cs": "CSharp"}
            language = ext_map.get(ext, "Text")
        code_lines = code.count("\n") + 1
        kb_overlay = None
        kb_h = 0
        if cfg.show_kb:
            kb_overlay = KeyboardOverlay(pw, ph, cfg.kb_layout, theme,
                                         (ph - 2 * cfg.pad) // 3, cfg.kb_pos, cfg.kb_op)
            kb_h = kb_overlay.height_needed()
        current_font = cfg.j_font if i == 0 else cfg.s_font
        font_size = (CodeRenderer.auto_font_size(code_lines, pw, ph, cfg.pad, cfg.chrome,
                      cfg.ln, 4, code, current_font, kb_h) if cfg.font_size_auto
                     else max(8, int(cfg.font_size)))
        renderer = CodeRenderer(width=pw, height=ph, theme_name=cfg.theme,
                                font_family=current_font, font_size=font_size,
                                title_text=title, language=language,
                                keyboard_overlay=kb_overlay, padding=cfg.pad,
                                show_window_chrome=cfg.chrome,
                                show_line_numbers=cfg.ln,
                                bg_image_path=cfg.bg_image_path if i == 0 else None,
                                total_code_lines=code_lines,
                                cursor_glow=cfg.cursor_glow,
                                show_watermark=cfg.watermark and i == 0,
                                accent_color=getattr(theme, accent_key, "#7aa2f7"))
        animator = TypingAnimator(code, profile, seed=i * 1000 + 42)
        panes.append((renderer, animator))
    return MultiPaneRenderer(panes=panes, frame_w=w, frame_h=h, num_panes=2,
                             pane_layout=cfg.pane_layout, gap=cfg.gap)

class AudioWorker(QObject):
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, sound_params: Tuple[str, bool, float],
                 timestamps: List[Tuple[float, str]],
                 volume: float, filepath: Optional[str] = None,
                 bgm_path: Optional[str] = None,
                 bgm_volume: float = 0.3, bgm_duck: float = 0.6):
        super().__init__()
        self.sound_params = sound_params
        self.timestamps = timestamps
        self.volume = volume
        self.filepath = filepath
        self.bgm_path = bgm_path
        self.bgm_volume = bgm_volume
        self.bgm_duck = bgm_duck

    @Slot()
    def run(self):
        preset, stereo, reverb_amt = self.sound_params
        try:
            sg = SimpleSoundGen(sr=44100, preset=preset, stereo=stereo, reverb_amt=reverb_amt)
            if self.filepath and self.bgm_path and os.path.exists(self.bgm_path):
                tmp_typing = self.filepath + ".typing.wav"
                sg.generate_track(self.timestamps, tmp_typing, self.volume)
                try:
                    dur_proc = subprocess.run(
                        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                         "-of", "default=noprint_wrappers=1:nokey=1", tmp_typing],
                        capture_output=True, text=True, timeout=10,
                        startupinfo=_subprocess_startup_info())
                    audio_dur = float(dur_proc.stdout.strip())
                    fade_out_str = f"afade=t=out:st={max(0, audio_dur-3):.3f}:d=3,"
                except Exception:
                    fade_out_str = ""
                sc_ratio = max(1.0, 1.0 + self.bgm_duck * 9.0)
                sc_threshold = max(0.005, 0.10 - self.bgm_duck * 0.08)
                sc_makeup = 1.0 + self.bgm_duck * 1.0
                cmd = ["ffmpeg", "-y", "-i", tmp_typing, "-i", self.bgm_path,
                       "-filter_complex",
                       f"[1:a]highpass=f=60,lowpass=f=5500,volume={self.bgm_volume},"
                       f"afade=t=in:st=0:d=2,{fade_out_str}[bgm_eq];"
                       f"[bgm_eq][0:a]sidechaincompress="
                       f"threshold={sc_threshold:.3f}:ratio={sc_ratio:.2f}:"
                       f"attack=0.015:release=0.3:makeup={sc_makeup:.2f}[bgm_duck];"
                       f"[0:a][bgm_duck]amix=inputs=2:duration=first:normalize=0,"
                       f"alimiter=limit=0.95[aout]",
                       "-map", "[aout]", "-b:a", "256k", "-ar", "48000", self.filepath]
                proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                       startupinfo=_subprocess_startup_info())
                _, stderr = proc.communicate(timeout=180)
                if proc.returncode != 0:
                    shutil.move(tmp_typing, self.filepath)
                else:
                    os.remove(tmp_typing)
            elif self.filepath:
                sg.generate_track(self.timestamps, self.filepath, self.volume)
            else:
                pcm = sg.generate_pcm(self.timestamps, self.volume)
                if len(pcm) > 0:
                    self.finished.emit(_pcm_to_wav_bytes(pcm, sg.sr, 2 if stereo else 1))
                else:
                    self.finished.emit(None)
                return
            self.finished.emit(self.filepath if self.filepath else None)
        except Exception as e:
            self.error.emit(str(e))

class VideoExporter(QThread):
    progress = Signal(int)
    status = Signal(str)
    finished_ok = Signal(str)
    error = Signal(str)

    CHUNK_DURATION: float = 300.0

    def __init__(self, export_config: ExportConfig, output: str,
                 fps: int = 30, sound_params: Optional[Tuple[str, bool, float]] = None,
                 volume: float = 0.5, encoder_name: str = "YouTube Pro (H.264/AAC)",
                 bgm_path: Optional[str] = None, bgm_volume: float = 0.3,
                 bgm_duck: float = 0.6, parent=None):
        super().__init__(parent)
        self.setObjectName("VideoExporter")
        self.config = export_config
        self.output = output
        self.fps = fps
        self.sound_params = sound_params
        self.volume = volume
        self.encoder_name = encoder_name
        self.bgm_path = bgm_path
        self.bgm_volume = bgm_volume
        self.bgm_duck = bgm_duck
        self.multi_pane = _build_panes_from_config(self.config)
        self._cancel = threading.Event()
        self.finished.connect(self.deleteLater)

    def cancel(self) -> None:
        self._cancel.set()

    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    def run(self) -> None:
        audio_executor: Optional[ThreadPoolExecutor] = None
        try:
            anim_dur = self.multi_pane.duration()
            total_dur = self.config.target_duration if self.config.target_duration > 0 else anim_dur
            num_chunks = max(1, math.ceil(total_dur / self.CHUNK_DURATION))

            with temp_directory() as tmp_dir:
                audio_executor = ThreadPoolExecutor(max_workers=1) if self.sound_params else None
                audio_future: Optional[Future] = None
                final_audio = os.path.join(tmp_dir, "final_audio.wav")

                if audio_executor:
                    self.status.emit("Starting audio generation...")
                    all_ts = self.multi_pane.char_timestamps_all()
                    if all_ts:
                        audio_future = audio_executor.submit(self._gen_full_audio, all_ts, final_audio)

                video_chunks: List[str] = []
                for c_idx in range(num_chunks):
                    if self._cancel.is_set(): break
                    start_t = c_idx * self.CHUNK_DURATION
                    end_t = min(total_dur, start_t + self.CHUNK_DURATION)
                    start_frame = int(start_t * self.fps)
                    n_chunk = math.ceil(end_t * self.fps) - start_frame
                    self.status.emit(f"Rendering Video Chunk {c_idx + 1}/{num_chunks}...")
                    chunk_vid = os.path.join(tmp_dir, f"vid_{c_idx}.mp4")
                    if not self._render_chunk_video(start_frame, n_chunk, chunk_vid):
                        raise RenderingError(f"Failed to render video for chunk {c_idx + 1}")
                    video_chunks.append(chunk_vid)
                    self.progress.emit(int(((c_idx + 1) / num_chunks) * 80))

                if self._cancel.is_set():
                    self.error.emit("Cancelled")
                    return

                self.status.emit("Stitching video chunks...")
                final_video = os.path.join(tmp_dir, "final_video.mp4")
                self._concat_files(video_chunks, final_video)

                if audio_future:
                    self.status.emit("Waiting for audio generation to finish...")
                    audio_ok = audio_future.result()
                    if audio_ok:
                        self.status.emit("Muxing final video and audio...")
                        self._mux_final(final_video, final_audio, self.output)
                    else:
                        shutil.move(final_video, self.output)
                else:
                    shutil.move(final_video, self.output)

                self.progress.emit(100)
                self.status.emit(f"Done → {self.output}")
                self.finished_ok.emit(self.output)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            if audio_executor:
                audio_executor.shutdown(wait=True)

    def _gen_full_audio(self, ts: List[Tuple[float, str]], filepath: str) -> bool:
        try:
            preset, stereo, reverb_amt = self.sound_params
            sg = SimpleSoundGen(sr=44100, preset=preset, stereo=stereo, reverb_amt=reverb_amt)
            sg.generate_track(ts, filepath, self.volume)
            return os.path.exists(filepath) and os.path.getsize(filepath) > 44
        except Exception:
            return False

    def _concat_files(self, paths: List[str], out_path: str) -> None:
        concat_path = os.path.join(os.path.dirname(out_path), "concat_list.txt")
        with open(concat_path, "w", encoding="utf-8") as f:
            for p in paths:
                safe = p.replace("\\", "/").replace("'", "\\'")
                f.write(f"file '{safe}'\n")
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
               "-i", concat_path, "-c", "copy", out_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                startupinfo=_subprocess_startup_info())
        _, stderr = proc.communicate(timeout=max(120, len(paths) * 60))
        if proc.returncode != 0:
            err_msg = stderr.decode(errors='ignore')[-400:]
            raise FFmpegError(f"Concat failed: {err_msg}")

    def _mux_final(self, vid_path: str, aud_path: str, out_path: str) -> None:
        if self.bgm_path and os.path.exists(self.bgm_path):
            try:
                dur_proc = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", aud_path],
                    capture_output=True, text=True, timeout=15,
                    startupinfo=_subprocess_startup_info())
                audio_dur = float(dur_proc.stdout.strip())
                fade_out_st = max(0.0, audio_dur - 3.0)
                fade_out_str = f"afade=t=out:st={fade_out_st:.3f}:d=3,"
            except Exception:
                fade_out_str = ""

            sc_ratio = max(1.0, 1.0 + self.bgm_duck * 9.0)
            sc_threshold = max(0.005, 0.10 - self.bgm_duck * 0.08)
            sc_makeup = 1.0 + self.bgm_duck * 1.0

            cmd = ["ffmpeg", "-y", "-i", vid_path, "-i", aud_path, "-i", self.bgm_path,
                   "-c:v", "copy",
                   "-filter_complex",
                   f"[2:a]highpass=f=60,lowpass=f=5500,volume={self.bgm_volume},"
                   f"afade=t=in:st=0:d=2,{fade_out_str}"
                   f"aresample=48000[bgm_eq];"
                   f"[bgm_eq][1:a]sidechaincompress="
                   f"threshold={sc_threshold:.3f}:ratio={sc_ratio:.2f}:"
                   f"attack=0.015:release=0.3:makeup={sc_makeup:.2f}[bgm_duck];"
                   f"[1:a][bgm_duck]amix=inputs=2:duration=longest:normalize=0,"
                   f"alimiter=limit=0.95[aout]",
                   "-map", "0:v", "-map", "[aout]",
                   "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
                   "-movflags", "+faststart", out_path]
        else:
            cmd = ["ffmpeg", "-y", "-i", vid_path, "-i", aud_path,
                   "-map", "0:v", "-map", "1:a",
                   "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                   "-ar", "48000", "-movflags", "+faststart", out_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                startupinfo=_subprocess_startup_info())
        _, stderr = proc.communicate(
            timeout=max(120, os.path.getsize(vid_path) // (1024 * 1024) * 2))
        if proc.returncode != 0:
            err_msg = stderr.decode(errors='ignore')[-400:]
            raise FFmpegError(f"Final mux failed: {err_msg}")

    def _render_chunk_video(self, start_frame: int, n_frames: int,
                           out_path: str, allow_fallback: bool = True) -> bool:
        from .core import _format_eta
        w, h, fps = self.multi_pane.frame_w, self.multi_pane.frame_h, self.fps
        cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgra",
               "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0"]
        enc_cfg = ENCODERS.get(self.encoder_name, ENCODERS["YouTube Pro (H.264/AAC)"])
        cmd += enc_cfg["base"]
        if enc_cfg.get("use_dynamic_gop", False):
            gop = int(fps * 2)
            cmd += ["-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0"]
        if enc_cfg.get("use_dynamic_vbv", False):
            pixels = w * h
            maxrate = int((pixels / 2073600.0) * 12000000)
            cmd += ["-maxrate", f"{maxrate}", "-bufsize", f"{maxrate * 2}"]
        if enc_cfg.get("color_tags", False):
            cmd += ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
        cmd += ["-an", out_path]

        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                    startupinfo=_subprocess_startup_info())
        except FileNotFoundError:
            self.error.emit("FFmpeg not found on PATH.")
            return False

        stderr_chunks: List[bytes] = []
        def _drain():
            while True:
                chunk = proc.stderr.read(8192)
                if not chunk: break
                stderr_chunks.append(chunk)
        dt = threading.Thread(target=_drain, daemon=True)
        dt.start()

        from collections import deque
        wq: deque = deque()
        wl = threading.Lock()
        wcv = threading.Condition(wl)
        we: List[str] = []
        wd = [False]

        def _writer():
            try:
                while True:
                    with wl:
                        while not wq and not wd[0]:
                            wcv.wait(timeout=0.5)
                        if not wq and wd[0]: break
                        fd = wq.popleft()
                        if fd is None: break
                    try:
                        proc.stdin.write(fd)
                        proc.stdin.flush()
                    except BrokenPipeError:
                        we.append("BrokenPipe"); break
                    except Exception as ex:
                        we.append(str(ex)); break
            except Exception as ex:
                we.append(str(ex))
            finally:
                from contextlib import suppress
                with suppress(Exception):
                    if proc.stdin:
                        proc.stdin.flush()
                        proc.stdin.close()
                wd[0] = True
                with wl: wcv.notify_all()

        wt = threading.Thread(target=_writer, daemon=True)
        wt.start()

        scratch = QImage(w, h, QImage.Format_RGB32)
        for fi in range(n_frames):
            if self._cancel.is_set() or we: break
            t = (start_frame + fi) / self.fps
            if fi % 100 == 0:
                self.status.emit(f"Rendering frame {fi}/{n_frames} ({_format_eta(t)})...")
            state = self.multi_pane.compute_state(t)
            qimg = self.multi_pane.render_frame(t, scratch, precomputed_state=state)
            ptr = qimg.constBits()
            try: img_bytes = bytes(ptr)
            except TypeError: img_bytes = ptr.tobytes()
            except AttributeError:
                ptr.setsize(qimg.sizeInBytes())
                img_bytes = bytes(ptr)
            with wl:
                while len(wq) >= 6: wcv.wait(timeout=0.1)
                if self._cancel.is_set() or we: break
                wq.append(img_bytes)
                wcv.notify_all()

        if not self._cancel.is_set() and not we:
            with wl:
                wq.append(None)
                wcv.notify_all()

        wt.join(timeout=600)

        if we:
            if allow_fallback and self.encoder_name != "x264 (CPU, Fast)":
                self.encoder_name = "x264 (CPU, Fast)"
                self.status.emit("Hardware encoder failed. Retrying with CPU fallback...")
                return self._render_chunk_video(start_frame, n_frames, out_path, allow_fallback=False)
            return False

        proc.wait(timeout=600)
        dt.join(timeout=5)

        if proc.returncode != 0:
            if allow_fallback and self.encoder_name != "x264 (CPU, Fast)":
                self.encoder_name = "x264 (CPU, Fast)"
                self.status.emit("Hardware encoder failed. Retrying with CPU fallback...")
                return self._render_chunk_video(start_frame, n_frames, out_path, allow_fallback=False)
            return False
        return True
