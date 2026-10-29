"""Simple Tk window backed by the same resumable core as the CLI."""
from dataclasses import asdict
import os
from pathlib import Path
import queue
import sys
import threading
import time
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .application import job_from_spec, new_job, save_session
from .checkpoint import atomic_json, load_json
from .geometry import Rect
from .media import command, probe
from .windows_drop import FileDrop


class App:
    def __init__(self, root, jobs_dir=None):
        self.root = root
        self.directory = Path(jobs_dir) if jobs_dir is not None else Path(
            os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "MosaicRefiner/jobs"
        self.recent = self.directory / "recent.json"
        self.events, self.worker, self.job = queue.Queue(), None, None
        self.spec, self.info, self.photo = None, None, None
        self.preview_token = 0
        self.diagnostic = self.directory.parent / "gui-state.json"
        self.last_diagnostic = 0
        self.selection = None
        root.title("Mosaic Refiner")
        root.geometry("740x600")
        root.minsize(620, 520)
        body = ttk.Frame(root, padding=18)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="動画をドロップして、開始", font=("Yu Gothic UI", 18)).pack(anchor="w")
        ttk.Label(body, text="原本を保持して、別の動画へ保存します。公式H3由来 / 空間2倍拡大 / 既定5秒窓").pack(anchor="w", pady=(4, 12))
        self.path = tk.StringVar(value="動画をここへドロップ、またはファイルを選択")
        ttk.Label(body, textvariable=self.path, wraplength=680).pack(anchor="w")
        self.choose_button = ttk.Button(body, text="動画を選択…", command=self.choose)
        self.choose_button.pack(anchor="w", pady=8)
        self.canvas = tk.Canvas(body, height=280, background="#20252b", highlightthickness=0)
        self.canvas.pack(fill="x", pady=8)
        self.canvas.create_text(340, 140, text="動画を選ぶとプレビューが表示されます", fill="white")
        self.canvas.bind("<ButtonPress-1>", self.begin_roi)
        self.canvas.bind("<B1-Motion>", self.move_roi)
        self.canvas.bind("<ButtonRelease-1>", self.end_roi)
        ttk.Label(body, text="既定はブロック領域の自動候補検出。手動指定する場合はプレビューで囲みます。", wraplength=680).pack(anchor="w")
        self.context = tk.StringVar(value="5")
        self.options = ttk.Frame(body)
        ttk.Label(self.options, text="生成context（秒、実測に応じて選択）").pack(side="left")
        self.context_box = ttk.Combobox(self.options, textvariable=self.context, values=("5", "10", "15"),
                                       width=5, state="readonly")
        self.context_box.pack(side="left", padx=8)
        ttk.Button(self.options, text="出力先を選ぶ…", command=self.choose_output).pack(side="left")
        self.options_button = ttk.Button(body, text="詳細設定", command=self.toggle_options)
        self.options_button.pack(anchor="w", pady=4)
        ttk.Button(body, text="初回セットアップ…", command=self.open_setup).pack(anchor="w")
        buttons = ttk.Frame(body)
        buttons.pack(fill="x", pady=10)
        self.start_button = ttk.Button(buttons, text="開始", command=self.start, state="disabled")
        self.start_button.pack(side="left")
        self.cancel_button = ttk.Button(buttons, text="取消し", command=self.cancel, state="disabled")
        self.cancel_button.pack(side="left", padx=8)
        self.resume_button = ttk.Button(buttons, text="前の処理を再開", command=self.resume,
                                       state="normal" if self.recent.exists() else "disabled")
        self.resume_button.pack(side="left")
        self.open_button = ttk.Button(buttons, text="出力を開く", command=self.open_output, state="disabled")
        self.open_button.pack(side="right")
        ttk.Button(buttons, text="自動検出に戻す", command=self.clear_roi).pack(side="right", padx=8)
        self.bar = ttk.Progressbar(body, maximum=100)
        self.bar.pack(fill="x", pady=4)
        self.status = tk.StringVar(value="動画をドロップ・選択してください。前回の動画は「前の処理を再開」から開けます"
                                  if self.recent.exists() else "動画をドロップ、または「動画を選択」で読み込んでください")
        ttk.Label(body, textvariable=self.status, wraplength=680).pack(anchor="w", pady=6)
        self.drop = FileDrop(root, self.drop_files)
        root.report_callback_exception = self.callback_error
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self.poll)

    def busy(self):
        return self.worker is not None and self.worker.is_alive()

    def callback_error(self, exception, value, tb):
        traceback.print_exception(exception, value, tb)
        self.status.set("操作に失敗しました: " + str(value))
        self.write_diagnostic(force=True)

    def write_diagnostic(self, force=False):
        now = time.monotonic()
        if not force and now - self.last_diagnostic < 1:
            return
        self.last_diagnostic = now
        try:
            atomic_json(self.diagnostic, {"pid": os.getpid(), "updated_at": time.time(),
                        "status": self.status.get(), "video_selected": self.spec is not None,
                        "preview_ready": self.photo is not None, "worker_running": self.busy(),
                        "start_button": str(self.start_button["state"])})
        except OSError:
            traceback.print_exc()

    def open_setup(self):
        from .setup_gui import SetupWindow
        SetupWindow(tk.Toplevel(self.root))

    def toggle_options(self):
        if self.options.winfo_manager():
            self.options.pack_forget()
        else:
            self.options.pack(after=self.options_button, fill="x", pady=4)

    def choose_output(self):
        if not self.spec or self.busy():
            return
        output = filedialog.asksaveasfilename(title="原本とは別の出力先", defaultextension=".mp4",
                                             filetypes=[("MP4", "*.mp4"), ("lossless NUT", "*.nut")])
        if output:
            self.spec["output"] = output

    def choose(self):
        name = filedialog.askopenfilename(title="処理する動画", filetypes=[("動画", "*.mp4 *.mkv *.mov *.nut *.avi *.webm"), ("すべて", "*.*")])
        if name:
            self.select(Path(name))

    def drop_files(self, files):
        if len(files) != 1:
            self.status.set("動画を1本ずつドロップしてください")
        elif not self.busy():
            self.select(Path(files[0]))

    def select(self, source):
        if self.busy():
            return
        self.spec = new_job(source, self.directory)
        from .processing_options import resolve_options
        self.spec['processing_options']=resolve_options().to_dict()
        self.clear_roi()
        self.photo, self.info = None, None
        self.canvas.delete("all")
        self.path.set(str(source))
        self.start_button.configure(state="disabled")
        self.open_button.configure(state="disabled")
        self.status.set("入力とプレビューを確認しています")
        self.preview_token += 1
        token = self.preview_token
        def prepare():
            try:
                info = probe(source, allow_timeline=True)
                scale = min(640 / info.width, 260 / info.height, 1)
                w, h = max(1, int(info.width * scale)), max(1, int(info.height * scale))
                image = command(["ffmpeg", "-v", "error", "-nostdin", "-noautorotate", "-i", str(source),
                                 "-frames:v", "1", "-vf", f"scale={w}:{h}", "-f", "image2pipe", "-vcodec", "ppm", "pipe:1"])
                self.events.put(("preview", (token, info, image)))
            except Exception as exc:
                self.events.put(("preview-error", (token, str(exc))))
        threading.Thread(target=prepare, daemon=True).start()

    def clear_roi(self):
        self.selection = None
        self.canvas.delete("roi")
        if self.spec:
            self.spec["roi"] = None

    def point(self, event):
        return (min(max(event.x, 0), self.photo.width()), min(max(event.y, 0), self.photo.height()))

    def begin_roi(self, event):
        if self.photo is None or self.busy():
            return
        self.anchor = self.point(event)
        self.canvas.delete("roi")
        self.selection = self.canvas.create_rectangle(*self.anchor, *self.anchor, outline="#5dd8bc", width=2, tags="roi")

    def move_roi(self, event):
        if self.selection is not None and not self.busy():
            self.canvas.coords(self.selection, *self.anchor, *self.point(event))

    def end_roi(self, event):
        if self.selection is None or self.busy():
            return
        x0, y0 = self.anchor
        x1, y1 = self.point(event)
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        x, y = int(left * self.info.width / self.photo.width()), int(top * self.info.height / self.photo.height())
        endx = min(self.info.width, int(right * self.info.width / self.photo.width()))
        endy = min(self.info.height, int(bottom * self.info.height / self.photo.height()))
        if endx <= x or endy <= y:
            self.clear_roi()
            return
        self.spec["roi"] = asdict(Rect(x, y, endx - x, endy - y))
        self.status.set("囲んだ領域を全編で処理します")

    def start(self):
        if self.spec is None:
            self.status.set("動画をドロップ・選択してください。前回の動画は「前の処理を再開」から開けます")
            return
        if self.busy():
            return
        self.spec.setdefault("context_seconds", int(self.context.get()))
        save_session(self.spec, self.recent)
        self.job = job_from_spec(self.spec, lambda **event: self.events.put(("progress", event)))
        self.start_button.configure(state="disabled")
        self.choose_button.configure(state="disabled")
        self.resume_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.bar["value"] = 0
        self.status.set("処理を開始しています")
        def process():
            try:
                self.events.put(("finished", self.job.execute()))
            except Exception as exc:
                self.events.put(("error", str(exc)))
        self.worker = threading.Thread(target=process, daemon=False)
        self.worker.start()

    def cancel(self):
        if self.job:
            self.job.cancel()
            self.status.set("取消し要求を受け付けました。H3実行中は現在の窓を保存して停止します")

    def resume(self):
        if self.busy():
            return
        try:
            self.spec = load_json(self.recent)
            self.path.set(self.spec["source"])
            self.start()
        except Exception as exc:
            self.status.set(str(exc))

    def poll(self):
        try:
            self.poll_events()
        except Exception as exc:
            self.callback_error(type(exc), exc, exc.__traceback__)
        finally:
            self.write_diagnostic()
            self.root.after(100, self.poll)

    def poll_events(self):
        self.drop.dispatch()
        while True:
            try:
                kind, data = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "preview":
                token, info, image = data
                if token != self.preview_token:
                    continue
                self.info = info
                self.photo = tk.PhotoImage(data=image, format="PPM")
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
                self.start_button.configure(state="normal")
                self.status.set("開始できます。自動検出、または対象を囲んでください")
            elif kind == "preview-error":
                if data[0] == self.preview_token:
                    self.status.set(data[1])
            elif kind == "progress":
                self.status.set(data["phase"] + (f" / {data['elapsed_seconds']:.0f}秒" if "elapsed_seconds" in data else ""))
                if data.get("total"):
                    percent = 100 * data["frame"] / data["total"]
                    self.bar["value"] = percent
                    self.status.set(self.status.get() + f" / {data['frame']:,}/{data['total']:,}フレーム ({percent:.1f}%)")
            elif kind in ("finished", "error"):
                self.cancel_button.configure(state="disabled")
                self.choose_button.configure(state="normal")
                self.resume_button.configure(state="normal")
                self.start_button.configure(state="normal")
                if kind == "error":
                    self.status.set("停止: " + data)
                elif data["state"] == "complete":
                    self.status.set("処理完了。時刻・音声の検査に合格（画質の採用判断は別途）"
                                    if data["stage"] == "h3-region-video" else "CPU診断完了（AI補正ではありません）")
                    self.bar["value"] = 100
                    self.open_button.configure(state="normal")
                else:
                    self.status.set("取消しました。確定した区間は再開できます")

    def open_output(self):
        if self.spec and Path(self.spec["output"]).is_file():
            try:
                os.startfile(self.spec["output"])
            except OSError:
                self.status.set("関連付けで開けませんでした。出力: " + self.spec["output"])

    def close(self):
        if self.busy():
            self.cancel()
            self.status.set("処理を安全に停止してから閉じます")
            self.root.after(200, self.close)
            return
        self.drop.close()
        self.root.destroy()


def main():
    from .runtime_settings import selection_ready
    if not selection_ready():
        from .setup_gui import main as setup_main
        setup_main()
        return
    import faulthandler
    if sys.stderr is not None:
        faulthandler.enable()
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
