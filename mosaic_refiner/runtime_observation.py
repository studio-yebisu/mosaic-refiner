"""Verify canonical process identity and observe, without changing the runtime."""
from pathlib import Path
import json
import subprocess
import threading
import time


def identity(connection=None):
    import psutil
    from .runtime_settings import RUNTIME
    connection = (connection or RUNTIME).validate()
    connections = [c for c in psutil.net_connections("tcp")
                   if c.laddr.port == connection.port and c.status == "LISTEN"]
    if len({c.pid for c in connections}) != 1:
        raise ValueError("選択したAPIのlisten processを一意に確認できません")
    process = psutil.Process(connections[0].pid)
    root = connection.root
    cwd=Path(process.cwd()).resolve()
    if cwd != root and not any(
            (Path(arg) if Path(arg).is_absolute() else cwd/arg).resolve() == root/'main.py'
            for arg in process.cmdline()[1:] if Path(arg).name=='main.py'):
        raise ValueError("ComfyUIのmain.py / process cwdが選択rootと異なります")
    parent=process.parent()
    candidates = [Path(process.exe()).resolve()]
    if parent:candidates.append(Path(parent.exe()).resolve())
    if connection.python not in candidates:
        raise ValueError("ComfyUIが選択したPythonから起動されていません")
    return {"pid": process.pid, "root": str(root), "python": str(connection.python), "endpoint": connection.endpoint,
            "arguments": process.cmdline()[1:], "ram_total_bytes": psutil.virtual_memory().total,
            "ram_available_bytes": psutil.virtual_memory().available}


class Observation:
    def __init__(self, token, process_identity):
        import websocket
        self.socket = websocket.create_connection(process_identity['endpoint'].replace('http://', 'ws://')+f"/ws?clientId={token}", timeout=2)
        self.identity = process_identity
        self.stop = threading.Event()
        self.started = time.monotonic()
        self.events, self.samples = [], []
        self.threads = [threading.Thread(target=self.receive, daemon=True), threading.Thread(target=self.sample, daemon=True)]
        for thread in self.threads:
            thread.start()

    def receive(self):
        while not self.stop.is_set():
            try:
                message = self.socket.recv()
                if isinstance(message, str):
                    data = json.loads(message)
                    if data.get("type") in ("executing", "executed", "execution_start", "execution_error", "execution_cached"):
                        self.events.append({"seconds": time.monotonic() - self.started, **data})
            except Exception:
                if self.stop.is_set():
                    break
                # A disconnected websocket can fail immediately. Avoid spinning
                # while history polling still recovers this owned prompt.
                self.stop.wait(.2)

    def sample(self):
        import psutil
        process = psutil.Process(self.identity["pid"])
        while not self.stop.is_set():
            sample = {"seconds": time.monotonic() - self.started}
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                                        capture_output=True, text=True, timeout=5,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                sample["device_memory_mib"] = [[int(x.strip()) for x in line.split(",")]
                                                for line in result.stdout.strip().splitlines()]
                sample["server_rss_bytes"] = process.memory_info().rss
                sample["system_ram_used_bytes"] = psutil.virtual_memory().used
            except Exception as exc:
                sample["observation_error"] = type(exc).__name__
            self.samples.append(sample)
            self.stop.wait(1)

    def finish(self, prompt_id):
        self.stop.set()
        self.socket.close()
        for thread in self.threads:
            thread.join(timeout=6)
        relevant = [e for e in self.events if e["data"].get("prompt_id") == prompt_id]
        starts = [e for e in relevant if e["type"] == "executing"]
        timings = {}
        for first, following in zip(starts, starts[1:]):
            node = first["data"].get("node")
            if node is not None:
                timings[node] = timings.get(node, 0) + following["seconds"] - first["seconds"]
        return {"identity": self.identity, "events": relevant, "node_seconds": timings,
                "samples": self.samples, "allocator_peak_available": False,
                "device_peak_mib_including_other_processes": max(
                    (d[0] for s in self.samples for d in s.get("device_memory_mib", [])), default=None),
                "server_peak_rss_bytes": max((s.get("server_rss_bytes", 0) for s in self.samples), default=None)}
