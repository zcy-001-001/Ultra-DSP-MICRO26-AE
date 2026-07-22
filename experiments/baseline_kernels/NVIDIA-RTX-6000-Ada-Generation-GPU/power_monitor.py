"""
GPU power monitor using NVML (pynvml).

For very short GEMV kernels, Python threads can be starved by the tight launch
loop and miss too many samples. This monitor uses a separate process so NVML
sampling keeps running independently of the benchmark loop.
"""

import multiprocessing as mp
import time


def _sample_worker(gpu_index: int, sample_interval_ms: float, conn):
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    interval = sample_interval_ms / 1000.0
    samples: list[float] = []

    try:
        while True:
            if conn.poll():
                message = conn.recv()
                if message == "stop":
                    break
            try:
                mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                samples.append(mw / 1000.0)
            except pynvml.NVMLError:
                pass
            time.sleep(interval)
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        conn.send(samples)
        conn.close()


class PowerMonitor:
    """Sample GPU power draw via NVML from a dedicated helper process."""

    def __init__(self, gpu_index: int = 0, sample_interval_ms: float = 5.0):
        self._gpu_index = gpu_index
        self._sample_interval_ms = sample_interval_ms
        self._process: mp.Process | None = None
        self._parent_conn = None

    def start(self):
        parent_conn, child_conn = mp.Pipe()
        self._parent_conn = parent_conn
        self._process = mp.Process(
            target=_sample_worker,
            args=(self._gpu_index, self._sample_interval_ms, child_conn),
            daemon=True,
        )
        self._process.start()

    def stop(self) -> dict:
        if self._process is None or self._parent_conn is None:
            return {"avg_w": 0.0, "min_w": 0.0, "max_w": 0.0, "samples": 0}

        self._parent_conn.send("stop")
        samples = self._parent_conn.recv()
        self._process.join(timeout=2.0)

        self._process = None
        self._parent_conn.close()
        self._parent_conn = None

        return self._stats(samples)

    @staticmethod
    def _stats(samples: list[float]) -> dict:
        if not samples:
            return {"avg_w": 0.0, "min_w": 0.0, "max_w": 0.0, "samples": 0}

        avg = sum(samples) / len(samples)
        return {
            "avg_w": round(avg, 2),
            "min_w": round(min(samples), 2),
            "max_w": round(max(samples), 2),
            "samples": len(samples),
        }
