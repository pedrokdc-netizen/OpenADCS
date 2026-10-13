from dataclasses import dataclass
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading

from .environment import EnvironmentSample
from .scenario import Readout, Scenario, decode_sample, sample_readout


@dataclass(frozen=True)
class SampleEvent:
    sample: EnvironmentSample
    readout: Readout


@dataclass(frozen=True)
class FinishedEvent:
    status: str
    diagnostics: str


class RunWorker:
    """Drain a fresh CLI subprocess without touching Tk from the reader thread."""

    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.events: queue.Queue[SampleEvent | FinishedEvent] = queue.Queue(maxsize=128)
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def cancel(self) -> None:
        self._cancel.set()
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()

    def _emit(self, event: SampleEvent) -> bool:
        while not self._cancel.is_set():
            try:
                self.events.put(event, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        status = "failed"
        diagnostics = ""
        process = None
        try:
            with tempfile.TemporaryDirectory(prefix="openadcs-") as directory:
                folder = Path(directory)
                tle_path = folder / "input.tle"
                tle_path.write_text("\n".join(self.scenario.tle) + "\n", encoding="ascii")
                with (folder / "diagnostics.log").open("w", encoding="utf-8") as stderr:
                    env = dict(os.environ, PYTHONIOENCODING="utf-8")
                    python = Path(sys.executable)
                    if python.name.lower() == "pythonw.exe":
                        python = python.with_name("python.exe")
                    process = subprocess.Popen(
                        [
                            str(python), "-u", "-m", "openadcs_sim.cli",
                            "--tle", str(tle_path),
                            "--kernels", str(self.scenario.kernel_directory.resolve()),
                            "--start", self.scenario.start_utc.isoformat(),
                            "--samples", str(self.scenario.sample_count),
                            "--step-seconds", str(self.scenario.step_seconds),
                        ],
                        stdout=subprocess.PIPE, stderr=stderr, text=True, encoding="utf-8",
                        env=env,
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                    )
                    with self._lock:
                        self._process = process
                        if self._cancel.is_set() and process.poll() is None:
                            process.terminate()
                    if process.stdout is None:
                        raise RuntimeError("Simulation worker has no output stream.")
                    received = 0
                    with process.stdout:
                        for line in process.stdout:
                            if self._cancel.is_set():
                                break
                            sample = decode_sample(line)
                            if not self._emit(SampleEvent(sample, sample_readout(sample))):
                                break
                            received += 1
                    if self._cancel.is_set():
                        if process.poll() is None:
                            process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        status = "cancelled"
                    else:
                        code = process.wait()
                        status = "completed" if code == 0 and received == self.scenario.sample_count else "failed"
                        if status == "failed":
                            diagnostics = (
                                f"Worker exited with code {code}; received {received} of "
                                f"{self.scenario.sample_count} expected samples.\n"
                            )
                diagnostics += (folder / "diagnostics.log").read_text(encoding="utf-8")
        except (OSError, ValueError, TypeError, OverflowError, RuntimeError) as error:
            diagnostics = f"{type(error).__name__}: {error}"
        finally:
            with self._lock:
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait()
                self._process = None
            self.events.put(FinishedEvent(status, diagnostics))
