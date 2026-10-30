"""MLXFold in its own persistent process.

Why a process: Mol-1-MLX measured MLX running about a third slower in a process where torch had done
CPU-side work (loading a checkpoint, featurizing), cause unidentified, cured by a process boundary. The
parent here is the torch side (featurizing, ESMC); the worker never imports torch.

Protocol over the worker's stdin/stdout (stray prints go to stderr): each message is an 8-byte
big-endian length and a pickle. Arrays never travel over the pipe: the parent puts inputs in a
shared-memory block, the worker puts outputs in a block it creates and hands over (the parent copies
it out and unlinks it).

    parent -> worker  ("fold", {in_shm, in_layout, feature_keys, options, draw_keys})  |  ("quit", None)
    worker -> parent  ("ready", {pid, torch_imported}) once, then ("ok", {out_shm, out_layout, seconds}) | ("error", traceback)

A worker that dies (its memory guard, the OS, a crash) raises ``WorkerError`` naming the exit code and
the cap; the next call starts a new one. There is never a silent fallback to torch.
"""

from __future__ import annotations

import argparse
import atexit
import dataclasses
import os
import pickle
import struct
import subprocess
import sys
import traceback
from multiprocessing import resource_tracker, shared_memory
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

Layout = List[Tuple[str, Tuple[int, ...], str, int]]  # (key, shape, dtype, byte offset)
_ALIGN = 64


def _send(stream, obj: Any) -> None:
    blob = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack(">Q", len(blob)))
    stream.write(blob)
    stream.flush()


def _recv(stream) -> Any:
    head = stream.read(8)
    if len(head) < 8:
        raise EOFError("peer closed the pipe")
    (n,) = struct.unpack(">Q", head)
    blob = stream.read(n)
    if len(blob) < n:
        raise EOFError("peer closed the pipe mid-message")
    return pickle.loads(blob)


def pack(arrays: Dict[str, np.ndarray]) -> Tuple[shared_memory.SharedMemory, Layout]:
    """Copy ``arrays`` into one new shared-memory block. The caller owns it (close + unlink)."""
    layout: Layout = []
    offset = 0
    arrays = {k: np.ascontiguousarray(v) for k, v in arrays.items()}
    for key, a in arrays.items():
        layout.append((key, tuple(a.shape), a.dtype.str, offset))
        offset += (a.nbytes + _ALIGN - 1) // _ALIGN * _ALIGN
    shm = shared_memory.SharedMemory(create=True, size=max(offset, _ALIGN))
    for (key, shape, dtype, off), a in zip(layout, arrays.values()):
        np.ndarray(shape, dtype=dtype, buffer=shm.buf, offset=off)[...] = a
    return shm, layout


def views(shm: shared_memory.SharedMemory, layout: Layout) -> Dict[str, np.ndarray]:
    return {k: np.ndarray(shape, dtype=dt, buffer=shm.buf, offset=off) for k, shape, dt, off in layout}


def _untrack(shm: shared_memory.SharedMemory) -> None:
    """Python registers every block with this process's resource tracker, which would unlink it (and
    warn) when this process exits; ownership belongs to the other side, so undo that."""
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass


# ----------------------------------------------------------------------------------- worker side
def _serve(weights_dir: str, cap_gib: Optional[float]) -> None:
    import contextlib
    import time

    from .config import FoldOptions
    from .fold import MLXFold
    from .guard import memory_guard
    from .rng import ReplayDraws

    proto_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)  # anything printed from here on goes to stderr
    proto_in = sys.stdin.buffer
    guard = memory_guard(max_gib=cap_gib, cache_gib=2.0) if cap_gib else contextlib.nullcontext()
    with guard:
        fold = MLXFold(weights_dir)
        _send(proto_out, ("ready", {"pid": os.getpid(), "torch_imported": "torch" in sys.modules}))
        while True:
            try:
                kind, req = _recv(proto_in)
            except EOFError:
                return
            if kind == "quit":
                return
            try:
                start = time.perf_counter()
                shm_in = shared_memory.SharedMemory(name=req["in_shm"])
                _untrack(shm_in)
                try:
                    a = views(shm_in, req["in_layout"])
                    features = {k: np.array(a[k]) for k in req["feature_keys"]}
                    lm = np.array(a["__lm__"]) if "__lm__" in a else None
                    draws = ReplayDraws({k: np.array(a[k]) for k in req["draw_keys"]}) if req["draw_keys"] else None
                    del a
                finally:
                    shm_in.close()
                out = fold.fold(features, lm, FoldOptions(**req["options"]), draws=draws)
                shm_out, layout = pack(out)
                _untrack(shm_out)  # the parent unlinks it
                shm_out.close()
                _send(proto_out, ("ok", {"out_shm": shm_out.name, "out_layout": layout, "seconds": time.perf_counter() - start}))
            except Exception:
                _send(proto_out, ("error", traceback.format_exc()))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="MLX fold worker (started by esmfold2_mlx.worker.RemoteFold)")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--cap-gib", type=float, default=None, help="stop this worker if its memory passes this many GiB")
    args = parser.parse_args(argv)
    _serve(args.weights, args.cap_gib)


# ----------------------------------------------------------------------------------- parent side
class WorkerError(RuntimeError):
    pass


class RemoteFold:
    """``MLXFold.fold`` in a persistent child process; numpy in, numpy out."""

    def __init__(self, weights_dir: str, cap_gib: Optional[float] = None) -> None:
        self.weights_dir, self.cap_gib = weights_dir, cap_gib
        self.proc: Optional[subprocess.Popen] = None
        self.ready_info: dict = {}
        self.last_seconds: Optional[float] = None
        self.last_shm_names: List[str] = []

    @property
    def pid(self) -> Optional[int]:
        return None if self.proc is None else self.proc.pid

    def start(self) -> None:
        if self.proc is not None:
            if self.proc.poll() is None:
                return
            # died while idle (killed from outside): say so once; the next call starts a new worker
            raise WorkerError(self._death("while idle"))
        cmd = [sys.executable, "-m", "esmfold2_mlx.worker", "--weights", self.weights_dir]
        if self.cap_gib:
            cmd += ["--cap-gib", str(self.cap_gib)]
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ)
        env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        atexit.register(self.close)
        try:
            kind, info = _recv(self.proc.stdout)
        except EOFError:
            raise WorkerError(self._death("during startup"))
        if kind != "ready":
            raise WorkerError(f"unexpected first message from the MLX worker: {kind!r}")
        self.ready_info = info

    def _death(self, when: str) -> str:
        code = None if self.proc is None else self.proc.wait(timeout=10)
        self.proc = None
        cap = f"; its memory cap was {self.cap_gib} GiB" if self.cap_gib else ""
        return f"MLX worker died {when} (exit code {code}{cap}); see its stderr above"

    def close(self) -> None:
        p, self.proc = self.proc, None
        if p is None or p.poll() is not None:
            return
        try:
            _send(p.stdin, ("quit", None))
            p.wait(timeout=10)
        except Exception:
            p.kill()

    def __call__(self, features: Dict[str, np.ndarray], lm_hidden_states: Optional[np.ndarray], options,
                 torch_draws: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        self.start()
        payload = {k: np.asarray(v) for k, v in features.items() if v is not None}
        if lm_hidden_states is not None:
            payload["__lm__"] = np.asarray(lm_hidden_states, dtype=np.float32)
        draw_keys = sorted(torch_draws) if torch_draws else []
        payload.update({k: np.asarray(torch_draws[k]) for k in draw_keys})
        shm_in, layout = pack(payload)
        self.last_shm_names = [shm_in.name]
        try:
            request = {"in_shm": shm_in.name, "in_layout": layout, "feature_keys": [k for k in features if features[k] is not None],
                       "options": dataclasses.asdict(options), "draw_keys": draw_keys}
            try:
                _send(self.proc.stdin, ("fold", request))
                kind, info = _recv(self.proc.stdout)
            except (EOFError, BrokenPipeError):
                raise WorkerError(self._death("during a fold"))
            if kind == "error":
                raise WorkerError(f"MLX worker raised:\n{info}")
            shm_out = shared_memory.SharedMemory(name=info["out_shm"])
            self.last_shm_names.append(shm_out.name)
            try:
                out_views = views(shm_out, info["out_layout"])
                out = {k: v.copy() for k, v in out_views.items()}
                del out_views  # release the buffer exports, or close() raises BufferError
            finally:
                shm_out.close()
                shm_out.unlink()
            self.last_seconds = info["seconds"]
            return out
        finally:
            shm_in.close()
            shm_in.unlink()


if __name__ == "__main__":
    main()
