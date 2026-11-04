"""Server lifecycle for Gufo and reference servers."""

from __future__ import annotations

import ctypes
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_READY_TIMEOUT = 3600.0


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


DROP_CACHES = "sync; echo 3 > /proc/sys/vm/drop_caches"


def _purge_windows_file_cache() -> bool:
    """Empty the Windows file cache: trim working sets, then purge standby.

    Needs an elevated process (SeProfileSingleProcessPrivilege). Purging the
    standby list alone leaves pages that are still in the system cache.
    """
    from ctypes import wintypes

    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    ntdll = ctypes.WinDLL("ntdll")
    kernel32 = ctypes.WinDLL("kernel32")
    advapi.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE

    class Luid(ctypes.Structure):
        _fields_ = [("low", wintypes.DWORD), ("high", wintypes.LONG)]

    class Privileges(ctypes.Structure):
        _fields_ = [("count", wintypes.DWORD), ("luid", Luid), ("attributes", wintypes.DWORD)]

    token = wintypes.HANDLE()
    if not advapi.OpenProcessToken(kernel32.GetCurrentProcess(), 0x28, ctypes.byref(token)):
        return False
    privileges = Privileges(1, Luid(), 2)  # SE_PRIVILEGE_ENABLED
    advapi.LookupPrivilegeValueW(None, "SeProfileSingleProcessPrivilege", ctypes.byref(privileges.luid))
    advapi.AdjustTokenPrivileges(token, False, ctypes.byref(privileges), 0, None, None)
    if ctypes.get_last_error() != 0:  # ERROR_NOT_ALL_ASSIGNED: not elevated
        return False
    for command in (2, 4):  # MemoryEmptyWorkingSets, MemoryPurgeStandbyList
        value = ctypes.c_int(command)
        if ntdll.NtSetSystemInformation(80, ctypes.byref(value), ctypes.sizeof(value)) != 0:
            return False
    return True


def drop_file_cache(command: str | None = None) -> None:
    """Drop the page cache so the next launch measures a cold file cache."""
    if command is None and sys.platform == "win32":
        if _purge_windows_file_cache():
            return
        raise SystemExit(
            "cold-file-cache loading on Windows needs an elevated shell to purge "
            "the standby list; pass --drop-caches with a command that does it, "
            "or skip the loading table"
        )
    candidates = [command] if command else [
        DROP_CACHES,
        f"sudo -n sh -c '{DROP_CACHES}'",
        f"doas -n sh -c '{DROP_CACHES}'",
    ]
    for candidate in candidates:
        completed = subprocess.run(candidate, shell=True, capture_output=True, text=True)
        if completed.returncode == 0:
            return
    raise SystemExit(
        "cold-file-cache loading needs privileges to run "
        f"`{DROP_CACHES}`; pass --drop-caches with a command that does it "
        "(for example `doas sh -c '...'`), or skip the loading table"
    )


class HipMemory:
    """Device-global used memory from `hipMemGetInfo`, the counter Gufo's loader logs.

    An external process sees every process's HIP allocations on this driver, so
    both servers are measured the same way; `rocm-smi` VRAM+GTT does not count
    Gufo's weight mapping on unified memory.
    """

    def __init__(self, library: Path):
        self.library = library
        self._lib = ctypes.CDLL(str(library))

    @classmethod
    def for_binary(cls, gufo_binary: Path) -> "HipMemory | None":
        """Resolve libamdhip64 the way the Gufo binary links it."""
        executable = gufo_binary.resolve()
        if sys.platform == "win32":
            # Under WDDM, hipMemGetInfo counts only the calling process's
            # allocations: beside a server holding 35 GB it reports 0.15 GiB
            # used. An outside sampler cannot measure the server there.
            return None
        # Nix's makeWrapper keeps the production ELF beside its launch script.
        # ldd cannot inspect the script that selects Gufo's patched runtime.
        for candidate in (executable, executable.with_name(f".{executable.name}-wrapped")):
            try:
                with candidate.open("rb") as binary:
                    if binary.read(4) == b"\x7fELF":
                        executable = candidate
                        break
            except OSError:
                pass
        else:
            return None
        completed = subprocess.run(["ldd", str(executable)], capture_output=True, text=True)
        for line in completed.stdout.splitlines():
            if "libamdhip64" in line and "=>" in line:
                path = Path(line.split("=>", 1)[1].split()[0])
                if path.exists():
                    return cls(path)
        return None

    def used_gib(self) -> float | None:
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        if self._lib.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total)) != 0:
            return None
        return (total.value - free.value) / (1024 ** 3)


class Server:
    """A benchmark server process bound to a private loopback port."""

    def __init__(self, command: list[str], readiness: str, log_path: Path, env: dict[str, str] | None = None):
        self.command = command
        self.readiness = readiness
        self.log_path = log_path
        self.env = env
        self.port = free_port()
        self.process: subprocess.Popen[bytes] | None = None
        self.ready_seconds: float | None = None
        self.failure_exit_code: int | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout: float = DEFAULT_READY_TIMEOUT) -> "Server":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("ab")
        env = dict(os.environ)
        if self.env:
            env.update(self.env)
        started = time.perf_counter()
        # Own session (POSIX) or process group (Windows), so the terminal's
        # interrupt never reaches the server and stop() can signal it alone.
        group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32"
                 else {"start_new_session": True})
        self.process = subprocess.Popen(
            self.command, stdout=log, stderr=subprocess.STDOUT, env=env, **group,
        )
        try:
            deadline = started + timeout
            url = self.base_url + self.readiness
            while time.perf_counter() < deadline:
                if self.process.poll() is not None:
                    self.failure_exit_code = self.process.returncode
                    raise RuntimeError(
                        f"server exited with {self.process.returncode} before readiness; see {self.log_path}"
                    )
                try:
                    with urllib.request.urlopen(url, timeout=5) as response:
                        if response.status == 200:
                            self.ready_seconds = time.perf_counter() - started
                            return self
                except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, TimeoutError):
                    pass
                time.sleep(0.05)
            self.stop()
            raise RuntimeError(f"server not ready within {timeout:.0f} s; see {self.log_path}")
        except BaseException:
            # Ctrl-C included: the server runs in its own session, so the
            # terminal's SIGINT never reaches it, and __exit__ never runs when
            # __enter__ raises.
            self.stop()
            raise

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        if sys.platform == "win32":
            # CTRL_BREAK reaches the new process group only; Gufo and
            # llama-server shut down gracefully on it.
            self.process.send_signal(signal.CTRL_BREAK_EVENT)
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            return
        os.killpg(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()

    def __enter__(self) -> "Server":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        if exc and exc[0] is not None and self.process is not None:
            # Preserve the actual failure state before our own SIGTERM.
            # A benchmark validation error is not a server crash.
            self.failure_exit_code = self.process.poll()
        self.stop()


def wait_process_exit(server: Server) -> None:
    server.stop()
    # Give the driver time to release device memory before the next launch.
    time.sleep(2.0)
