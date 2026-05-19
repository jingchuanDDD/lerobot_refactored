"""Keyboard event listener. Supports both local (evdev) and SSH (stdin) connections.

For local keyboards: uses evdev to capture raw key events.
For SSH sessions: uses terminal raw mode + stdin to read escape sequences.

Key mappings (same for both modes):
    Right arrow  → early exit current episode
    Left arrow   → rerecord current episode
    ESC          → stop all recording
"""

from __future__ import annotations

import os
import select
import sys
import termios
import threading
import tty
from typing import Any


def init_keyboard_listener(
    device_paths: list[str] | None = None,
) -> tuple[Any, dict[str, bool]]:
    """Start a background thread listening for keyboard events.

    Auto-detects: tries evdev first, falls back to stdin for SSH/telnet.

    Args:
        device_paths: List of evdev input device paths. Defaults to common keyboard paths.

    Returns:
        (listener, events): Tuple of (listener_thread or None, events_dict).
            events contains keys: "exit_early", "rerecord_episode", "stop_recording"
    """
    events: dict[str, bool] = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
    }

    # Check if we're in an SSH session → must use stdin (SSH keypresses
    # arrive as terminal escape sequences, not evdev events).
    in_ssh = bool(os.environ.get("SSH_CLIENT") or os.environ.get("SSH_TTY"))

    # Try evdev first, but ONLY if not in SSH session
    if not in_ssh:
        listener = _try_evdev_listener(events, device_paths)
        if listener is not None:
            return listener, events

    # Fall back to stdin (SSH session or no evdev devices)
    if sys.stdin.isatty():
        print("Using stdin for keyboard input (SSH/terminal mode).")
        print("  → = exit episode  |  ← = rerecord  |  ESC = stop")
        listener = _start_stdin_listener(events)
        return listener, events

    return None, events


def _try_evdev_listener(
    events: dict[str, bool], device_paths: list[str] | None = None
) -> threading.Thread | None:
    """Try to start an evdev-based listener. Returns thread if successful, None if not."""
    if device_paths is None:
        device_paths = [
            "/dev/input/event2",
            "/dev/input/event3",
            "/dev/input/event4",
            "/dev/input/event5",
            "/dev/input/event6",
            "/dev/input/event7",
        ]

    try:
        import evdev
        from evdev import ecodes
    except ImportError:
        return None

    devices = []
    for path in device_paths:
        try:
            d = evdev.InputDevice(path)
            name = d.name.lower()
            # Only keep actual keyboard devices
            if "keyboard" in name or "at translated" in name:
                devices.append(d)
            else:
                d.close()
        except Exception:
            pass

    if not devices:
        return None

    print(f"Keyboard listener active (evdev): {[d.name for d in devices]}")

    def _evdev_listen():
        from selectors import DefaultSelector, EVENT_READ
        selector = DefaultSelector()
        for dev in devices:
            try:
                selector.register(dev, EVENT_READ)
            except Exception:
                pass

        while True:
            for key, _mask in selector.select():
                device = key.fileobj
                try:
                    for ev in device.read():
                        if ev.type == ecodes.EV_KEY and ev.value == 1:
                            _handle_key(ev.code, events)
                except Exception:
                    pass

    t = threading.Thread(target=_evdev_listen, daemon=True)
    t.start()
    return t


def _start_stdin_listener(events: dict[str, bool]) -> threading.Thread | None:
    """Start a stdin-based listener for SSH/terminal sessions.

    Uses termios raw mode to read individual bytes from stdin without Enter.
    Parses ANSI escape sequences for arrow keys.
    """
    try:
        old_settings = termios.tcgetattr(sys.stdin)
    except termios.error:
        return None

    # Switch to raw input mode (no echo, no line-buffering)
    tty.setraw(sys.stdin.fileno())

    def _stdin_listen():
        buf = b""
        try:
            while True:
                r, _, _ = select.select([sys.stdin], [], [], 0.2)
                if not r:
                    continue
                data = os.read(sys.stdin.fileno(), 32)
                if not data:
                    break

                buf += data
                # Try to parse escape sequences
                while buf:
                    prev_len = len(buf)
                    key, buf = _parse_escape_sequence(buf)
                    if key is not None:
                        _handle_key(key, events)
                    if len(buf) == prev_len:
                        break  # no progress, avoid infinite loop
        except Exception:
            pass
        finally:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except Exception:
                pass

    t = threading.Thread(target=_stdin_listen, daemon=True)
    t.start()
    return t


def _parse_escape_sequence(data: bytes) -> tuple[int | None, bytes]:
    """Parse an ANSI escape sequence from stdin bytes.

    Returns (key_code, remaining_bytes).
    key_code is None if no complete sequence available yet.
    """
    if not data:
        return None, data

    # Single-byte: ESC alone (0x1b) → treat as ESC press
    if data[0] == 0x1b:
        if len(data) == 1:
            return 1, b""  # KEY_ESC
        if data[1:2] == b"[":
            if len(data) >= 3:
                # \x1b[C = right, \x1b[D = left
                if data[2] == ord("C"):
                    return 106, data[3:]  # KEY_RIGHT (linux/input-event-codes.h)
                elif data[2] == ord("D"):
                    return 105, data[3:]  # KEY_LEFT
                else:
                    return None, data[3:]  # unknown, skip
            return None, data  # wait for more
        else:
            # ESC + non-bracket = individual ESC press (stop recording)
            return 1, data[1:]  # KEY_ESC

    # Control-C → treat as ESC
    if data[0] == 0x03:
        return 1, data[1:]

    # Regular keys: skip non-interesting chars
    if data[0] < 0x20:
        return None, data[1:]  # skip control chars
    return None, data[1:]  # skip printable chars


def _handle_key(key_code: int, events: dict[str, bool]):
    """Map key codes to events."""
    # evdev/input key codes:
    #   KEY_RIGHT = 106, KEY_LEFT = 105, KEY_ESC = 1
    if key_code == 106:  # Right arrow
        print("\n→ Episode exit triggered")
        events["exit_early"] = True
    elif key_code == 105:  # Left arrow
        print("\n← Rerecord triggered")
        events["rerecord_episode"] = True
        events["exit_early"] = True
    elif key_code == 1:  # ESC
        print("\nESC Stop triggered")
        events["stop_recording"] = True
        events["exit_early"] = True
