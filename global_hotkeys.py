"""System-wide hotkeys (Windows RegisterHotKey) that work without window focus."""

from __future__ import annotations

import queue
import sys
import threading

MOD_ALT, MOD_CONTROL, MOD_NOREPEAT = 0x0001, 0x0002, 0x4000
WM_HOTKEY, WM_QUIT = 0x0312, 0x0012

# Ctrl+Alt avoids browser/editor shortcuts such as Ctrl+R or Ctrl+Shift+V.
# Only these exact combinations are captured; every other key passes through.
HOTKEYS = {
    "reappear": ("Ctrl+Alt+R", MOD_CONTROL | MOD_ALT, ord("R")),
    "vanish": ("Ctrl+Alt+V", MOD_CONTROL | MOD_ALT, ord("V")),
    "capture": ("Ctrl+Alt+B", MOD_CONTROL | MOD_ALT, ord("B")),
}


class GlobalHotkeys:
    """Registers HOTKEYS on a background thread; poll() returns fired actions."""

    def __init__(self, hotkeys: dict = HOTKEYS):
        self.hotkeys = hotkeys
        self.events: queue.Queue[str] = queue.Queue()
        self.thread_id: int | None = None
        self.thread: threading.Thread | None = None
        if sys.platform != "win32":
            print("[Tasti] Scorciatoie globali disponibili solo su Windows.")
            return
        ready = threading.Event()
        self.thread = threading.Thread(target=self._run, args=(ready,), daemon=True)
        self.thread.start()
        ready.wait(2.0)

    def _run(self, ready: threading.Event) -> None:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        self.thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
        actions = list(self.hotkeys)
        registered = []
        for hotkey_id, action in enumerate(actions, start=1):
            label, modifiers, key = self.hotkeys[action]
            if user32.RegisterHotKey(None, hotkey_id, modifiers | MOD_NOREPEAT, key):
                registered.append(label)
            else:
                print(f"[Tasti] {label} è già usato da un altro programma: non disponibile.")
        if registered:
            print(f"[Tasti] Scorciatoie globali attive: {', '.join(registered)}")
        ready.set()
        message = wintypes.MSG()
        try:
            while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
                if message.message == WM_HOTKEY and 1 <= message.wParam <= len(actions):
                    self.events.put(actions[message.wParam - 1])
        finally:
            for hotkey_id in range(1, len(actions) + 1):
                user32.UnregisterHotKey(None, hotkey_id)

    def poll(self) -> list[str]:
        fired = []
        while not self.events.empty():
            fired.append(self.events.get_nowait())
        return fired

    def close(self) -> None:
        if self.thread is not None and self.thread_id is not None:
            import ctypes

            ctypes.windll.user32.PostThreadMessageW(self.thread_id, WM_QUIT, 0, 0)
            self.thread.join(1.0)
