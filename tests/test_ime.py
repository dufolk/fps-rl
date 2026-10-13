"""Windows IME lock: detach while playing, restore for menus."""
import ctypes
import os
import sys
from ctypes import c_int, c_long, c_size_t, c_ssize_t, c_uint, c_ulong, c_void_p, c_wchar_p

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpsrl.render import ime  # noqa: E402


def _window():
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.CreateWindowExW.argtypes = [c_ulong, c_wchar_p, c_wchar_p, c_ulong, c_int, c_int, c_int, c_int,
                                       c_void_p, c_void_p, c_void_p, c_void_p]
    user32.CreateWindowExW.restype = c_void_p
    user32.DestroyWindow.argtypes = [c_void_p]
    user32.DestroyWindow.restype = c_long
    user32.SendMessageW.argtypes = [c_void_p, c_uint, c_size_t, c_ssize_t]
    user32.SendMessageW.restype = c_ssize_t
    hwnd = user32.CreateWindowExW(0, "Static", "ime-test", 0, 0, 0, 0, 0, None, None, None, None)
    if not hwnd:
        raise ctypes.WinError(ctypes.get_last_error())
    return user32, hwnd


def test_playing_detaches_ime_and_menu_restores_it():
    assert ctypes.sizeof(ime._GUID) == 16
    user32, hwnd = _window()
    imm = ime._imm32
    imm.ImmCreateContext.argtypes = []
    imm.ImmCreateContext.restype = c_void_p
    imm.ImmDestroyContext.argtypes = [c_void_p]
    imm.ImmDestroyContext.restype = c_long
    created = imm.ImmCreateContext()
    assert created, "ImmCreateContext failed"
    try:
        imm.ImmAssociateContext(hwnd, created)
        assert ime._ptr(ime.context(hwnd)) == ime._ptr(created)

        ime._lock.suspend(hwnd)
        assert ime._lock.mgr, "text services did not start"
        assert ime.context(hwnd) is None
        assert ime._ptr(ime._user32.GetWindowLongPtrW(hwnd, -4)) == ime._lock.proc_ptr
        assert user32.SendMessageW(hwnd, 0x0281, 1, -1) == 0  # WM_IME_SETCONTEXT eaten
        user32.SendMessageW(hwnd, 0x0100, 0x10, 0)  # WM_KEYDOWN VK_SHIFT still delivered

        ime._lock.suspend(hwnd)  # a later frame must not drop the saved context
        ime._lock.resume(hwnd)
        assert ime.context(hwnd) is not None
        assert ime._ptr(ime.context(hwnd)) == ime._ptr(created)
        assert ime._ptr(ime._user32.GetWindowLongPtrW(hwnd, -4)) != ime._lock.proc_ptr
    finally:
        imm.ImmAssociateContext(hwnd, None)
        user32.DestroyWindow(hwnd)
        imm.ImmDestroyContext(created)


class _Box:
    def __init__(self, text):
        self.text = text
        self.cur = len(text)

    def get(self):
        return self.text

    def getCursorPosition(self):
        return self.cur

    def enterText(self, text):
        self.text = text
        self.cur = len(text)

    def set(self, text):
        self.text = text

    def setCursorPosition(self, i):
        self.cur = i


def test_menu_field_takes_characters():
    box = _Box("玩家")
    lock = ime._lock
    prev = (lock.entry, lock.maxlen, lock.swallow, lock.suspended)
    lock.entry = box
    lock.maxlen = 4
    lock.swallow = 0
    lock.suspended = False
    try:
        assert lock._feed(0, 0x0102, ord("A"), 0) == 0
        assert box.text == "玩家A"
        assert lock._feed(0, 0x0102, ord("中"), 0) == 0
        assert box.text == "玩家A中"
        assert lock._feed(0, 0x0102, ord("B"), 0) == 0
        assert box.text == "玩家A中"  # maxlen 4 drops the extra character
        box.cur = 2
        assert lock._feed(0, 0x0102, 8, 0) == 0  # backspace
        assert box.text == "玩A中"
        assert box.cur == 1
        assert lock._feed(0, 0x0102, 13, 0) is None  # enter still reaches the widget
        lock.suspended = True
        assert lock._feed(0, 0x0102, ord("Z"), 0) is None
        assert box.text == "玩A中"
    finally:
        lock.entry, lock.maxlen, lock.swallow, lock.suspended = prev


def test_borderless_covers_monitor_and_restores():
    user32, hwnd = _window()
    try:
        ime.place_window(hwnd, True)
        style = ime._ptr(ime._user32.GetWindowLongPtrW(hwnd, -16))
        assert style & 0x80000000  # WS_POPUP
        assert not (style & 0x00C00000)  # WS_CAPTION
        got = ime._window_rect(hwnd)
        x, y, w, h = ime.target_rect(hwnd, True)
        assert abs(got[0] - x) <= 2 and abs(got[1] - y) <= 2
        assert abs((got[2] - got[0]) - w) <= 2 and abs((got[3] - got[1]) - h) <= 2
        assert not (style & 0x10000000)  # stayed hidden

        ime.place_window(hwnd, False, (400, 300))
        style = ime._ptr(ime._user32.GetWindowLongPtrW(hwnd, -16))
        assert style & 0x00C00000
        assert not (style & 0x80000000)
        got = ime._window_rect(hwnd)
        assert abs((got[2] - got[0]) - 400) <= 8
        assert abs((got[3] - got[1]) - 300) <= 8
    finally:
        user32.DestroyWindow(hwnd)


if __name__ == "__main__":
    test_playing_detaches_ime_and_menu_restores_it()
    test_menu_field_takes_characters()
    test_borderless_covers_monitor_and_restores()
    print("ok")
