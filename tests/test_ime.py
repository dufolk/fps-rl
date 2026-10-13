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


if __name__ == "__main__":
    test_playing_detaches_ime_and_menu_restores_it()
    print("ok")
