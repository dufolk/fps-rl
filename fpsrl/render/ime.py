"""Keep the Windows IME off the game window while a match is running.

Shift is the Microsoft Pinyin Chinese/English toggle. Once Chinese mode is on, the
input method eats keydowns and the player cannot move. Menus still need it for the
name, room and address fields, so the lock is only held during play.
"""
from __future__ import annotations

import sys
import uuid
from ctypes import (HRESULT, POINTER, WINFUNCTYPE, Structure, WinDLL, addressof, byref, c_long, c_size_t,
                    c_ssize_t, c_ubyte, c_ulong, c_ushort, c_uint, c_void_p, cast, memmove, sizeof)

_WIN = sys.platform == "win32"

# legacy IMM messages the default window proc would hand to the IME
_EAT = {
    0x0050,  # WM_INPUTLANGCHANGEREQUEST
    0x0051,  # WM_INPUTLANGCHANGE
    0x010D,  # WM_IME_STARTCOMPOSITION
    0x010E,  # WM_IME_ENDCOMPOSITION
    0x010F,  # WM_IME_COMPOSITION
    0x0281,  # WM_IME_SETCONTEXT
    0x0282,  # WM_IME_NOTIFY
    0x0283,  # WM_IME_CONTROL
    0x0284,  # WM_IME_COMPOSITIONFULL
    0x0285,  # WM_IME_SELECT
    0x0286,  # WM_IME_CHAR
    0x0288,  # WM_IME_REQUEST
    0x0290,  # WM_IME_KEYDOWN
    0x0291,  # WM_IME_KEYUP
}
_IME_CMODE_NATIVE = 0x0001
_GWLP_WNDPROC = -4
_CLSCTX_INPROC_SERVER = 1
_COINIT_APARTMENTTHREADED = 2


class _GUID(Structure):
    _fields_ = [("Data1", c_ulong), ("Data2", c_ushort), ("Data3", c_ushort), ("Data4", c_ubyte * 8)]


def _guid(text: str) -> _GUID:
    g = _GUID()
    memmove(addressof(g), uuid.UUID(text).bytes_le, sizeof(_GUID))
    return g


def _ptr(value) -> int:
    if not value:
        return 0
    return int(value) & ((1 << (sizeof(c_void_p) * 8)) - 1)


if _WIN:
    _user32 = WinDLL("user32", use_last_error=True)
    _imm32 = WinDLL("imm32", use_last_error=True)
    _ole32 = WinDLL("ole32", use_last_error=True)

    _user32.GetWindowLongPtrW.argtypes = [c_void_p, c_long]
    _user32.GetWindowLongPtrW.restype = c_void_p
    _user32.SetWindowLongPtrW.argtypes = [c_void_p, c_long, c_void_p]
    _user32.SetWindowLongPtrW.restype = c_void_p
    _user32.CallWindowProcW.argtypes = [c_void_p, c_void_p, c_uint, c_size_t, c_ssize_t]
    _user32.CallWindowProcW.restype = c_ssize_t

    _imm32.ImmGetContext.argtypes = [c_void_p]
    _imm32.ImmGetContext.restype = c_void_p
    _imm32.ImmReleaseContext.argtypes = [c_void_p, c_void_p]
    _imm32.ImmReleaseContext.restype = c_long
    _imm32.ImmAssociateContext.argtypes = [c_void_p, c_void_p]
    _imm32.ImmAssociateContext.restype = c_void_p
    _imm32.ImmSetOpenStatus.argtypes = [c_void_p, c_long]
    _imm32.ImmSetOpenStatus.restype = c_long
    _imm32.ImmGetConversionStatus.argtypes = [c_void_p, POINTER(c_ulong), POINTER(c_ulong)]
    _imm32.ImmGetConversionStatus.restype = c_long
    _imm32.ImmSetConversionStatus.argtypes = [c_void_p, c_ulong, c_ulong]
    _imm32.ImmSetConversionStatus.restype = c_long

    _ole32.CoInitializeEx.argtypes = [c_void_p, c_ulong]
    _ole32.CoInitializeEx.restype = HRESULT
    _ole32.CoCreateInstance.argtypes = [POINTER(_GUID), c_void_p, c_ulong, POINTER(_GUID), POINTER(c_void_p)]
    _ole32.CoCreateInstance.restype = HRESULT

    _WNDPROC = WINFUNCTYPE(c_ssize_t, c_void_p, c_uint, c_size_t, c_ssize_t)
else:
    _user32 = _imm32 = _ole32 = _WNDPROC = None


def _vt(unk, index: int, restype, *argtypes):
    obj = cast(unk, POINTER(c_void_p))
    table = cast(obj[0], POINTER(c_void_p))
    return WINFUNCTYPE(restype, c_void_p, *argtypes)(table[index])


def _release(ptr) -> None:
    if ptr:
        _vt(c_void_p(ptr), 2, c_ulong)(c_void_p(ptr))


class _Lock:
    def __init__(self):
        self.suspended = False
        self.hwnd = 0
        self.himc = None
        self.mgr = None
        self.prev_doc = None
        self.old_proc = None
        self.proc = None
        self.proc_ptr = 0
        self.tsf_failed = False
        self.in_proc = False

    def ensure_tsf(self) -> None:
        if not _WIN or self.mgr or self.tsf_failed:
            return
        hr = _ole32.CoInitializeEx(None, _COINIT_APARTMENTTHREADED)
        if hr < 0 and (hr & 0xFFFFFFFF) != 0x80010106:  # RPC_E_CHANGED_MODE: COM already up
            self.tsf_failed = True
            return
        clsid = _guid("529A9E6B-6587-4F23-AB9E-9C7D683E3C50")
        iid = _guid("AA80E801-2021-11D2-93E0-0060B067B86E")
        mgr = c_void_p()
        if _ole32.CoCreateInstance(byref(clsid), None, _CLSCTX_INPROC_SERVER, byref(iid), byref(mgr)) < 0 or not mgr:
            self.tsf_failed = True
            return
        client = c_ulong()
        if _vt(mgr, 3, HRESULT, POINTER(c_ulong))(mgr, byref(client)) < 0:
            _release(mgr)
            self.tsf_failed = True
            return
        self.mgr = mgr

    def _wndproc(self, hwnd, msg, wp, lp):
        if self.in_proc:
            return _user32.CallWindowProcW(self.old_proc, hwnd, msg, wp, lp) if self.old_proc else 0
        self.in_proc = True
        try:
            if self.suspended and msg in _EAT:
                return 0
            result = _user32.CallWindowProcW(self.old_proc, hwnd, msg, wp, lp) if self.old_proc else 0
            # focus brings the IME back; drop it again before the next key
            if self.suspended and msg in (0x0006, 0x0007):
                _imm32.ImmAssociateContext(hwnd, None)
                self._clear_tsf(_ptr(hwnd))
            return result
        finally:
            self.in_proc = False

    def _subclass(self, hwnd: int) -> None:
        if self.proc is None:
            self.proc = _WNDPROC(self._wndproc)
            self.proc_ptr = _ptr(cast(self.proc, c_void_p).value)
        current = _ptr(_user32.GetWindowLongPtrW(hwnd, _GWLP_WNDPROC))
        old = _ptr(self.old_proc)
        if current and current != self.proc_ptr and current != old:
            self.old_proc = current
        if current != self.proc_ptr:
            _user32.SetWindowLongPtrW(hwnd, _GWLP_WNDPROC, c_void_p(self.proc_ptr))

    def _unsubclass(self, hwnd: int) -> None:
        if self.old_proc and _ptr(_user32.GetWindowLongPtrW(hwnd, _GWLP_WNDPROC)) == self.proc_ptr:
            _user32.SetWindowLongPtrW(hwnd, _GWLP_WNDPROC, c_void_p(_ptr(self.old_proc)))
        self.old_proc = None

    def _force_english(self, hwnd: int) -> None:
        himc = _imm32.ImmGetContext(hwnd)
        if not himc:
            return
        _imm32.ImmSetOpenStatus(himc, 0)
        conv, sent = c_ulong(), c_ulong()
        if _imm32.ImmGetConversionStatus(himc, byref(conv), byref(sent)):
            _imm32.ImmSetConversionStatus(himc, conv.value & ~_IME_CMODE_NATIVE, sent.value)
        _imm32.ImmReleaseContext(hwnd, himc)

    def _clear_tsf(self, hwnd: int) -> None:
        if not self.mgr:
            return
        prev = c_void_p()
        # ITfThreadMgr::AssociateFocus is vtable slot 9 (IUnknown x3 + 6).
        if _vt(self.mgr, 9, HRESULT, c_void_p, c_void_p, POINTER(c_void_p))(
                self.mgr, hwnd, None, byref(prev)) < 0:
            return
        if prev.value:
            if self.prev_doc is None:
                self.prev_doc = prev.value
            else:
                _release(prev.value)
        _vt(self.mgr, 8, HRESULT, c_void_p)(self.mgr, None)  # SetFocus(NULL)

    def suspend(self, hwnd: int) -> None:
        self.ensure_tsf()
        self._force_english(hwnd)
        prev = _imm32.ImmAssociateContext(hwnd, None)
        if prev and self.himc is None:
            self.himc = prev
        self._clear_tsf(hwnd)
        self._subclass(hwnd)
        self.hwnd = hwnd
        self.suspended = True

    def resume(self, hwnd: int) -> None:
        if not self.suspended:
            return
        self._unsubclass(hwnd)
        if self.mgr and self.prev_doc:
            doc = c_void_p(self.prev_doc)
            dummy = c_void_p()
            _vt(self.mgr, 9, HRESULT, c_void_p, c_void_p, POINTER(c_void_p))(
                self.mgr, hwnd, doc, byref(dummy))
            if dummy.value:
                _release(dummy.value)
            _vt(self.mgr, 8, HRESULT, c_void_p)(self.mgr, doc)
            _release(self.prev_doc)
            self.prev_doc = None
        if self.himc:
            _imm32.ImmAssociateContext(hwnd, self.himc)
            self.himc = None
        self.suspended = False


_lock = _Lock()


def hwnd_of(win) -> int:
    if win is None:
        return 0
    handle = win.getWindowHandle()
    if handle is None:
        return 0
    return _ptr(handle.getIntHandle())


def context(hwnd: int):
    """Current IMM context, or None. The caller does not own a reference."""
    if not _WIN or not hwnd:
        return None
    himc = _imm32.ImmGetContext(hwnd)
    if himc:
        _imm32.ImmReleaseContext(hwnd, himc)
    return himc or None


def hold(win, allow: bool) -> None:
    """`allow` keeps the IME available (menus). Otherwise the match window ignores it."""
    if not _WIN:
        return
    hwnd = hwnd_of(win)
    if not hwnd:
        return
    if allow:
        _lock.resume(hwnd)
    else:
        _lock.suspend(hwnd)
        closer = getattr(win, "closeIme", None)
        if closer is not None:
            closer()
