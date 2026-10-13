"""Keep the Windows IME off the game window while a match is running.

Shift is the Microsoft Pinyin Chinese/English toggle. Once Chinese mode is on, the
input method eats keydowns and the player cannot move. Menus still need it for the
name, room and address fields, so the lock is only held during play.
"""
from __future__ import annotations

import sys
import uuid
from ctypes import (HRESULT, POINTER, WINFUNCTYPE, Structure, WinDLL, addressof, byref, c_int, c_long, c_short,
                    c_size_t, c_ssize_t, c_ubyte, c_ulong, c_ushort, c_uint, c_void_p, cast, create_unicode_buffer,
                    memmove, sizeof, wstring_at)

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
_GWL_STYLE = -16
_GWL_EXSTYLE = -20
_CLSCTX_INPROC_SERVER = 1
_COINIT_APARTMENTTHREADED = 2
_WM_KEYDOWN = 0x0100
_WM_CHAR = 0x0102
_WM_PASTE = 0x0302
_GCS_RESULTSTR = 0x0800
_VK_CONTROL = 0x11
_VK_V = 0x56
_CF_UNICODETEXT = 13
_WS_OVERLAPPEDWINDOW = 0x00CF0000
_WS_POPUP = 0x80000000
_WS_VISIBLE = 0x10000000
_WS_EX_EDGES = 0x00020301  # DLGMODALFRAME | WINDOWEDGE | CLIENTEDGE | STATICEDGE
_SWP_FRAMECHANGED = 0x0020
_SWP_SHOWWINDOW = 0x0040
_SWP_NOACTIVATE = 0x0010
_SWP_NOZORDER = 0x0004
_GA_ROOT = 2
_MONITOR_DEFAULTTONEAREST = 2


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
    _imm32.ImmCreateContext.argtypes = []
    _imm32.ImmCreateContext.restype = c_void_p
    _imm32.ImmGetCompositionStringW.argtypes = [c_void_p, c_ulong, c_void_p, c_ulong]
    _imm32.ImmGetCompositionStringW.restype = c_long
    _imm32.ImmSetCompositionWindow.argtypes = [c_void_p, c_void_p]
    _imm32.ImmSetCompositionWindow.restype = c_long
    _imm32.ImmSetCandidateWindow.argtypes = [c_void_p, c_void_p]
    _imm32.ImmSetCandidateWindow.restype = c_long

    _user32.DefWindowProcW.argtypes = [c_void_p, c_uint, c_size_t, c_ssize_t]
    _user32.DefWindowProcW.restype = c_ssize_t
    _user32.GetKeyState.argtypes = [c_int]
    _user32.GetKeyState.restype = c_short
    _user32.GetAncestor.argtypes = [c_void_p, c_uint]
    _user32.GetAncestor.restype = c_void_p
    _user32.MonitorFromWindow.argtypes = [c_void_p, c_ulong]
    _user32.MonitorFromWindow.restype = c_void_p
    _user32.GetMonitorInfoW.argtypes = [c_void_p, c_void_p]
    _user32.GetMonitorInfoW.restype = c_long
    _user32.GetWindowRect.argtypes = [c_void_p, c_void_p]
    _user32.GetWindowRect.restype = c_long
    _user32.SetWindowPos.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_uint]
    _user32.SetWindowPos.restype = c_long
    _user32.OpenClipboard.argtypes = [c_void_p]
    _user32.OpenClipboard.restype = c_long
    _user32.CloseClipboard.argtypes = []
    _user32.CloseClipboard.restype = c_long
    _user32.GetClipboardData.argtypes = [c_uint]
    _user32.GetClipboardData.restype = c_void_p
    _user32.GetCursorPos.argtypes = [c_void_p]
    _user32.GetCursorPos.restype = c_long
    _user32.ScreenToClient.argtypes = [c_void_p, c_void_p]
    _user32.ScreenToClient.restype = c_long

    _kernel32 = WinDLL("kernel32", use_last_error=True)
    _kernel32.GlobalLock.argtypes = [c_void_p]
    _kernel32.GlobalLock.restype = c_void_p
    _kernel32.GlobalUnlock.argtypes = [c_void_p]
    _kernel32.GlobalUnlock.restype = c_long

    _ole32.CoInitializeEx.argtypes = [c_void_p, c_ulong]
    _ole32.CoInitializeEx.restype = HRESULT
    _ole32.CoCreateInstance.argtypes = [POINTER(_GUID), c_void_p, c_ulong, POINTER(_GUID), POINTER(c_void_p)]
    _ole32.CoCreateInstance.restype = HRESULT

    _WNDPROC = WINFUNCTYPE(c_ssize_t, c_void_p, c_uint, c_size_t, c_ssize_t)

    class _POINT(Structure):
        _fields_ = [("x", c_long), ("y", c_long)]

    class _RECT(Structure):
        _fields_ = [("left", c_long), ("top", c_long), ("right", c_long), ("bottom", c_long)]

    class _MONITORINFO(Structure):
        _fields_ = [("cbSize", c_ulong), ("rcMonitor", _RECT), ("rcWork", _RECT), ("dwFlags", c_ulong)]

    class _COMPOSITIONFORM(Structure):
        _fields_ = [("dwStyle", c_ulong), ("ptCurrentPos", _POINT), ("rcArea", _RECT)]

    class _CANDIDATEFORM(Structure):
        _fields_ = [("dwIndex", c_ulong), ("dwStyle", c_ulong), ("ptCurrentPos", _POINT), ("rcArea", _RECT)]
else:
    _user32 = _imm32 = _ole32 = _kernel32 = _WNDPROC = None


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
        self.entry = None
        self.maxlen = 24
        self.swallow = 0

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
        # runs before the re-entry guard so a composition result can see the characters
        # DefWindowProc synthesizes while we are still inside this call
        eaten = self._feed(hwnd, msg, wp, lp)
        if eaten is not None:
            return eaten
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
        try:
            hr = _vt(self.mgr, 9, HRESULT, c_void_p, c_void_p, POINTER(c_void_p))(
                self.mgr, hwnd, None, byref(prev))
        except OSError:
            return
        if hr < 0:
            return
        if prev.value:
            if self.prev_doc is None:
                self.prev_doc = prev.value
            else:
                _release(prev.value)
        try:
            _vt(self.mgr, 8, HRESULT, c_void_p)(self.mgr, None)  # SetFocus(NULL)
        except OSError:
            return

    def suspend(self, hwnd: int) -> None:
        self.entry = None
        self.swallow = 0
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
            try:
                _vt(self.mgr, 9, HRESULT, c_void_p, c_void_p, POINTER(c_void_p))(
                    self.mgr, hwnd, doc, byref(dummy))
                _vt(self.mgr, 8, HRESULT, c_void_p)(self.mgr, doc)
            except OSError:
                pass
            if dummy.value:
                _release(dummy.value)
            _release(self.prev_doc)
            self.prev_doc = None
        if self.himc:
            _imm32.ImmAssociateContext(hwnd, self.himc)
            self.himc = None
        self.suspended = False

    def _ensure_context(self, hwnd: int) -> None:
        if context(hwnd):
            return
        if self.himc:
            _imm32.ImmAssociateContext(hwnd, self.himc)
            self.himc = None
            return
        created = _imm32.ImmCreateContext()
        if created:
            _imm32.ImmAssociateContext(hwnd, created)

    def _park_ime(self, hwnd: int) -> None:
        """Open the IME and park its candidate window on the cursor."""
        himc = _imm32.ImmGetContext(hwnd)
        if not himc:
            return
        try:
            pt = _POINT()
            if _user32.GetCursorPos(byref(pt)):
                _user32.ScreenToClient(hwnd, byref(pt))
            form = _COMPOSITIONFORM()
            form.dwStyle = 2  # CFS_POINT
            form.ptCurrentPos = pt
            _imm32.ImmSetCompositionWindow(himc, byref(form))
            cand = _CANDIDATEFORM()
            cand.dwStyle = 2
            cand.ptCurrentPos = pt
            _imm32.ImmSetCandidateWindow(himc, byref(cand))
        finally:
            _imm32.ImmReleaseContext(hwnd, himc)

    def _insert(self, text: str) -> bool:
        entry = self.entry
        if entry is None or not text:
            return False
        cur = entry.get()
        i = max(0, min(int(entry.getCursorPosition()), len(cur)))
        nxt = cur[:i] + text + cur[i:]
        if self.maxlen > 0:
            nxt = nxt[:self.maxlen]
        entry.enterText(nxt)
        entry.setCursorPosition(min(i + len(text), len(nxt)))
        return True

    def _backspace(self) -> None:
        entry = self.entry
        if entry is None:
            return
        cur = entry.get()
        i = max(0, min(int(entry.getCursorPosition()), len(cur)))
        if i <= 0:
            return
        entry.set(cur[:i - 1] + cur[i:])
        entry.setCursorPosition(i - 1)

    def _compose_result(self, hwnd) -> str:
        himc = _imm32.ImmGetContext(hwnd)
        if not himc:
            return ""
        try:
            n = _imm32.ImmGetCompositionStringW(himc, _GCS_RESULTSTR, None, 0)
            if n <= 0:
                return ""
            buf = create_unicode_buffer(n // 2 + 1)
            if _imm32.ImmGetCompositionStringW(himc, _GCS_RESULTSTR, buf, n) <= 0:
                return ""
            return buf.value
        finally:
            _imm32.ImmReleaseContext(hwnd, himc)

    def _clipboard(self) -> str:
        if not _user32.OpenClipboard(None):
            return ""
        try:
            handle = _user32.GetClipboardData(_CF_UNICODETEXT)
            if not handle:
                return ""
            ptr = _kernel32.GlobalLock(handle)
            if not ptr:
                return ""
            try:
                text = wstring_at(ptr)
            finally:
                _kernel32.GlobalUnlock(handle)
        finally:
            _user32.CloseClipboard()
        return "".join(ch for ch in text if ch >= " " and ch != "\x7f")

    def _feed(self, hwnd, msg, wp, lp):
        """Type into the focused menu field. None lets the game window handle the message."""
        if not _WIN or self.entry is None or self.suspended:
            return None
        try:
            msg = int(msg)
            if msg == 0x010F and (int(lp) & _GCS_RESULTSTR):
                text = self._compose_result(hwnd)
                if text:
                    self._insert(text)
                    self.swallow = len(text)
                result = _user32.DefWindowProcW(hwnd, msg, wp, lp)
                self.swallow = 0
                return result
            if msg in (_WM_CHAR, 0x0286):
                if self.swallow:
                    self.swallow -= 1
                    return 0
                ch = int(wp) & 0xFFFF
                if ch == 8:
                    self._backspace()
                    return 0
                if ch < 32 or ch == 0x7F:
                    return None
                return 0 if self._insert(chr(ch)) else None
            if msg == _WM_KEYDOWN and (int(wp) & 0xFF) == _VK_V and _user32.GetKeyState(_VK_CONTROL) < 0:
                pasted = self._clipboard()
                return 0 if pasted and self._insert(pasted) else None
            if msg == _WM_PASTE:
                self._insert(self._clipboard())
                return 0
        except Exception:
            return None
        return None


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


def begin_edit(win, entry, maxlen: int = 24) -> None:
    """Send keystrokes and IME commits into `entry` until `end_edit`."""
    if not _WIN or entry is None:
        return
    hwnd = hwnd_of(win)
    if not hwnd:
        return
    _lock.entry = entry
    _lock.maxlen = int(maxlen) if maxlen else 0
    _lock.swallow = 0
    if _lock.suspended:
        return
    _lock._ensure_context(hwnd)
    _lock._subclass(hwnd)
    _lock._park_ime(hwnd)


def end_edit(win) -> None:
    if not _WIN:
        return
    _lock.entry = None
    _lock.swallow = 0
    if _lock.suspended:
        return
    hwnd = hwnd_of(win)
    if hwnd:
        _lock._unsubclass(hwnd)


def top_hwnd(win) -> int:
    """Top-level window that owns the frame. Falls back to the graphics hwnd."""
    hwnd = hwnd_of(win)
    if not _WIN or not hwnd:
        return 0
    root = _ptr(_user32.GetAncestor(hwnd, _GA_ROOT))
    return root or hwnd


def target_rect(hwnd: int, borderless: bool, windowed_size=(1600, 900)):
    """(x, y, w, h) in the window's own pixel space. (0, 0, 0, 0) if unknown."""
    if not _WIN or not hwnd:
        return 0, 0, 0, 0
    monitor = _user32.MonitorFromWindow(hwnd, _MONITOR_DEFAULTTONEAREST)
    if not monitor:
        return 0, 0, 0, 0
    info = _MONITORINFO()
    info.cbSize = sizeof(_MONITORINFO)
    if not _user32.GetMonitorInfoW(monitor, byref(info)):
        return 0, 0, 0, 0
    if borderless:
        rect = info.rcMonitor
        return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top
    rect = info.rcWork
    avail_w, avail_h = rect.right - rect.left, rect.bottom - rect.top
    w = min(int(windowed_size[0]), avail_w)
    h = min(int(windowed_size[1]), avail_h)
    x = rect.left + max(0, (avail_w - w) // 2)
    y = rect.top + max(0, (avail_h - h) // 2)
    return x, y, w, h


def _window_rect(hwnd: int):
    rect = _RECT()
    if not _user32.GetWindowRect(hwnd, byref(rect)):
        return None
    return rect.left, rect.top, rect.right, rect.bottom


def _drop_border(hwnd: int) -> None:
    try:
        dwm = WinDLL("dwmapi")
    except OSError:
        return
    dwm.DwmSetWindowAttribute.argtypes = [c_void_p, c_uint, c_void_p, c_uint]
    dwm.DwmSetWindowAttribute.restype = c_long
    corner = c_int(1)  # DWMWCP_DONOTROUND
    dwm.DwmSetWindowAttribute(hwnd, 33, byref(corner), sizeof(corner))
    color = c_uint(0xFFFFFFFE)  # DWMWA_COLOR_NONE
    dwm.DwmSetWindowAttribute(hwnd, 34, byref(color), sizeof(color))


_placing: set = set()


def place_window(hwnd: int, borderless: bool, windowed_size=(1600, 900)) -> None:
    """Size `hwnd` to the monitor it is on, or restore a decorated window of `windowed_size`."""
    if not _WIN or not hwnd or hwnd in _placing:
        return
    x, y, w, h = target_rect(hwnd, borderless, windowed_size)
    if w <= 0 or h <= 0:
        return
    style = _ptr(_user32.GetWindowLongPtrW(hwnd, _GWL_STYLE))
    visible = bool(style & _WS_VISIBLE)
    if borderless:
        want = (style & ~_WS_OVERLAPPEDWINDOW) | _WS_POPUP
        ex = _ptr(_user32.GetWindowLongPtrW(hwnd, _GWL_EXSTYLE))
        want_ex = ex & ~_WS_EX_EDGES
    else:
        want = (style & ~_WS_POPUP) | _WS_OVERLAPPEDWINDOW
        ex = want_ex = None
    if visible:
        want |= _WS_VISIBLE
    else:
        want &= ~_WS_VISIBLE
    dirty = style != want or (want_ex is not None and want_ex != ex)
    current = _window_rect(hwnd)
    target = (x, y, x + w, y + h)
    if not dirty and current is not None and all(abs(a - b) <= 2 for a, b in zip(current, target)):
        return
    if style != want:
        _user32.SetWindowLongPtrW(hwnd, _GWL_STYLE, c_void_p(want))
    if want_ex is not None and want_ex != ex:
        _user32.SetWindowLongPtrW(hwnd, _GWL_EXSTYLE, c_void_p(want_ex))
        _drop_border(hwnd)
    flags = _SWP_FRAMECHANGED | _SWP_NOACTIVATE
    if visible:
        flags |= _SWP_SHOWWINDOW
    else:
        flags |= _SWP_NOZORDER
    _placing.add(hwnd)
    try:
        _user32.SetWindowPos(c_void_p(hwnd), c_void_p(0), int(x), int(y), int(w), int(h), flags)
    finally:
        _placing.discard(hwnd)
