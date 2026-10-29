"""Windows Shell WM_DROPFILES integration for Tk; no GUI package dependency."""
import ctypes
from ctypes import wintypes
import os
import queue


class FileDrop:
    def __init__(self, root, callback):
        self.root, self.callback = root, callback
        self.pending = queue.SimpleQueue()
        self.old_proc = None
        if os.name != "nt":
            return
        user, shell = ctypes.windll.user32, ctypes.windll.shell32
        root.update_idletasks()
        user.GetParent.argtypes = [wintypes.HWND]
        user.GetParent.restype = wintypes.HWND
        self.hwnd = user.GetParent(root.winfo_id()) or root.winfo_id()
        self.user, self.shell = user, shell
        setter = user.SetWindowLongPtrW if ctypes.sizeof(ctypes.c_void_p) == 8 else user.SetWindowLongW
        setter.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
        setter.restype = ctypes.c_void_p
        self.setter = setter
        user.CallWindowProcW.argtypes = [ctypes.c_void_p, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        user.CallWindowProcW.restype = ctypes.c_ssize_t
        shell.DragQueryFileW.argtypes = [wintypes.HANDLE, wintypes.UINT, wintypes.LPWSTR, wintypes.UINT]
        shell.DragQueryFileW.restype = wintypes.UINT
        shell.DragFinish.argtypes = [wintypes.HANDLE]
        shell.DragAcceptFiles.argtypes = [wintypes.HWND, wintypes.BOOL]
        proc_type = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

        def procedure(hwnd, message, wparam, lparam):
            if message == 0x233:
                try:
                    count = shell.DragQueryFileW(wparam, 0xFFFFFFFF, None, 0)
                    files = []
                    for index in range(count):
                        length = shell.DragQueryFileW(wparam, index, None, 0)
                        buffer = ctypes.create_unicode_buffer(length + 1)
                        shell.DragQueryFileW(wparam, index, buffer, length + 1)
                        files.append(buffer.value)
                    # Never enter Tcl from this native callback: Tk's mainloop
                    # has released its thread state while dispatching messages.
                    self.pending.put(files)
                finally:
                    shell.DragFinish(wparam)
                return 0
            return user.CallWindowProcW(self.old_proc, hwnd, message, wparam, lparam)
        self.proc = proc_type(procedure)
        self.old_proc = setter(self.hwnd, -4, ctypes.cast(self.proc, ctypes.c_void_p))
        if not self.old_proc:
            raise OSError("Windowsのファイルdropを登録できません")
        shell.DragAcceptFiles(self.hwnd, True)

    def dispatch(self):
        """Deliver native drops from an ordinary Tk event callback."""
        while True:
            try:
                files = self.pending.get_nowait()
            except queue.Empty:
                return
            self.callback(files)

    def close(self):
        if self.old_proc:
            self.shell.DragAcceptFiles(self.hwnd, False)
            self.setter(self.hwnd, -4, self.old_proc)
            self.old_proc = None
