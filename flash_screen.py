import sys
import time

try:
    import tkinter as tk
except Exception:
    tk = None


def main():
    # Duration and speed of flashes
    duration_sec = 1.0
    interval_sec = 0.1  # 100ms per toggle

    if tk is None:
        # If Tkinter is unavailable, just sleep as a no-op fallback
        time.sleep(duration_sec)
        return

    root = tk.Tk()
    # Fullscreen, topmost overlay without borders
    try:
        root.overrideredirect(True)
    except Exception:
        pass
    try:
        root.attributes("-topmost", True)
        root.attributes("-fullscreen", True)
    except Exception:
        pass

    # A simple frame to paint background color
    frame = tk.Frame(root, bg="black")
    frame.pack(fill="both", expand=True)

    colors = ["black", "white"]
    state = {"i": 0}

    def step():
        i = state["i"]
        color = colors[i % 2]
        frame.configure(bg=color)
        root.configure(bg=color)
        state["i"] += 1
        if state["i"] * interval_sec >= duration_sec:
            root.destroy()
        else:
            root.after(int(interval_sec * 1000), step)

    step()
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Do not crash the caller
        pass
