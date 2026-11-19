"""Textul final rezultat din apăsări (aplică backspace, enter, tab).

E textul „văzut” de tastatură, nu cel afișat de aplicație: corecturile
automate (diacritice, majuscule după punct, completări) apar doar pe ecran
și nu produc apăsări, deci nu sunt aici.
"""
import csv
from pathlib import Path

# Taste cu efect pe text pe care etichetele nu-l pot reproduce: mutarea
# cursorului și ștergerea de cuvânt/linie cu Option/Cmd + Backspace.
UNMODELED_KEYS = {"left", "right", "up", "down", "delete", "home", "end",
                  "page_up", "page_down", "cmd", "cmd_r", "alt", "alt_r"}

def final_chars(events):
    """events: [(timp, tastă)] → ([(timp, caracter)] rămase, nr. taste nemodelate)."""
    out = []
    unmodeled = 0
    for t, k in events:
        if k == "backspace":
            if out:
                out.pop()
        elif k == "space":
            out.append((t, " "))
        elif k == "enter":
            out.append((t, "\n"))
        elif k == "tab":
            out.append((t, "\t"))
        elif len(k) == 1:
            out.append((t, k))
        elif k in UNMODELED_KEYS:
            unmodeled += 1
    return out, unmodeled

def final_lines(chars):
    """[(timp, caracter)] → [(nr_linie, t_start, t_end, text)], câte un rând pe linie."""
    lines, cur = [], []
    def flush():
        times = [t for t, _ in cur]
        lines.append((len(lines) + 1,
                      min(times) if times else None,
                      max(times) if times else None,
                      "".join(c for _, c in cur)))
    for t, c in chars:
        if c == "\n":
            flush()
            cur = []
        else:
            cur.append((t, c))
    if cur:
        flush()
    return lines

def read_events(labels_path: Path):
    with open(labels_path) as f:
        return [(float(r["timestamp_sec"]), r["key"]) for r in csv.DictReader(f)]

def write_final_text(labels_path: Path, out_path: Path):
    """Scrie final_text.csv (line, start_sec, end_sec, text). Returnează (text, nemodelate)."""
    chars, unmodeled = final_chars(read_events(labels_path))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["line", "start_sec", "end_sec", "text"])
        for n, t0, t1, text in final_lines(chars):
            w.writerow([n,
                        "" if t0 is None else f"{t0:.4f}",
                        "" if t1 is None else f"{t1:.4f}",
                        text])
    return "".join(c for _, c in chars), unmodeled
