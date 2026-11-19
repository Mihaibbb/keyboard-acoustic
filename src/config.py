from pathlib import Path

SR = 44100
LOWCUT = 100.0
HIGHCUT = 8000.0
# Anvelopă fină: tastarea rapidă are pauze de ~60-120ms între taste.
RMS_WIN_MS = 10.0
RMS_HOP_MS = 5.0
PEAK_MAD_K = 3.5
# Trebuie să fie < intervalul minim dintre taste, altfel se pierd apăsări.
PEAK_MIN_DIST_MS = 50.0
# Fereastra începe puțin înainte de vârf. Scurtă (25ms) ca să nu prindă apăsarea
# următoare: la tastare rapidă (<80ms între taste) o fereastră de 80ms conținea
# și începutul tastei vecine și o confunda. 25ms acoperă atacul tastei, unde e
# informația care o deosebește, și crește acuratețea mai ales pe scrisul rapid.
WINDOW_MS = 25.0
PRE_PEAK_MS = 10.0
# Augmentare la antrenare: fiecare fereastră apare și decalată cu ±atât.
# Poziția vârfului are o imprecizie de câțiva ms (pasul anvelopei e 5ms), iar
# spectrograma întreagă e sensibilă la ea; modelul învață să o tolereze.
AUG_SHIFT_MS = 2.5
N_MFCC = 20
N_FFT = 1024
HOP_LENGTH = 256
# Potrivire vârf ↔ etichetă după corecția decalajului (strâns, 1:1).
TOLERANCE_SEC = 0.02
# Vârfurile mai departe de orice etichetă decât asta devin clasa NONE_LABEL
# (sunete de eliberare a tastei, zgomot); cele între TOLERANCE și asta se ignoră.
NONE_MIN_DIST_SEC = 0.04
NONE_LABEL = "_none"
# Căutarea decalajului dintre ceasul etichetelor și ceasul audio.
OFFSET_SEARCH_SEC = 0.3
OFFSET_STEP_SEC = 0.002

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
SEED = 42

WINDOW_SAMPLES = int(SR * WINDOW_MS / 1000)
PRE_PEAK_SAMPLES = int(SR * PRE_PEAK_MS / 1000)
AUG_SHIFT_SAMPLES = int(SR * AUG_SHIFT_MS / 1000)
RMS_WIN = int(SR * RMS_WIN_MS / 1000)
RMS_HOP = int(SR * RMS_HOP_MS / 1000)
PEAK_MIN_DIST = int(SR * PEAK_MIN_DIST_MS / 1000)
