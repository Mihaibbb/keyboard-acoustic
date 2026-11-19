"""Decodare cu model de limbaj peste probabilitățile acustice (shallow fusion).

Clasificatorul SVM dă, pentru fiecare vârf detectat, un vector de probabilități
pe clase (``model.predict_proba``). Clasele confundabile diferă acustic foarte
puțin (l/k, d/f, i/u, n/m ...), dar sunt ușor de separat din context lingvistic.
Aici combinăm scorul acustic (log-probabilitatea pe clasă) cu un model de limbaj
caracter-cu-caracter (bigram / trigram interpolat) estimat DOAR din textul real
tastat în sesiunile de antrenare, printr-o căutare beam.

Nimic din fișierele existente nu se modifică: acest modul doar se adaugă. Intrarea
tipică e matricea de probabilități a vârfurilor dintr-o linie (în ordinea timpului)
plus etichetele de bază ale SVM-ului; ieșirea e secvența de taste corectată, care
trece apoi prin ``src.text.final_chars`` exact ca în ``src.compare`` (calea „orbă”).

Taste structurale (space, backspace, enter, shift) trec practic nemodificate:
au sunet distinctiv, deci probabilitatea acustică le câștigă singură, iar modelul
de limbaj punctează mai ales literele. Space poate fi punctat ca graniță de cuvânt
(``score_space``), ceea ce ajută modelarea tranzițiilor între cuvinte.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Simbolul de început (boundary) folosit ca context înaintea primului caracter
# și imediat după un enter (început de linie nouă).
BOS = "\x02"
# Alfabetul modelului de limbaj: 26 litere + spațiu. Orice altceva din textul de
# antrenare (cifre, punctuație, newline din interiorul unei celule) devine spațiu.
_LETTERS = "abcdefghijklmnopqrstuvwxyz"
LM_ALPHABET = _LETTERS + " "
_LETTER_SET = set(_LETTERS)


def _to_symbol_stream(text: str) -> str:
    """Normalizează textul de antrenare la alfabetul LM (a-z și spațiu)."""
    out = []
    prev_space = True  # evită spații duplicate / spațiu la început
    for ch in text.lower():
        if ch in _LETTER_SET:
            out.append(ch)
            prev_space = False
        else:
            # cifre, punctuație, newline, tab -> graniță de cuvânt
            if not prev_space:
                out.append(" ")
                prev_space = True
    return "".join(out)


class CharLM:
    """Model de limbaj caracter-cu-caracter cu interpolare liniară (back-off moale).

    Pentru ordinul ``order`` (2 = bigram, 3 = trigram) probabilitatea unui simbol
    dat contextul este o combinație a estimărilor pe context de lungime
    0..order-1, cu o podea uniformă. Nu dă niciodată 0, deci log-ul e mereu finit.
    """

    def __init__(self, order: int = 3):
        if order < 1:
            raise ValueError("order >= 1")
        self.order = order
        self.V = len(LM_ALPHABET)
        self._idx = {c: i for i, c in enumerate(LM_ALPHABET)}
        # counts[m] : dict context(tuple de m simboluri) -> np.array(V) de counts
        self._counts: List[Dict[Tuple[str, ...], np.ndarray]] = [
            defaultdict(lambda: np.zeros(self.V, dtype=np.float64))
            for _ in range(order)
        ]
        self._totals: List[Dict[Tuple[str, ...], float]] = [
            defaultdict(float) for _ in range(order)
        ]
        # Greutăți de interpolare (de la cel mai lung context la cel mai scurt) +
        # podea uniformă. Suma = 1. Valori rezonabile; se pot regla din exterior.
        self.weights = self._default_weights(order)
        self._fitted = False

    @staticmethod
    def _default_weights(order: int) -> np.ndarray:
        # ex. trigram -> [tri, bi, uni, floor]; mai multă masă pe contextul lung.
        if order == 1:
            w = np.array([0.95, 0.05])
        elif order == 2:
            w = np.array([0.6, 0.35, 0.05])
        else:
            w = np.array([0.5, 0.3, 0.18, 0.02])
            if order > 3:
                # extinde uniform pentru ordine mai mari
                extra = order - 3
                base = np.linspace(0.5, 0.15, order)
                base = base / base.sum() * 0.98
                w = np.append(base, 0.02)
        return w

    def set_weights(self, weights: Sequence[float]) -> "CharLM":
        w = np.asarray(weights, dtype=np.float64)
        if len(w) != self.order + 1:
            raise ValueError(f"nevoie de {self.order + 1} greutăți (contexte + podea)")
        self.weights = w / w.sum()
        return self

    def fit_text(self, text: str) -> "CharLM":
        stream = _to_symbol_stream(text)
        if not stream:
            return self
        # Prefixează cu BOS pentru contextul de la început de linie.
        syms = [BOS] * (self.order - 1) + list(stream)
        for i in range(self.order - 1, len(syms)):
            c = syms[i]
            ci = self._idx[c]
            for m in range(self.order):
                ctx = tuple(syms[i - m:i]) if m else ()
                self._counts[m][ctx][ci] += 1.0
                self._totals[m][ctx] += 1.0
        self._fitted = True
        return self

    def fit_texts(self, texts: Sequence[str]) -> "CharLM":
        for t in texts:
            self.fit_text(t)
        return self

    def _ctx_prob(self, m: int, ctx: Tuple[str, ...], ci: int) -> float:
        """MLE a simbolului ci pe context de lungime m (0 dacă contextul lipsește)."""
        tot = self._totals[m].get(ctx, 0.0)
        if tot <= 0.0:
            return -1.0  # context nevăzut -> semnalează lipsă
        return self._counts[m][ctx][ci] / tot

    def prob(self, symbol: str, context: Tuple[str, ...]) -> float:
        """P(symbol | context) prin interpolare; context = simboluri recente."""
        ci = self._idx.get(symbol)
        if ci is None:
            # simbol în afara alfabetului (nu ar trebui); dă podea uniformă
            return self.weights[-1] / self.V
        # context tăiat la order-1 simboluri
        ctx_full = tuple(context[-(self.order - 1):]) if self.order > 1 else ()
        p = self.weights[-1] / self.V  # podea uniformă
        for m in range(self.order):
            if m == 0:
                ctx = ()
            else:
                ctx = ctx_full[-m:]
                if len(ctx) < m:
                    continue  # context prea scurt pentru acest ordin
            pm = self._ctx_prob(m, ctx, ci)
            if pm >= 0.0:
                # weights sunt [order-1, order-2, ..., 0, floor]; m=order-1 e primul
                p += self.weights[self.order - 1 - m] * pm
        return p

    def logprob(self, symbol: str, context: Tuple[str, ...]) -> float:
        return math.log(self.prob(symbol, context) + 1e-12)


def train_lm_from_sessions(session_dirs: Sequence[Path], order: int = 3) -> CharLM:
    """Estimează modelul de limbaj din ``final_text.csv`` al sesiunilor date.

    Dacă lipsește ``final_text.csv``, se reconstruiește textul din ``labels.csv``
    prin ``src.text.final_chars`` (aceeași semantică de backspace/enter).
    """
    lm = CharLM(order=order)
    texts: List[str] = []
    for d in session_dirs:
        d = Path(d)
        ft = d / "final_text.csv"
        if ft.exists():
            with open(ft, newline="") as f:
                for row in csv.DictReader(f):
                    texts.append(row.get("text", "") or "")
        else:
            # reconstruiește din labels.csv
            from .prepare import normalize_key
            from .text import final_chars, read_events
            events = [(t, normalize_key(k)) for t, k in read_events(d / "labels.csv")]
            chars, _ = final_chars(events)
            texts.append("".join(c for _, c in chars))
    lm.fit_texts(texts)
    return lm


# --- Căutare beam ------------------------------------------------------------

class _TextNode:
    """Lanț imuabil de caractere emise (pentru context LM + backspace în O(1))."""
    __slots__ = ("parent", "ch")

    def __init__(self, parent: Optional["_TextNode"], ch: str):
        self.parent = parent
        self.ch = ch  # caracter a-z / ' ' / '\n'


def _apply_key(node: Optional[_TextNode], key: str) -> Optional[_TextNode]:
    """Aplică o tastă la fluxul de text, exact ca ``src.text.final_chars``."""
    if key == "backspace":
        return node.parent if node is not None else None
    if key == "space":
        return _TextNode(node, " ")
    if key == "enter":
        return _TextNode(node, "\n")
    if key == "tab":
        return _TextNode(node, "\t")
    if len(key) == 1:
        return _TextNode(node, key)
    # shift / shift_r / orice tastă nemodelată -> fără efect pe text
    return node


def _lm_context(node: Optional[_TextNode], k: int) -> Tuple[str, ...]:
    """Ultimele k simboluri LM (a-z / spațiu) până la începutul liniei.

    Se oprește la newline (enter = început de linie nou, context BOS). Caracterele
    în afara alfabetului (tab etc.) sunt tratate ca graniță de cuvânt (spațiu).
    """
    if k <= 0:
        return ()
    syms: List[str] = []
    cur = node
    while cur is not None and len(syms) < k:
        ch = cur.ch
        if ch == "\n":
            break  # început de linie
        if ch in _LETTER_SET:
            syms.append(ch)
        else:
            syms.append(" ")  # spațiu / tab / alt -> graniță
        cur = cur.parent
    syms.reverse()
    pad = k - len(syms)
    if pad > 0:
        syms = [BOS] * pad + syms
    return tuple(syms)


def _key_symbol(key: str) -> Optional[str]:
    """Simbolul LM pe care îl emite o tastă: literă, ' ' (spațiu), sau None."""
    if len(key) == 1 and key in _LETTER_SET:
        return key
    if key == "space":
        return " "
    return None  # backspace / enter / shift -> structural, fără scor LM de literă


class _Hyp:
    __slots__ = ("score", "text", "parent", "key")

    def __init__(self, score: float, text: Optional[_TextNode],
                 parent: Optional["_Hyp"], key: Optional[str]):
        self.score = score
        self.text = text
        self.parent = parent
        self.key = key


def beam_decode(
    probs: np.ndarray,
    classes: Sequence[str],
    lm: CharLM,
    lam: float = 0.7,
    beam_width: int = 12,
    topk: int = 6,
    score_space: bool = True,
    force_labels: Optional[Sequence[str]] = None,
) -> List[str]:
    """Decodare beam a unei secvențe de vârfuri (în ordinea timpului).

    Parametri
    ---------
    probs : (n_peaks, n_classes) probabilități acustice (``model.predict_proba``).
    classes : etichetele coloanelor (``model.classes_`` ca str).
    lm : modelul de limbaj caracter.
    lam : greutatea termenului LM în fuziune (scor = acustic + lam * LM).
    beam_width : câte ipoteze se păstrează.
    topk : câți candidați acustici se încearcă per vârf.
    score_space : dacă spațiul primește și el scor LM (graniță de cuvânt).
    force_labels : eticheta preferată a SVM-ului per vârf (``model.predict``);
        se garantează că e mereu printre candidați, ca decodarea să nu piardă
        accesul la alegerea SVM-ului chiar dacă argmax(predict_proba) diferă.

    Întoarce lista de taste decodate (aceeași lungime ca ``probs``).
    """
    probs = np.asarray(probs, dtype=np.float64)
    n = probs.shape[0]
    classes = [str(c) for c in classes]
    if n == 0:
        return []
    logp = np.log(probs + 1e-12)
    # candidați per vârf: top-k clase după probabilitate (+ eticheta SVM forțată)
    order_desc = np.argsort(-probs, axis=1)
    cand_idx: List[List[int]] = []
    col = {c: i for i, c in enumerate(classes)}
    for i in range(n):
        cand = list(order_desc[i, :topk])
        if force_labels is not None:
            fi = col.get(str(force_labels[i]))
            if fi is not None and fi not in cand:
                cand.append(fi)
        cand_idx.append(cand)

    beam: List[_Hyp] = [_Hyp(0.0, None, None, None)]
    k_ctx = lm.order - 1
    for i in range(n):
        new: List[_Hyp] = []
        li = logp[i]
        cands = cand_idx[i]
        for h in beam:
            ctx = _lm_context(h.text, k_ctx)
            for ci in cands:
                key = classes[ci]
                a = li[ci]
                sym = _key_symbol(key)
                lm_lp = 0.0
                if sym is not None and (sym != " " or score_space):
                    lm_lp = lm.logprob(sym, ctx)
                new.append(_Hyp(h.score + a + lam * lm_lp,
                                _apply_key(h.text, key), h, key))
        # taie la beam_width după scor
        if len(new) > beam_width:
            new.sort(key=lambda z: z.score, reverse=True)
            beam = new[:beam_width]
        else:
            beam = new

    best = max(beam, key=lambda z: z.score)
    # reconstruiește secvența de taste urmând back-pointerii
    keys: List[str] = []
    cur: Optional[_Hyp] = best
    while cur is not None and cur.key is not None:
        keys.append(cur.key)
        cur = cur.parent
    keys.reverse()
    return keys


def decode_labels(
    model,
    probs: np.ndarray,
    base_labels: Sequence[str],
    lm: CharLM,
    lam: float = 0.7,
    beam_width: int = 12,
    topk: int = 6,
    score_space: bool = True,
) -> List[str]:
    """Interfață de nivel înalt: corectează etichetele de bază ale SVM-ului.

    ``probs`` = ``model.predict_proba`` pentru vârfurile (în ordinea timpului) ale
    unei linii, ``base_labels`` = ``model.predict`` pentru aceleași vârfuri.
    """
    return beam_decode(
        probs, [str(c) for c in model.classes_], lm,
        lam=lam, beam_width=beam_width, topk=topk,
        score_space=score_space, force_labels=base_labels,
    )
