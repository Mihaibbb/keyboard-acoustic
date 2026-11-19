"""Punct de intrare CLI: record | prepare | train | evaluate | predict."""
import argparse
import sys
from pathlib import Path
import joblib
from . import config as C

def cmd_record(args):
    from .record import record_session
    n, d = record_session(Path(args.output), args.duration, sr=C.SR)
    print(f"Înregistrate {n} apăsări în {d:.1f}s → {args.output}")
    _show_final_text(Path(args.output))

def _show_final_text(session: Path):
    from .text import write_final_text
    text, unmodeled = write_final_text(session / "labels.csv",
                                       session / "final_text.csv")
    print(f"\n[{session.name}] text final → {session / 'final_text.csv'}")
    if unmodeled:
        print(f"Atenție: {unmodeled} taste de editare/modificatori (săgeți, Cmd, "
              "Option) — textul poate diferi de cel de pe ecran.")
    print("-" * 60)
    print(text)
    print("-" * 60)

def cmd_text(args):
    raw = Path(args.raw)
    sessions = [raw] if (raw / "labels.csv").exists() else \
        [s for s in sorted(raw.iterdir()) if (s / "labels.csv").exists()]
    if not sessions:
        print(f"Niciun labels.csv în {raw}", file=sys.stderr)
        sys.exit(1)
    for s in sessions:
        _show_final_text(s)

def cmd_prepare(args):
    from .prepare import prepare_sessions
    shape, ncls = prepare_sessions(Path(args.raw), Path(args.out), sr=C.SR,
                                   include_none=args.with_none)
    print(f"Ferestre procesate: {shape}, clase: {ncls}")

def cmd_train(args):
    from collections import Counter
    from .train import (load_processed, train_model, split,
                        filter_classes, augment_time_shifts)
    from .features import FEATURE_KINDS

    X, y = load_processed(Path(args.data))
    X, y, counts = filter_classes(X, y, min_samples=args.min_samples)

    kept = len(set(y))
    total = len(counts)
    print(f"Clase păstrate: {kept}/{total} (min {args.min_samples} mostre)")
    print(f"Total ferestre: {len(X)}")
    print("Distribuție:")
    for k, v in Counter(y).most_common():
        print(f"  {k!r}: {v}")

    if kept < 2:
        print("Eroare: <2 clase după filtrare. Înregistrează mai multe date "
              "sau scade --min-samples.", file=sys.stderr)
        sys.exit(1)

    import numpy as np
    extract = FEATURE_KINDS[args.features]
    augment = not args.no_augment

    def fit(Xw, yw, probability=True):
        if augment:
            Xw, yw = augment_time_shifts(Xw, yw)
        return train_model(extract(Xw), yw, feature_kind=args.features,
                           probability=probability)

    Xtr, Xte, ytr, yte = split(X, y)
    # Cu --final, modelul de validare e doar pentru măsurat: fără probabilități
    # (calibrarea lor internă face antrenarea de câteva ori mai lungă).
    model = fit(Xtr, ytr, probability=not args.final)
    val_acc = model.score(extract(Xte), yte)
    if args.final:
        # Modelul salvat vede toate datele; cifra de validare rămâne cea
        # măsurată mai sus, pe modelul antrenat fără split-ul de test.
        model = fit(np.concatenate([Xtr, Xte]), np.concatenate([ytr, yte]))
        model.trained_on_all_ = True
    model.augmented_ = augment
    # evaluate trebuie să refacă exact același split → reține filtrul folosit.
    model.min_samples_ = args.min_samples
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model)
    print(f"Model salvat: {args.model}"
          + (" (antrenat pe toate datele)" if args.final else ""))
    print(f"Acuratete split validare: {val_acc:.4f}")

def cmd_evaluate(args):
    import numpy as np
    from .train import load_processed, split, filter_classes
    from .features import features_for
    from .evaluate import evaluate_model
    from .ensemble import load_any
    model = load_any(args.model)
    X, y = load_processed(Path(args.data))
    if args.holdout:
        # Sesiune separată, nevăzută la antrenare: se evaluează tot, mai puțin
        # clasele pe care modelul nu le cunoaște (ar fi ratări garantate).
        known = np.isin(y, model.classes_)
        if (~known).any():
            print(f"Ignorate {int((~known).sum())} ferestre din clase necunoscute "
                  f"modelului: {sorted(set(y[~known]))}")
        Xte, yte = X[known], y[known]
    else:
        # Același filtru + același split ca la train → fără leakage.
        min_samples = getattr(model, "min_samples_", args.min_samples)
        X, y, _ = filter_classes(X, y, min_samples=min_samples)
        _, Xte, _, yte = split(X, y)
        if getattr(model, "trained_on_all_", False):
            print("Atenție: modelul a fost antrenat cu --final, deci a văzut și "
                  "acest split. Cifrele de mai jos sunt umflate; pentru o "
                  "măsură cinstită folosește --holdout pe o sesiune separată.")
    if len(yte) == 0:
        print("Eroare: niciun eșantion de evaluat.", file=sys.stderr)
        sys.exit(1)
    fte = features_for(model, Xte)
    _, report, _, _ = evaluate_model(model, fte, yte, Path(args.reports))
    print(report)

def cmd_detector(args):
    from .detector import train_detector
    model, n = train_detector(Path(args.raw))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)
    print(f"Detector antrenat pe {n} vârfuri → {args.out}")

def _build_lm():
    """Model de limbaj din textul real al sesiunilor de antrenare (data/raw)."""
    from .decode import train_lm_from_sessions
    sess = [s for s in sorted(C.DATA_RAW.iterdir())
            if s.is_dir() and (s / "labels.csv").exists()]
    return train_lm_from_sessions(sess, order=3)

def cmd_predict(args):
    from .predict import predict_audio
    from .ensemble import load_any
    model = load_any(args.model)
    detector = joblib.load(args.detector) if args.detector else None
    lm = _build_lm() if args.decode else None
    results = predict_audio(model, args.audio, detector=detector,
                            threshold=args.threshold, lm=lm, lam=args.lam)
    if not results:
        print("Nicio apăsare detectată.")
        return
    print(f"Detectate {len(results)} apăsări:")
    print(f"{'timp (s)':>10}  {'tastă':^6}  {'încredere':>10}")
    for r in results:
        print(f"{r['time_sec']:>10.3f}  {r['key']:^6}  {r['confidence']*100:>9.1f}%")

def cmd_compare(args):
    from .compare import compare_session
    from .ensemble import load_any
    model = load_any(args.model)
    detector = joblib.load(args.detector) if args.detector else None
    lm = _build_lm() if args.decode else None
    root = Path(args.session)
    sessions = [root] if (root / "labels.csv").exists() else \
        [s for s in sorted(root.iterdir()) if (s / "labels.csv").exists()]
    if not sessions:
        print(f"Niciun labels.csv în {root}", file=sys.stderr)
        sys.exit(1)
    for session in sessions:
        # Implicit CSV-urile stau în folderul sesiunii, lângă labels.csv.
        out = Path(args.reports) / session.name if args.reports else session
        r = compare_session(model, session, out, detector=detector,
                            threshold=args.threshold, lm=lm, lam=args.lam)
        print(f"\n[{session.name}] apăsări {r['labels']} | vârfuri audio "
              f"{r['peaks']} | detectate {r['detected']}")
        print(f"Taste corecte (pe apăsările detectate): {r['key_acc']:.1%}")
        print(f"Caractere corecte, momente cunoscute:   {r['char_acc_known_times']:.1%}")
        print(f"Caractere corecte, orb (doar audio):    {r['char_acc_blind']:.1%}")
        print(f"→ {out / 'text_vs_model.txt'}\n→ {out / 'predicted_text.csv'}\n→ {out / 'keys_vs_pred.csv'}")
        for row in r["rows"] if args.preview else []:
            print(f"--- linia {row[0]} ({row[1]}–{row[2]}s)")
            print(f"real : {row[5][:args.preview]}")
            print(f"model: {row[6][:args.preview]}")
            print(f"orb  : {row[7][:args.preview]}")

def main(argv=None):
    p = argparse.ArgumentParser(prog="keyboard-acoustic")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record")
    r.add_argument("--output", required=True)
    r.add_argument("--duration", type=float, default=60.0)
    r.set_defaults(func=cmd_record)

    tx = sub.add_parser("text", help="scrie final_text.csv din labels.csv")
    tx.add_argument("--raw", required=True,
                    help="o sesiune sau un folder cu sesiuni")
    tx.set_defaults(func=cmd_text)

    pr = sub.add_parser("prepare")
    pr.add_argument("--raw", default=str(C.DATA_RAW))
    pr.add_argument("--out", default=str(C.DATA_PROCESSED))
    pr.add_argument("--with-none", action="store_true",
                    help=f"include vârfurile fără apăsare ca clasa {C.NONE_LABEL} "
                         "(experimental)")
    pr.set_defaults(func=cmd_prepare)

    t = sub.add_parser("train")
    t.add_argument("--data", default=str(C.DATA_PROCESSED))
    t.add_argument("--model", default=str(C.MODELS / "baseline.joblib"))
    t.add_argument("--min-samples", type=int, default=15)
    t.add_argument("--features", choices=["mel", "summary"], default="mel",
                   help="mel: spectrogramă log-mel întreagă (implicit); "
                        "summary: rezumat MFCC (varianta veche)")
    t.add_argument("--no-augment", action="store_true",
                   help="fără ferestrele decalate în timp la antrenare "
                        "(mai rapid, puțin mai slab)")
    t.add_argument("--final", action="store_true",
                   help="după validare, reantrenează pe toate datele și salvează acel model")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate")
    e.add_argument("--model", default=str(C.MODELS / "baseline.joblib"))
    e.add_argument("--data", default=str(C.DATA_PROCESSED))
    e.add_argument("--reports", default=str(C.REPORTS))
    e.add_argument("--min-samples", type=int, default=15,
                   help="folosit doar dacă modelul nu are min_samples_ salvat")
    e.add_argument("--holdout", action="store_true",
                   help="--data e o sesiune separată: evaluează tot, fără split")
    e.set_defaults(func=cmd_evaluate)

    dt = sub.add_parser("detector", help="antrenează detectorul binar apăsare/non-apăsare")
    dt.add_argument("--raw", default=str(C.DATA_RAW))
    dt.add_argument("--out", default=str(C.MODELS / "detector.joblib"))
    dt.set_defaults(func=cmd_detector)

    pr2 = sub.add_parser("predict")
    pr2.add_argument("--model", default=str(C.MODELS / "baseline.joblib"))
    pr2.add_argument("--audio", required=True)
    pr2.add_argument("--detector", help="detector binar: aruncă vârfurile "
                     "care nu sunt apăsări (eliberări, zgomot)")
    pr2.add_argument("--threshold", type=float, default=0.6)
    pr2.add_argument("--decode", action="store_true",
                     help="recombină tastele cu un model de limbaj (beam search)")
    pr2.add_argument("--lam", type=float, default=0.6,
                     help="greutatea modelului de limbaj la --decode")
    pr2.set_defaults(func=cmd_predict)

    cp = sub.add_parser("compare", help="text real vs text reconstruit din audio")
    cp.add_argument("--model", default=str(C.MODELS / "baseline.joblib"))
    cp.add_argument("--session", required=True,
                    help="o sesiune sau un folder cu sesiuni")
    cp.add_argument("--reports", default=None,
                    help="alt folder de ieșire (implicit: folderul sesiunii)")
    cp.add_argument("--detector", help="detector binar pentru reconstrucția oarbă")
    cp.add_argument("--threshold", type=float, default=0.6)
    cp.add_argument("--decode", action="store_true",
                    help="reconstrucția oarbă folosește un model de limbaj "
                         "(beam search peste probabilitățile acustice)")
    cp.add_argument("--lam", type=float, default=0.6,
                    help="greutatea modelului de limbaj la --decode")
    cp.add_argument("--preview", type=int, default=160,
                    help="câte caractere din fiecare linie se afișează")
    cp.set_defaults(func=cmd_compare)

    args = p.parse_args(argv)
    args.func(args)

if __name__ == "__main__":
    main()
