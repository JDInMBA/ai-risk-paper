"""
Compare results/ against a reference copy and report whether the pipeline reproduced.

Two runs on the same input data should be bit-for-bit identical. Files are read as
raw text and compared cell by cell: a cell that parses as a number is compared
numerically, anything else exactly. Reading raw text keeps the comparison
independent of headers, index columns, dtypes and duplicate column names, which
vary across these outputs. bsadf_sequences.csv and var_forecasts.csv carry a
two-row header, and the SHAP arrays are .npy rather than tables.

    python code/verify_reproduction.py results_reference_20261118
"""
import csv, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF  = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "results_reference_20261118")
NEW  = ROOT / "results"
TOL  = 1e-8          # relative, with the scale floored at 1.0

BLANK = {"", "na", "nan", "n/a", "none", "null"}

def num(s):
    """The float this field denotes, nan if it is blank, None if it is not a number."""
    t = s.strip()
    if t.lower() in BLANK:
        return float("nan")
    try:
        return float(t)
    except ValueError:
        return None

def compare_csv(pa, pb):
    with open(pa, newline="") as f: A = list(csv.reader(f))
    with open(pb, newline="") as f: B = list(csv.reader(f))
    exact, nviol, worst, where, notes = True, 0, 0.0, "", []
    if len(A) != len(B):
        return "tol", 1, 0.0, "", [f"{len(A)} rows -> {len(B)} rows"]
    for i, (ra, rb) in enumerate(zip(A, B), start=1):
        if len(ra) != len(rb):
            exact = False; nviol += 1
            if len(notes) < 3: notes.append(f"row {i}: {len(ra)} fields -> {len(rb)}")
            continue
        for j, (ca, cb) in enumerate(zip(ra, rb), start=1):
            if ca == cb:
                continue
            exact = False
            x, y = num(ca), num(cb)
            if x is None or y is None:                    # at least one is text
                nviol += 1
                if len(notes) < 3: notes.append(f"r{i}c{j}: {ca!r} -> {cb!r}")
                continue
            if np.isnan(x) and np.isnan(y):
                continue                                   # both blank, written differently
            if np.isnan(x) != np.isnan(y):
                nviol += 1
                if len(notes) < 3: notes.append(f"r{i}c{j}: {ca!r} -> {cb!r} (blank in one run)")
                continue
            rel = abs(x - y) / max(abs(x), 1.0)
            if rel > worst: worst, where = rel, f"r{i}c{j}"
            if rel > TOL:
                nviol += 1
                if len(notes) < 3: notes.append(f"r{i}c{j}: rel diff {rel:.3e}")
    return ("exact" if exact else "tol"), nviol, worst, where, notes

def compare_npy(pa, pb):
    a, b = np.load(pa), np.load(pb)
    if a.shape != b.shape:
        return "tol", 1, 0.0, "", [f"{a.shape} -> {b.shape}"]
    if np.array_equal(a, b, equal_nan=True):
        return "exact", 0, 0.0, "", []
    m = ~(np.isnan(a) | np.isnan(b))
    rel = np.abs(a[m] - b[m]) / np.maximum(np.abs(a[m]), 1.0)
    w = float(rel.max()) if rel.size else 0.0
    n = int((rel > TOL).sum()) + int((np.isnan(a) != np.isnan(b)).sum())
    return "tol", n, w, "array", []

# blank statsmodels' run timestamp
CLOCK = [(re.compile(r"(Date:\s+)\w{3}, \d{2} \w{3} \d{4}"), r"\1<run date>"),
         (re.compile(r"(Time:\s+)\d{2}:\d{2}:\d{2}"),        r"\1<run time>")]

def compare_txt(pa, pb):
    a, b = pa.read_bytes(), pb.read_bytes()
    if a == b:
        return "exact", 0, 0.0, "", []
    ta, tb = a.decode(), b.decode()
    for rx, sub in CLOCK:
        ta, tb = rx.sub(sub, ta), rx.sub(sub, tb)
    if ta == tb:
        return "clock", 0, 0.0, "", ["identical apart from the run date and time"]
    return "bad", 1, 0.0, "", ["text output differs"]

HANDLERS = {".csv": compare_csv, ".npy": compare_npy, ".txt": compare_txt}

# provenance note, not an output
NOTE = "PROVENANCE.txt"
prov = REF / NOTE
if prov.exists():
    print(prov.read_text().rstrip()); print("-" * 72)

ref = {p.name: p for p in sorted(REF.iterdir()) if p.suffix in HANDLERS and p.name != NOTE}
new = {p.name: p for p in sorted(NEW.iterdir()) if p.suffix in HANDLERS and p.name != NOTE}

missing = sorted(set(ref) - set(new))
extra   = sorted(set(new) - set(ref))

n_exact = n_tol = n_clock = n_bad = 0
worst = (0.0, "")
kinds = {}

for name, pa in ref.items():
    if name not in new:
        continue
    kind, nviol, w, where, notes = HANDLERS[pa.suffix](pa, new[name])
    kinds[name] = kind
    if w > worst[0]:
        worst = (w, f"{name}:{where}")
    if nviol:
        print(f"DIFFERS    {name}: {nviol} cell(s) beyond tolerance" +
              ("; " + "; ".join(notes) if notes else ""))
        n_bad += 1
    elif kind == "exact":
        n_exact += 1
    elif kind == "clock":
        n_clock += 1
        print(f"timestamp  {name}: {notes[0]}")
    else:
        n_tol += 1
        print(f"last digit {name}: identical within {TOL:g}" +
              (f", largest relative gap {w:.3e} at {where}" if w else ""))

print()
print(f"reference            : {REF.name}  ({len(ref)} files)")
print(f"bit-for-bit identical: {n_exact}")
print(f"equal within {TOL:g}   : {n_tol}")
if n_clock: print(f"equal but for a clock: {n_clock}")
print(f"differing            : {n_bad}")
if missing: print(f"missing from new run : {missing}")
if extra:   print(f"new files produced   : {extra}")
if worst[1]: print(f"largest relative gap : {worst[0]:.3e}  ({worst[1]})")
# the timestamp doubles as a run fingerprint
STAMPED = "event_pooled.txt"
same_run = STAMPED in ref and kinds.get(STAMPED) == "exact"

print()
if n_bad or missing:
    print("CHECK THE LINES ABOVE")
elif same_run:
    print(f"SAME RUN -- even {STAMPED}, which carries the run's own date and time, is")
    print("identical, so this reference is a copy of the run now in results/ rather than")
    print("an independent one. Nothing has been verified. Re-run the pipeline, then compare.")
elif n_tol:
    print(f"REPRODUCED within {TOL:g} -- {n_tol} file(s) differ only in printed digits")
elif n_clock:
    print(f"REPRODUCED -- every number identical; {n_clock} file(s) carry a run timestamp")
else:
    print("REPRODUCED -- bit-for-bit identical")
    if STAMPED not in ref:
        print(f"(no {STAMPED} in the reference, so this cannot tell two runs from one)")
