"""Re-hash every file and compare against MANIFEST.json."""
import hashlib, json, os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))

def sha256(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(buf), b""):
            h.update(c)
    return h.hexdigest()

man = json.load(open(os.path.join(ROOT, "MANIFEST.json"), encoding="utf-8"))
bad = []
for e in man["files"]:
    p = os.path.join(ROOT, e["path"].replace("/", os.sep))
    if not os.path.exists(p):
        bad.append((e["path"], "MISSING")); continue
    if os.path.getsize(p) != e["bytes"]:
        bad.append((e["path"], "wrong size")); continue
    if sha256(p) != e["sha256"]:
        bad.append((e["path"], "sha256 mismatch"))
print("checked %d files" % len(man["files"]))
if bad:
    for p, why in bad:
        print("  [FAIL] %s - %s" % (p, why))
    print("%d problem(s). Re-download or re-copy the package." % len(bad))
    sys.exit(1)
print("OK - every file matches the manifest.")
