import json

with open("eval_terbaru.json", "r", encoding="utf-8") as f:
    data = json.load(f)

errors = data.get("errors_high_conf", [])
print(f"Total errors: {len(errors)}")

# Print errors with conf >= 0.95
high_errors = [e for e in errors if e["conf"] >= 0.95]
print(f"Errors with conf >= 0.95: {len(high_errors)}")
for e in sorted(high_errors, key=lambda x: (x["true"], x["pred"], -x["conf"])):
    print(f"[{e['true']} -> {e['pred']}] {e['file']} (conf: {e['conf']:.4f})")
