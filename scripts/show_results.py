import json
m = json.load(open("models/metrics.json"))
print(m["dataset"])
for k, d in m["models"].items():
    print(k, {x: round(d[x], 3) for x in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "inference_ms")})
for k, d in m["ablation"].items():
    print(k, "acc", round(d["accuracy"], 3), "f1", round(d["f1_macro"], 3))
