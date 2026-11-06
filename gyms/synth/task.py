"""Two independent swaps, varying only opaque file names and their presentation."""
import random
import string

GRAMMAR = """Workers execute a strict command language, not natural language.
Each nonempty line must be exactly COPY source.txt target.txt or CHECK target.txt original_source.txt.
COPY overwrites the target with the source's CURRENT contents. CHECK compares the target's current contents with the named file's INITIAL contents; it never changes files.
Commands in one run execute in order. Workers cannot plan, create files, or interpret prose or Markdown. Only the six named files exist. All workers share the workspace. The two scratch files initially contain nothing.
Use different workers for the two independent swaps and launch both before waiting. After the swaps, use a separate worker for read-only verification. Final prose alone cannot change the files."""


def make_task(split, index):
    if split not in {"train", "eval"} or index < 0:
        raise ValueError("Use train/eval and a nonnegative index")
    rng = random.Random(f"synth-swap-v1:{split}:{index}")
    names = []
    while len(names) < 6:
        name = "".join(rng.choices(string.ascii_lowercase, k=8)) + ".txt"
        if name not in names:
            names.append(name)
    a, b, c, d, t, u = names
    initial = dict(zip(names, ["ONE", "TWO", "THREE", "FOUR", "", ""]))
    display = names.copy()
    rng.shuffle(display)
    prompt = (
        f"Swap the contents of {a} and {b}. Independently, swap the contents of {c} and {d}. "
        f"Scratch files: {t}, {u}. All six files: {', '.join(display)}. "
        "Both swaps must preserve the original contents and should be delegated in parallel.\n\n" + GRAMMAR
    )
    return {"task_id": f"{split}-{index:04d}", "split": split, "index": index,
            "prompt": prompt, "initial": initial, "pairs": [[a, b], [c, d]],
            "scratch": [t, u], "expected": {a: "TWO", b: "ONE", c: "FOUR", d: "THREE"}}


def reference_plans(task):
    return [f"COPY {a} {t}\nCOPY {b} {a}\nCOPY {t} {b}"
            for (a, b), t in zip(task["pairs"], task["scratch"])]
