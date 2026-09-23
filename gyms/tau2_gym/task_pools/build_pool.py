#!/usr/bin/env python3
"""Build the Decomposer task pool from Timur's canonical-harness calibration.

Pool = (canon_full_sft | canon_full_grpo | canon_full_dpo | dead)
       INTERSECT domains the Decomposer can drive (ScriptedUser)
       MINUS every held-out set and QUARANTINE.

Bucket assignments are deliberately NOT inherited: they were cut on a solo
Qwen3.5-4B, and the Decomposer is a stronger system, so its band sits
elsewhere. What we inherit is the task list, the defect hygiene, and the
harness stamp.
"""
import json, glob, os, sys, collections, hashlib, datetime

ROOT = os.path.expanduser("~/tau2-gym")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.expanduser("~/decomposer_sft/external/tau2_gym/training"))
from tau2.registry import registry
import tau2_env_manager as M

FD = os.path.join(ROOT, "data/training_splits")
SCRIPTED = {d for d in registry.get_domains() if M._user_type_for_domain(d) == "scripted"}

def flat(name):
    p = os.path.join(FD, name)
    j = json.load(open(p)); j.pop("_meta", None)
    return {(d, i) for d, v in j.items() if isinstance(v, list) for i in v}

SRC = ["canon_full_sft.json", "canon_full_grpo.json", "canon_full_dpo.json"]
EXC = ["HELDOUT_v2.json", "heldout_exec_v5.json", "cycle_heldout.json", "QUARANTINE.json"]

union = set().union(*(flat(n) for n in SRC))
excl  = set().union(*(flat(n) for n in EXC))

# dead tasks (p=0) from the same calibration - canon drops them by construction
passrate = {}
for p in sorted(glob.glob(os.path.join(ROOT, "progress/gaia2/unified/full/full_report_*.json"))):
    j = json.load(open(p))
    trials_default = j.get("trials", 8)
    for r in j.get("results", []):
        d, t = r.get("domain"), (r.get("task_id") or r.get("task"))
        if d is None or t is None: continue
        tr = r.get("trials", trials_default)
        n = r.get("n_pass", r.get("passes"))
        if n is None and r.get("pass_rate") is not None:
            n = round(r["pass_rate"] * tr)
        if n is not None:
            passrate[(d, t)] = (n, tr)
dead = {k for k, (n, _) in passrate.items() if n == 0}

pool = {k for k in (union | dead) if k[0] in SCRIPTED} - excl

by_dom = collections.defaultdict(list)
for d, i in sorted(pool):
    by_dom[d].append(i)
out = {d: sorted(v) for d, v in sorted(by_dom.items())}

OUT = os.path.expanduser("~/decomposer_sft/gyms/tau2_gym/task_pools")
os.makedirs(OUT, exist_ok=True)
fp = os.path.join(OUT, "decomposer_pool_v1.json")
json.dump(out, open(fp, "w"), indent=1)

meta = {
    "name": "decomposer_pool_v1",
    "built": datetime.date.today().isoformat(),
    "built_by": "gyms/tau2_gym/task_pools/build_pool.py",
    "purpose": "Decomposer (DeepSeek manager + Qwen3.5-4B subagents) task pool. "
               "Membership only - bucket assignment requires our own calibration.",
    "sources": SRC + ["progress/gaia2/unified/full/full_report_*.json (dead p=0)"],
    "exclusions": EXC,
    "executable_filter": "tau2_env_manager._user_type_for_domain(d) == 'scripted'",
    "harness_stamp_inherited_from": "canon_full_splits.meta.json",
    "counts": {
        "canon_union": len(union),
        "dead_from_calibration": len(dead),
        "scripted_domains": len(SCRIPTED),
        "pool_tasks": len(pool),
        "pool_domains": len(out),
        "of_which_dead": len({k for k in pool if k in dead}),
    },
    "inherited_bucket_reference": {
        n.replace(".json", ""): len({k for k in flat(n) if k in pool}) for n in SRC
    },
    "sha256": hashlib.sha256(open(fp, "rb").read()).hexdigest()[:16],
}
json.dump(meta, open(fp.replace(".json", ".meta.json"), "w"), indent=1)

print("wrote", fp)
print(json.dumps(meta["counts"], indent=1))
print("inherited bucket reference:", json.dumps(meta["inherited_bucket_reference"]))
# hygiene assertion
assert not (pool & excl), "held-out leak"
print("overlap with every held-out/quarantine set: 0  OK")
