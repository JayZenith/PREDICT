"""Paired McNemar test + bootstrap CI on per-task pass/fail outcomes.

Usage: python3 docs/stats.py TRACES_A.jsonl TRACES_B.jsonl [NAME_A] [NAME_B]

Each traces.jsonl line must have task.data.name and a rewards dict; a task
counts as a pass if its (single) reward value is > 0.
"""
import json
import math
import random
import sys


def load_outcomes(path):
    out = {}
    for line in open(path):
        t = json.loads(line)
        name = t["task"]["data"].get("name")
        reward = list((t.get("rewards") or {"x": 0}).values())[0]
        out[name] = 1 if reward > 0 else 0
    return out


def compare(name_a, outcomes_a, name_b, outcomes_b, bootstrap_n=10000, seed=0):
    common = sorted(set(outcomes_a) & set(outcomes_b))
    n = len(common)
    pass_a = sum(outcomes_a[k] for k in common)
    pass_b = sum(outcomes_b[k] for k in common)

    both_pass = a_only = b_only = both_fail = 0
    for k in common:
        x, y = outcomes_a[k], outcomes_b[k]
        if x and y:
            both_pass += 1
        elif x and not y:
            a_only += 1
        elif y and not x:
            b_only += 1
        else:
            both_fail += 1

    print(f"\n=== {name_a} vs {name_b} (n={n}) ===")
    print(f"{name_a}: {pass_a}/{n} = {pass_a/n*100:.1f}%   {name_b}: {pass_b}/{n} = {pass_b/n*100:.1f}%")
    print(f"diff ({name_b}-{name_a}): {(pass_b-pass_a)/n*100:.2f} points")
    print(f"contingency: both_pass={both_pass} {name_a}_only={a_only} {name_b}_only={b_only} both_fail={both_fail}")

    if a_only + b_only == 0:
        print("McNemar: no discordant pairs, test undefined")
    else:
        stat = (abs(a_only - b_only) - 1) ** 2 / (a_only + b_only)
        try:
            from scipy import stats as sstats

            pval = 1 - sstats.chi2.cdf(stat, df=1)
            exact_p = sstats.binomtest(min(a_only, b_only), a_only + b_only, 0.5).pvalue
            print(f"McNemar chi2={stat:.4f}, p={pval:.4f}, exact binomial p={exact_p:.4f}")
        except ImportError:
            z = math.sqrt(stat)
            pval = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
            print(f"McNemar chi2={stat:.4f}, p(normal approx)={pval:.4f}")

    random.seed(seed)
    common_list = list(common)
    arr_a = [outcomes_a[k] for k in common_list]
    arr_b = [outcomes_b[k] for k in common_list]
    diffs = []
    for _ in range(bootstrap_n):
        idxs = [random.randrange(n) for _ in range(n)]
        diffs.append((sum(arr_b[i] for i in idxs) - sum(arr_a[i] for i in idxs)) / n)
    diffs.sort()
    lo = diffs[int(0.025 * bootstrap_n)]
    hi = diffs[int(0.975 * bootstrap_n)]
    mean = sum(diffs) / len(diffs)
    print(f"Bootstrap 95% CI for ({name_b}-{name_a}) diff: [{lo*100:.2f}, {hi*100:.2f}] points, mean={mean*100:.2f}")


if __name__ == "__main__":
    path_a, path_b = sys.argv[1], sys.argv[2]
    name_a = sys.argv[3] if len(sys.argv) > 3 else "A"
    name_b = sys.argv[4] if len(sys.argv) > 4 else "B"
    compare(name_a, load_outcomes(path_a), name_b, load_outcomes(path_b))
