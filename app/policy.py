"""Cost-based decisions for expensive candidate tests.

A commit costs (busiest worker + 2 s per worker) + 300 s if a failing test is left out,
plus the CPU we spend running tests ourselves. Cheap candidates are always run first.
Whether one of them failed ("live" change) is strong evidence about the expensive ones.

PRIOR[test] = (P(fails | a cheap candidate failed), P(fails | all cheap candidates passed)).
Estimated by running every candidate on synthetic single-line mutants of the pinned
commit (see tools/campaign.py); unknown tests use DEFAULT.
"""

WORKER_MS = 2000
MISS_MS = 300000
CPU_RATIO = 1.3  # our CPU seconds per recorded test second (tests not in maps/cpu.m; with margin)
JOB_CPU_MS = 150  # fixed CPU of one test child (fork, overlay, imports)

# P(expensive test fails) by regime, measured by running every candidate on clean synthetic
# single-line mutants of the pinned commit (tools/campaign.py, tools/validate.py), smoothed.
#   cat    : >= 80% of the cheap candidates failed (the change breaks almost everything)
#   broad  : 30-80% failed
#   narrow : some, but under 30%, failed
#   dead   : none failed (weak evidence when only a few cheap tests cover the change)
CAT = 0.95
BROAD_MIN = 0.5
NARROW_DEFAULT = 0.2
NARROW = {
    "test_max_nodes": 0.04,
    "test_generator_memory": 0.04,
    "test_agg_order": 0.05,
    "test_annotate_funcs": 0.05,
    "test_cache_does_not_retain_queries": 0.05,
    "test_canonicalize": 0.05,
    "test_optimize_in_nulls": 0.04,
    "test_execute_tpch": 0.14,
    "test_scope": 0.11,
    "test_merge_subqueries": 0.18,
    "test_optimized_tpch": 0.18,
    "test_execute_tpcds": 0.23,
    "test_tpcds": 0.25,
    "test_qualify_columns": 0.25,
}
# dead regime: an expensive test "dedicated" to the changed file (it runs >= 40% of that
# file's covered lines, e.g. an optimizer rule's own fixture test) can catch what no cheap
# test does (seen in the data); unrelated expensive tests essentially never did (0 of ~100).
DEAD_DEDICATED = 0.3
# broad real-data tests: their targeted cheap siblings (executor tests) already passed
DEAD_DEDICATED_BY_TEST = {"test_execute_tpcds": 0.02, "test_execute_tpch": 0.02, "test_optimized_tpch": 0.02}  # 0/29, 0/9, 0/11
DEAD_DEFAULT = 0.005  # 0/176 non-dedicated expensive tests failed in the dead regime (clean data)
DEAD = {"test_max_nodes": 0.004, "test_generator_memory": 0.006}
WEAK_EVIDENCE = 5  # fewer cheap candidates than this: a clean pass says little


def prob(test_id, regime, n_cheap, dedicated=False):
    name = test_id.rsplit("::", 1)[-1]
    narrow = NARROW.get(name, NARROW_DEFAULT)
    if regime == "cat":
        return CAT
    if regime == "broad":
        return max(BROAD_MIN, narrow)
    if regime == "narrow":
        p = narrow if n_cheap >= WEAK_EVIDENCE else max(narrow, 0.3)
        return max(p, 0.3) if dedicated else p
    if regime == "dead":
        if n_cheap < WEAK_EVIDENCE:
            return 0.5
        return DEAD_DEDICATED_BY_TEST.get(name, DEAD_DEDICATED) if dedicated else DEAD.get(name, DEAD_DEFAULT)
    return 0.5  # unknown: nothing cheap to learn from


# narrow regime, refined by what the cheap tests of the expensive test's own class did and by
# the share of all cheap candidates that failed (validator + campaign mutants, smoothed):
#   same class all passed, <2% failing: 2/94 failed; 2-10%: 3/25; >=10%: 2/14
#   same class had a failure, <2%: 3/31; 2-10%: 36/90; >=10%: 27/38; no same-class cheap test: 0/7
NARROW_CELL = {"P<2": 0.03, "P<10": 0.12, "P>10": 0.2, "F<2": 0.12, "N": 0.08}
NARROW_F = {  # same class had a failure and >= 2% of cheap candidates failed
    "test_annotate_funcs": 0.65, "test_qualify_columns": 0.85, "test_merge_subqueries": 0.5,
    "test_canonicalize": 0.45, "test_tpcds": 0.65, "test_scope": 0.35, "test_execute_tpcds": 0.5,
    "test_optimized_tpch": 0.3, "test_optimize_in_nulls": 0.2, "test_agg_order": 0.15,
    "test_execute_tpch": 0.15, "test_cache_does_not_retain_queries": 0.1,
}


def narrow_prob(test_id, same_cls, cls_failed, frac):
    name = test_id.rsplit("::", 1)[-1]
    if not same_cls:
        return NARROW_CELL["N"]
    if not cls_failed:
        if frac < 0.02:
            p = NARROW_CELL["P<2"]
            if name in ("test_qualify_columns", "test_tpcds"):
                p = 0.1
            return p
        return NARROW_CELL["P<10"] if frac < 0.1 else NARROW_CELL["P>10"]
    if frac < 0.02:
        return NARROW_CELL["F<2"] if name != "test_scope" else 0.3
    return max(NARROW_F.get(name, 0.4), 0.7 if frac >= 0.1 else 0.0) if frac >= 0.1 else NARROW_F.get(name, 0.4)


def pack_cost(tests, ms):
    """Cost in ms of the best packing (LPT per worker count)."""
    if not tests:
        return 0
    items = sorted((ms[t] for t in tests), reverse=True)
    best = None
    for k in range(1, min(8, len(items)) + 1):
        loads = [0] * k
        for x in items:
            i = loads.index(min(loads))
            loads[i] += x
        cost = max(loads) + WORKER_MS * k
        if best is None or cost < best:
            best = cost
    return best


def decide(exp_tests, base, regime, n_cheap, ms, ids, run_ms=None, dedicated=frozenset(), pmap=None, cpu=None):
    """For each expensive test choose 'blind', 'verify' or 'skip' by expected cost.

    base: tests already in the output. pmap: probabilities that override prob().
    cpu: our measured CPU ms to run a test (else CPU_RATIO x recorded/filtered ms).
    Returns (blind, verify) sets.
    """
    blind, verify = set(), set()
    cur = set(base)
    cur_cost = pack_cost(cur, ms)
    for x in sorted(exp_tests, key=lambda t: -ms[t]):
        p = pmap[x] if pmap and x in pmap else prob(ids[x], regime, n_cheap, x in dedicated)
        m = pack_cost(cur | {x}, ms) - cur_cost
        c_blind = m
        c_verify = verify_cpu(x, ms, run_ms, cpu) + p * m
        c_skip = MISS_MS * p
        best = min(c_blind, c_verify, c_skip)
        if best == c_blind:
            blind.add(x)
            cur.add(x)
            cur_cost += m
        elif best == c_verify:
            verify.add(x)
    return blind, verify


def verify_cpu(x, ms, run_ms=None, cpu=None):
    if cpu and x in cpu:
        return cpu[x]
    return CPU_RATIO * (run_ms or {}).get(x, ms[x])


def early_stop_ok(cheap, exp_tests, ms, ids, run_ms=None, dedicated=frozenset(), cpu=None):
    """Stop running cheap tests at the first failure when the expensive decisions would be
    the same whether the change turns out broad or narrow, a big test goes on its own
    worker either way, and all cheap candidates fit next to it."""
    if not exp_tests or not cheap:
        return None
    b_cat, v_cat = decide(exp_tests, set(), "cat", len(cheap), ms, ids, run_ms, dedicated, cpu=cpu)
    b_broad, v_broad = decide(exp_tests, set(), "broad", len(cheap), ms, ids, run_ms, dedicated, cpu=cpu)
    b_narrow, v_narrow = decide(exp_tests, set(), "narrow", len(cheap), ms, ids, run_ms, dedicated, cpu=cpu)
    if not b_narrow or not (b_cat == b_broad == b_narrow) or not (v_cat == v_broad == v_narrow):
        return None
    full = pack_cost(set(cheap) | b_narrow, ms)
    lean = pack_cost(b_narrow, ms)
    if full - lean <= 300:
        return b_narrow
    return None
