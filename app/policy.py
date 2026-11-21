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
CPU_RATIO = 1.3  # our CPU seconds per recorded test second (setup + slower box margin)

# P(expensive test fails) by regime, from synthetic single-line mutants at the pinned commit
# (tools/campaign.py; smoothed (fails + 0.5) / (n + 2)).
#   broad : at least half of the cheap candidates failed  -> the change breaks a lot
#   narrow: some but under half failed
#   dead  : none failed; then P shrinks with the number of cheap candidates that passed
BROAD = 0.9
NARROW_DEFAULT = 0.2
NARROW = {
    "test_execute_tpcds": 0.23,
    "test_max_nodes": 0.10,
    "test_generator_memory": 0.06,
    "test_execute_tpch": 0.06,
    "test_optimized_tpch": 0.17,
    "test_agg_order": 0.06,
    "test_cache_does_not_retain_queries": 0.07,
    "test_merge_subqueries": 0.13,
    "test_optimize_in_nulls": 0.14,
    "test_canonicalize": 0.14,
    "test_qualify_columns": 0.54,
    "test_tpcds": 0.38,
    "test_annotate_funcs": 0.38,
    "test_scope": 0.23,
}
DEAD_BASE = 0.08
DEAD_FLOOR = 0.003


def prob(test_id, regime, n_cheap):
    name = test_id.rsplit("::", 1)[-1]
    if regime == "broad":
        return BROAD
    if regime == "narrow":
        return NARROW.get(name, NARROW_DEFAULT)
    if regime == "unknown":
        return 0.5
    return max(DEAD_FLOOR, DEAD_BASE / (1.0 + n_cheap / 5.0))


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


def decide(exp_tests, base, regime, n_cheap, ms, ids):
    """For each expensive test choose 'blind', 'verify' or 'skip' by expected cost.

    base: tests already in the output. Returns (blind, verify) sets.
    """
    blind, verify = set(), set()
    cur = set(base)
    cur_cost = pack_cost(cur, ms)
    for x in sorted(exp_tests, key=lambda t: -ms[t]):
        p = prob(ids[x], regime, n_cheap)
        m = pack_cost(cur | {x}, ms) - cur_cost
        c_blind = m
        c_verify = CPU_RATIO * ms[x] + p * m
        c_skip = MISS_MS * p
        best = min(c_blind, c_verify, c_skip)
        if best == c_blind:
            blind.add(x)
            cur.add(x)
            cur_cost += m
        elif best == c_verify:
            verify.add(x)
    return blind, verify


def early_stop_ok(cheap, exp_tests, ms, ids):
    """Stop running cheap tests at the first failure when the expensive decisions would be
    the same whether the change turns out broad or narrow, a big test goes on its own
    worker either way, and all cheap candidates fit next to it."""
    if not exp_tests or not cheap:
        return None
    b_broad, v_broad = decide(exp_tests, set(), "broad", len(cheap), ms, ids)
    b_narrow, v_narrow = decide(exp_tests, set(), "narrow", len(cheap), ms, ids)
    if not b_narrow or b_broad != b_narrow or v_broad != v_narrow:
        return None
    full = pack_cost(set(cheap) | b_narrow, ms)
    lean = pack_cost(b_narrow, ms)
    if full - lean <= 300:
        return b_narrow
    return None
