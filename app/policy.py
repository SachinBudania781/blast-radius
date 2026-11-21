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

DEFAULT = (0.7, 0.05)
PRIOR = {}


def prior(test_id):
    name = test_id.rsplit("::", 1)[-1]
    return PRIOR.get(name, DEFAULT)


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


def decide(exp_tests, base, live, ms, ids):
    """For each expensive test choose 'blind', 'verify' or 'skip'.

    base: tests already in the output. Returns (blind, verify) sets.
    """
    blind, verify = set(), set()
    cur = set(base)
    cur_cost = pack_cost(cur, ms)
    for x in sorted(exp_tests, key=lambda t: -ms[t]):
        p_live, p_dead = prior(ids[x])
        p = p_live if live else p_dead
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
    """If the change turns out live, would we put a big test on its own worker anyway, with
    all cheap candidates fitting next to it? Then stop running cheap tests at the first failure."""
    if not exp_tests or not cheap:
        return None
    blind, _ = decide(exp_tests, set(), True, ms, ids)
    if not blind:
        return None
    full = pack_cost(set(cheap) | blind, ms)
    lean = pack_cost(blind, ms)
    if full - lean <= 300:
        return blind
    return None
