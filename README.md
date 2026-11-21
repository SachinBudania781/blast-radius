# Blast Radius selector

For every patch to sqlglot (pinned at `eb11aec`), decide which tests can fail, run the
cheap ones against the patched code to see which really do, and print the failing ones
packed onto the cheapest number of workers.

## Maps (built once at the pinned commit, committed in `maps/`)

* `tools/br_mapper.py` runs the whole suite under `sys.monitoring` and records, per test,
  the source lines it executes, the classes whose methods it calls (type of `self`/`cls`),
  and the files it opens or lists. Process pools are replaced by an in-process pool that
  keeps the pickle round trip (`app/syncpool.py`), so work done in workers is attributed to
  the test. Lines executed while a module is being imported are recorded separately with the
  importing module; `setUpClass` lines and file reads are attributed to the whole class.
* `tools/compile_map.py` turns this into per-line candidate sets: statement coverage for
  function bodies (function-level coverage for structural edits); for class attributes the
  users of the class narrowed by runtime readers of the attribute; for module-level names the
  tests running code that references them; "module users" via a static reverse-import graph
  (used for import failures and deletions); readers for fixtures and other data files.
* `tools/tpc_query_map.py`, `tools/opt_query_map.py`: per-query coverage for the TPC executor
  tests and the optimizer fixture tests, so a change re-runs only the queries it can affect.

## Runtime (`app/selector.py`, `app/runner.py`, `app/policy.py`)

1. Parse each diff; comment/blank-only edits are skipped when provably harmless (not inside a
   string, not after a line continuation, no encoding cookie); docstrings likewise.
2. Look up candidates. For each commit fork a child that builds a symlink overlay of the
   patched tree in `/tmp`, checks the patched modules (and every dialect module) import, imports
   all needed test modules (like pytest collection) and runs the candidate tests with
   `unittest`. Every test reported failed, or never reported, is included; a test file that
   cannot be imported fails all of its tests and every file importing the patched module.
3. "Cheap" means under 1 s recorded *or* under 1 s of our own CPU (`maps/cpu.m`, measured
   at the pinned commit: some tests are slow in the recorded run only because of a process
   pool or class setup, e.g. `test_execute_tpch` 3.9 s recorded, 0.2 s for us). Expensive
   tests are decided by expected cost (run it, include it unrun, or leave it out) using
   measured CPU, how many cheap candidates failed (catastrophic / broad / narrow / none) and,
   in the narrow case, whether cheap tests of the expensive test's own class failed.
   Probabilities come from clean synthetic single-line mutants of the pinned commit
   (`tools/campaign.py`, `tools/validate.py`). A change that breaks almost everything stops
   early and includes all candidates.
   Proxies: for changes outside executor-side code, `test_execute_tpch` and the optimizer's
   `test_tpch` / `test_tpcds` (same queries, much cheaper) decide the TPC executor tests;
   early stopping at the first failure is only used when a big test would be added even on
   the best evidence, or once one of those cheap TPC tests (run first) fails.
   The slowest test has a cheap stand-in that runs its own checks in the child:
   `test_max_nodes` (the same node-limit checks at 1/100 scale). A failed stand-in includes the
   test; a passed one leaves it out. Not used when the test's own file is edited.
   Imported objects are frozen out of the cyclic GC (parent before forking, child after
   importing the test modules), which cuts collection work during tests.
4. Output: the failing tests, packed by LPT over 1..8 workers to minimise
   busiest worker + 2 s per worker.

Build: `docker build .` downloads sqlglot at the pinned commit and verifies every file
against `build/manifest.txt` (git blob hashes), so the maps always match the code.
