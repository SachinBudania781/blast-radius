# Blast Radius selector

For every patch to sqlglot (pinned at `eb11aec`), decide which tests can fail and run
the cheap ones against the patched code to find the ones that really do.

How it works
1. **Maps (built once at the pinned commit, committed in `maps/`)**: every test was run
   under `sys.monitoring` (`tools/br_mapper.py`) recording, per test, the source lines it
   executes, the classes whose methods it calls (type of `self`/`cls`), and the files it
   opens. Process pools are replaced by an in-process pool that keeps the pickle round trip,
   so code run in workers is attributed to the test. `tools/compile_map.py` turns this
   into per-line candidate sets (statement coverage for function bodies; class users for
   class attributes; name references for module-level definitions; readers for fixtures).
2. **Runtime (`app/selector.py`)**: parse each diff, skip comment-only edits (checked
   token-safely), look up candidates, then fork a child per commit that imports the patched
   tree from a symlink overlay in `/tmp` and runs the candidate tests with `unittest`.
   Tests that fail (or crash/time out) are printed, packed onto the cheapest number of workers.

Build: `docker build .` downloads sqlglot at the pinned commit and verifies every file
against `build/manifest.txt` so the maps always match the code.
