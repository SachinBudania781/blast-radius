# Same base image the commits were labeled in (its Python decides test behaviour).
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

WORKDIR /app

# Exactly the packages of the labeling environment (duckdb, pandas, numpy, ...).
COPY requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir --disable-pip-version-check -r /app/requirements.lock

# sqlglot at the pinned commit, verified file by file against build/manifest.txt.
COPY build /app/build
RUN python /app/build/fetch_sqlglot.py /app/sqlglot

COPY app /app/app
COPY maps /app/maps

# DuckDB's answers to the TPC fixture queries at the pinned commit, keyed by exact SQL text
RUN python /app/build/precompute_duck.py /app/sqlglot /app/cache/duck.pkl

# Precompile so the runtime never compiles unchanged modules; make everything readable by nobody.
RUN python -m compileall -q -j 0 /app/sqlglot/sqlglot /app/sqlglot/tests /app/app \
    && chmod -R a+rX /app

ENV BR_SRC=/app/sqlglot \
    BR_DATA=/data \
    BR_MAPS=/app/maps \
    BR_TMP=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    BR_DUCK_CACHE=/app/cache/duck.pkl

CMD ["python", "/app/app/selector.py"]
