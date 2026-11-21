"""In-process stand-in for ProcessPoolExecutor that keeps the pickle round trip
a real process boundary performs (arguments, initargs, results, exceptions)."""

import concurrent.futures as _cf
import pickle as _pickle


def _rt(obj):
    return _pickle.loads(_pickle.dumps(obj))


class SyncPool:
    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **kwargs):
        self._init = (initializer, initargs)
        self._inited = False

    def _ensure_init(self):
        if not self._inited:
            self._inited = True
            fn, args = self._init
            if fn is not None:
                fn2, args2 = _rt((fn, args))
                fn2(*args2)

    def submit(self, fn, /, *args, **kwargs):
        fut = _cf.Future()
        try:
            self._ensure_init()
            fn2, args2, kwargs2 = _rt((fn, args, kwargs))
            result = fn2(*args2, **kwargs2)
            fut.set_result(_rt(result))
        except BaseException as exc:  # noqa: BLE001
            if isinstance(exc, (KeyboardInterrupt, SystemExit)) and not isinstance(exc, Exception):
                raise
            try:
                exc2 = _rt(exc)
            except Exception:
                exc2 = exc
            fut.set_exception(exc2)
        return fut

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        futs = [self.submit(fn, *a) for a in zip(*iterables)]
        return (f.result() for f in futs)

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
        return False


def install():
    _cf.ProcessPoolExecutor = SyncPool
    try:
        import concurrent.futures.process as _p

        _p.ProcessPoolExecutor = SyncPool
    except Exception:
        pass
