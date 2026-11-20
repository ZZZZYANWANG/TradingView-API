from __future__ import annotations

import signal
import time

import run_research_v6_resilient as resilient
import run_research_v10_strict_coverage as strict

research = strict.research


class NetworkCallTimeout(TimeoutError):
    pass


def _alarm_handler(signum, frame):
    raise NetworkCallTimeout("network call exceeded hard timeout")


def hard_retry(call, *, label: str, attempts: int = 6):
    last = None
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    try:
        for attempt in range(1, attempts + 1):
            try:
                signal.alarm(75)
                value = call()
                signal.alarm(0)
                return value
            except Exception as exc:
                signal.alarm(0)
                last = exc
                print("HARD_RETRY", label, attempt, "/", attempts, repr(exc), flush=True)
                if attempt == attempts:
                    break
                try:
                    resilient._relogin()
                except Exception as login_exc:
                    print("RELOGIN_ERROR", label, repr(login_exc), flush=True)
                time.sleep(min(20.0, 3.0 * attempt))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    raise RuntimeError(f"{label} failed after {attempts} hard-timeout attempts") from last


resilient._retry = hard_retry


if __name__ == "__main__":
    research.main()
