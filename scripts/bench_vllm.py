#!/usr/bin/env python3
"""Lightweight vLLM benchmark helper.

- Runs a warmup request.
- Optionally runs concurrent batches and reports aggregate throughput.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = q * (len(values) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    if hi == lo:
        return values[lo]
    w = idx - lo
    return values[lo] * (1 - w) + values[hi] * w


def _send_request(url: str, headers: dict, payload: str, timeout: int) -> Tuple[float, int]:
    req = urllib.request.Request(
        url,
        data=payload.encode(),
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace") if exc.fp else ""
        raise RuntimeError(f"HTTP error {exc.code}: {detail}") from exc
    elapsed = time.perf_counter() - started
    completion_tokens = result.get("usage", {}).get("completion_tokens", 0)
    return elapsed, completion_tokens


def _run_batch(
    *,
    url: str,
    headers: dict,
    payload: str,
    requests: int,
    concurrency: int,
    timeout: int,
) -> Tuple[float, List[float], int]:
    t0 = time.perf_counter()
    latencies: List[float] = []
    total_tokens = 0

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [
            pool.submit(_send_request, url=url, headers=headers, payload=payload, timeout=timeout)
            for _ in range(requests)
        ]
        for f in as_completed(futures):
            elapsed, tokens = f.result()
            latencies.append(elapsed)
            total_tokens += tokens

    wall_time = time.perf_counter() - t0
    return wall_time, latencies, total_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--prompt", default="Explain tensor parallelism concisely, then give one example.")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer local",
    }

    payload_obj = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    payload = json.dumps(payload_obj)

    # warmup
    wall_time, latencies, tokens = _run_batch(
        url=args.url,
        headers=headers,
        payload=payload,
        requests=1,
        concurrency=1,
        timeout=args.timeout,
    )
    print(f"warmup: wall={wall_time:.3f}s requests=1 completion_tokens={tokens} tok_per_s={tokens / wall_time:.1f}")

    for i in range(1, args.runs + 1):
        wall_time, lats, total_tokens = _run_batch(
            url=args.url,
            headers=headers,
            payload=payload,
            requests=args.requests,
            concurrency=args.concurrency,
            timeout=args.timeout,
        )
        p50 = _percentile(lats, 0.50)
        p95 = _percentile(lats, 0.95)
        avg = statistics.mean(lats) if lats else 0.0
        throughput = (total_tokens / wall_time) if wall_time else 0.0
        print(
            f"run{i}: wall={wall_time:.3f}s requests={args.requests} "
            f"completion_tokens={total_tokens} tok_per_s={throughput:.1f} "
            f"lat_avg={avg:.3f}s p50={p50:.3f}s p95={p95:.3f}s"
        )


if __name__ == "__main__":
    main()
