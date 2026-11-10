"""Load test the forecast API. No dependencies beyond the project venv.

    python scripts/loadtest.py --url http://127.0.0.1:8000 --requests 2000 --concurrency 20

Prints throughput, p50/p95/p99 latency and the status mix. Used to produce the numbers quoted in
the README, so rerun it rather than trusting the old figure.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx


async def worker(client: httpx.AsyncClient, url: str, results: list[dict[str, object]]) -> None:
    while True:
        try:
            next(counter)
        except StopIteration:
            return
        started = time.perf_counter()
        try:
            response = await client.get(url)
            elapsed = (time.perf_counter() - started) * 1000
            results.append(
                {"status": response.status_code, "ms": elapsed, "bytes": len(response.content)}
            )
        except httpx.HTTPError as exc:  # pragma: no cover - network dependent
            results.append(
                {"status": 0, "ms": (time.perf_counter() - started) * 1000, "error": str(exc)}
            )


def _counter(total: int):  # noqa: ANN202 - small local helper
    return iter(range(total))


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--path", default="/v1/zones/161/forecast?slots=48")
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()

    global counter
    counter = _counter(args.requests)

    results: list[dict[str, object]] = []
    async with httpx.AsyncClient(timeout=10.0) as client:
        started = time.perf_counter()
        await asyncio.gather(
            *(worker(client, args.url + args.path, results) for _ in range(args.concurrency))
        )
        wall = time.perf_counter() - started

    latencies = sorted(float(item["ms"]) for item in results)
    statuses: dict[int, int] = {}
    for item in results:
        status = int(item["status"])  # type: ignore[arg-type]
        statuses[status] = statuses.get(status, 0) + 1

    def q(p: float) -> float:
        index = min(len(latencies) - 1, int(round(p * (len(latencies) - 1))))
        return latencies[index]

    print(f"requests      {len(results)}")
    print(f"concurrency   {args.concurrency}")
    print(f"wall          {wall:.2f} s")
    print(f"throughput    {len(results) / wall:,.0f} req/s")
    print(f"latency p50   {statistics.median(latencies):.2f} ms")
    print(f"latency p95   {q(0.95):.2f} ms")
    print(f"latency p99   {q(0.99):.2f} ms")
    print(f"latency max   {latencies[-1]:.2f} ms")
    print(f"status        {statuses}")
    return 0 if statuses.get(200, 0) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
