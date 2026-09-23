"""Download the verified Windows CUDA wheel in restartable HTTP ranges."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import time
import httpx

URL = "https://mirrors.nju.edu.cn/pytorch/whl/cu128/torch-2.11.0%2Bcu128-cp311-cp311-win_amd64.whl"
SIZE = 2753148611
SHA256 = "90ef0c2454e5296a9fb021ddd42252e4ce1abe2c0a4988a173ef90a6cded0bf5"
CHUNK = 8 * 1024 * 1024
ROOT = Path(__file__).resolve().parents[1] / ".download-cache" / "nju-cu128"


def fetch(index):
    start = index * CHUNK
    end = min(start + CHUNK, SIZE) - 1
    path = ROOT / f"part-{index:04d}"
    expected = end - start + 1
    if path.exists() and path.stat().st_size == expected:
        return expected
    for attempt in range(8):
        try:
            # Separate requests avoid throwing away a multi-GB stream on a reset.
            with httpx.Client(timeout=60, follow_redirects=True) as client:
                with client.stream("GET", URL, headers={"Range": f"bytes={start}-{end}",
                                                       "Accept-Encoding": "identity"}) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("content-range") != f"bytes {start}-{end}/{SIZE}":
                        raise ValueError("Server did not return the requested byte range")
                    with path.with_suffix(".partial").open("wb") as output:
                        count = 0
                        for data in response.iter_bytes():
                            count += len(data)
                            if count > expected:
                                raise ValueError("Oversized range")
                            output.write(data)
            if count != expected:
                raise ValueError("Truncated range")
            path.with_suffix(".partial").replace(path)
            return count
        except Exception as exc:
            print(f"retry part={index} attempt={attempt+1}: {type(exc).__name__}", flush=True)
            if attempt == 7:
                raise
            time.sleep(min(attempt + 1, 5))


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    total = 0
    count = (SIZE + CHUNK - 1) // CHUNK
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, index) for index in range(count)]
        for future in as_completed(futures):
            total += future.result()
            print(f"verified-size parts: {total}/{SIZE} bytes ({100*total/SIZE:.1f}%), elapsed={time.monotonic()-started:.0f}s", flush=True)
    target = ROOT / "torch-2.11.0+cu128-cp311-cp311-win_amd64.whl"
    digest = hashlib.sha256()
    partial = target.with_suffix(".partial")
    with partial.open("wb") as output:
        for index in range(count):
            with (ROOT / f"part-{index:04d}").open("rb") as source:
                while data := source.read(1024 * 1024):
                    digest.update(data)
                    output.write(data)
    if digest.hexdigest() != SHA256:
        raise RuntimeError("SHA256 mismatch; refusing installation")
    partial.replace(target)
    print(f"SHA256 OK: {target}", flush=True)


if __name__ == "__main__":
    main()
