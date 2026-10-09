#!/usr/bin/env python3
"""Stream safechat-100k from GCS into the benchmark pod's tmpfs, without touching local disk.

Reads the 32 parquet shards into memory with your gcloud credential, converts them to a
compact JSONL (one conversation per line) and writes it to stdout, meant to be piped into
the pod:

    venv/bin/python stream_dataset.py | gzip -1 \\
      | kubectl -n rbx-bench exec -i <pod> -- sh -c 'gunzip > /data/safechat.jsonl'
    # then, after checking the row count in the pod, touch /data/READY

Only the conversation turns, language and is_violative label are forwarded.
"""
import argparse
import io
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq

BUCKET = "gs://gdm-rbx-text-safety-collab-sdbx/tmp/safechat-100k/"


def fetch(path, attempts=4):
    env = dict(os.environ, CLOUDSDK_METRICS_ENVIRONMENT="datacloud.jetski")
    for i in range(attempts):
        proc = subprocess.run(["gcloud", "storage", "cat", path], capture_output=True, env=env)
        if proc.returncode == 0 and proc.stdout[:4] == b"PAR1":
            return pq.ParquetFile(io.BytesIO(proc.stdout)).read(columns=["content", "is_violative", "text_language"])
        print(f"fetch {path} attempt {i + 1} failed: {proc.stderr.decode(errors='replace').strip()[-300:]}", file=sys.stderr)
        time.sleep(2 * (i + 1))
    raise SystemExit(f"could not fetch {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="rows per shard (0 = all), for smoke tests")
    args = ap.parse_args()

    paths = [f"{BUCKET}part-{i:05d}.parquet" for i in range(args.shards)]
    with ThreadPoolExecutor(8) as ex:
        tables = list(ex.map(fetch, paths))

    out = sys.stdout
    n = 0
    for t in tables:
        contents = t.column("content").to_pylist()
        viol = t.column("is_violative").to_pylist()
        langs = t.column("text_language").to_pylist()
        if args.limit:
            contents, viol, langs = contents[: args.limit], viol[: args.limit], langs[: args.limit]
        for c, v, lang in zip(contents, viol, langs):
            turns = [[x["speaker"], x["text"]] for x in json.loads(c)]
            out.write(json.dumps({"i": n, "lang": lang, "v": bool(v), "c": turns}, ensure_ascii=False) + "\n")
            n += 1
    out.flush()
    print(f"streamed {n} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
