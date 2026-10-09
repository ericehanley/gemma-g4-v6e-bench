#!/usr/bin/env python3
"""Per-role CPU sampler for the benchmark container (stdlib only).

Every interval, reads utime+stime of every process in /proc and writes one CSV row per role:
    unix_ts, role, nproc, cores, max_proc_cores, host_busy_frac
cores is the CPU used by all processes of that role (1.0 = one core fully busy);
max_proc_cores is the busiest single process of the role. A single-threaded asyncio client
that is saturated shows max_proc_cores ~1.0 for role bench_client, which would mean the
client, not the server, limits the measured throughput.

Roles come from /proc/<pid>/cmdline; vLLM renames its processes with setproctitle
(VLLM::APIServer_N, VLLM::EngineCore_DPN, VLLM::DPCoordinator).
"""
import argparse
import os
import time

ROLES = (
    ("rbx_loadgen", "rbx_loadgen"),
    ("bench_client", "bench serve"),
    ("api_server", "APIServer"),
    ("engine_core", "EngineCore"),
    ("dp_coordinator", "DPCoordinator"),
    ("serve_main", "vllm serve"),
    ("tpu_info", "tpu-info"),
)


def role_of(cmd):
    for role, needle in ROLES:
        if needle in cmd:
            return role
    return "other"


def read_procs():
    procs = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/stat") as f:
                stat = f.read()
            # Fields after the parenthesised comm: state is field 3, utime 14, stime 15.
            fields = stat[stat.rfind(")") + 2:].split()
            ticks = int(fields[11]) + int(fields[12])
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="replace")
        except (OSError, ValueError, IndexError):
            continue
        procs[int(pid)] = (ticks, cmd)
    return procs


def read_host():
    with open("/proc/stat") as f:
        vals = [int(x) for x in f.readline().split()[1:]]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
    return sum(vals), idle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    clk = os.sysconf("SC_CLK_TCK")
    prev, prev_t, prev_host = read_procs(), time.time(), read_host()
    with open(args.out, "a", buffering=1) as out:
        out.write("unix_ts,role,nproc,cores,max_proc_cores,host_busy_frac\n")
        while True:
            time.sleep(args.interval)
            cur, now, host = read_procs(), time.time(), read_host()
            dt = now - prev_t
            agg = {}
            for pid, (ticks, cmd) in cur.items():
                if pid not in prev or dt <= 0:
                    continue
                used = (ticks - prev[pid][0]) / clk / dt
                if used < 0:
                    continue
                a = agg.setdefault(role_of(cmd), [0, 0.0, 0.0])
                a[0] += 1
                a[1] += used
                a[2] = max(a[2], used)
            tot, idle = host[0] - prev_host[0], host[1] - prev_host[1]
            busy = 1.0 - idle / tot if tot > 0 else 0.0
            for role, (n, cores, mx) in sorted(agg.items()):
                out.write(f"{now:.3f},{role},{n},{cores:.3f},{mx:.3f},{busy:.4f}\n")
            prev, prev_t, prev_host = cur, now, host


if __name__ == "__main__":
    main()
