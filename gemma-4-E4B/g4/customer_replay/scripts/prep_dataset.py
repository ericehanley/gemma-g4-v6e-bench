#!/usr/bin/env python3
"""Render the customer's prompt template over safechat-100k, inside the benchmark pod.

Input: /data/safechat.jsonl streamed into the pod's tmpfs by stream_dataset.py (one JSON
object per row: {"i": row, "lang": ..., "v": is_violative, "c": [[speaker, text], ...]}).

Output (tmpfs): prompts_bracket.json  "[s1]: text" lines, the format in the customer's email
                prompts_plain.json    "s1: text" lines, the format of the dataset's content_str
                labels.json           is_violative per row, same order
Output (results): dataset_stats.json  aggregate numbers only, no text.

Template, verbatim from the customer:
    <instruction block>
    Chat context:
    [s1]: hi
    [t]: you are so stupid lol
    You only need to analyze this user's messages in the context above:
    [t]: you are so stupid lol
"""
import argparse
import collections
import glob
import json
import os
import statistics

INSTR = (
    "In the following chat messages from target speaker t and possibly other speakers\n"
    "s1, s2, etc., detect abuse by speaker t.\n"
    "Output 'Safe []' if there are no violations, or 'Unsafe [\"category1\",\n"
    "\"category2\", ...]' with an alphabetically sorted list of all violated categories.\n"
    "In all cases (for both safe and unsafe decisions), follow the verdict with\n"
    "'Rationale: <rationale>' explaining your decision.\n"
    "Chat context:\n"
)
FOCUS = "You only need to analyze this user's messages in the context above:\n"


def render(turns, bracket):
    fmt = "[{0}]: {1}" if bracket else "{0}: {1}"
    ctx = "\n".join(fmt.format(spk, txt) for spk, txt in turns)
    tgt = "\n".join(fmt.format(spk, txt) for spk, txt in turns if spk == "t")
    return INSTR + ctx + "\n" + FOCUS + tgt


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1))))]


def summary(xs):
    return {"n": len(xs), "mean": round(statistics.fmean(xs), 1), "p10": pct(xs, 10), "p50": pct(xs, 50),
            "p90": pct(xs, 90), "p99": pct(xs, 99), "max": max(xs), "min": min(xs)}


def find_tokenizer(hf_home):
    hits = glob.glob(os.path.join(hf_home, "hub", "models--google--gemma-4-*", "snapshots", "*", "tokenizer.json"))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/data/safechat.jsonl")
    ap.add_argument("--data-dir", default="/data")
    ap.add_argument("--stats", default="/workspace/output/dataset_stats.json")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", "/hf-cache"))
    args = ap.parse_args()

    rows = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    convs = [[(t[0], t[1]) for t in r["c"]] for r in rows]
    labels = [bool(r.get("v")) for r in rows]
    langs = [r.get("lang") for r in rows]
    bracket = [render(c, True) for c in convs]
    plain = [render(c, False) for c in convs]

    for name, data in (("prompts_bracket.json", bracket), ("prompts_plain.json", plain), ("labels.json", labels)):
        with open(os.path.join(args.data_dir, name), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    # Does file order put growing windows of the same chat next to each other? If so, a
    # file-order replay would share long cached prefixes. Checked on the chat-context text.
    ctx_bracket = [render(c, True)[len(INSTR):].split("\n" + FOCUS)[0] for c in convs]
    consecutive_prefix = sum(
        1 for a, b in zip(ctx_bracket, ctx_bracket[1:]) if a.startswith(b) or b.startswith(a)
    )
    stats = {
        "rows": len(rows),
        "duplicate_prompts": len(bracket) - len(set(bracket)),
        "consecutive_rows_sharing_context_prefix": consecutive_prefix,
        "prompt_chars_bracket": summary([len(p) for p in bracket]),
        "prompt_chars_plain": summary([len(p) for p in plain]),
        "turns_per_conversation": summary([len(c) for c in convs]),
        "target_turns_per_conversation": summary([sum(1 for s, _ in c if s == "t") for c in convs]),
        "is_violative_rate": round(sum(labels) / len(labels), 4) if labels else None,
        "languages_top": collections.Counter(langs).most_common(10),
    }

    tok_path = find_tokenizer(args.hf_home)
    if tok_path:
        try:
            from tokenizers import Tokenizer

            tk = Tokenizer.from_file(tok_path)
            for name, prompts in (("bracket", bracket), ("plain", plain)):
                enc = tk.encode_batch(prompts, add_special_tokens=True)
                lens = [len(e.ids) + 9 for e in enc]  # +9: Gemma chat wrapper for one user turn
                stats[f"prompt_tokens_with_chat_template_{name}"] = summary(lens)
                stats[f"frac_over_512_{name}"] = round(sum(1 for n in lens if n > 512) / len(lens), 4)
                stats[f"frac_over_8192_{name}"] = round(sum(1 for n in lens if n > 8192) / len(lens), 6)
            stats["shared_instruction_prefix_tokens"] = len(tk.encode(INSTR, add_special_tokens=True).ids)
            stats["tokenizer"] = tok_path
        except Exception as e:  # noqa: BLE001
            stats["tokenizer_error"] = repr(e)
    else:
        stats["tokenizer"] = None

    os.makedirs(os.path.dirname(args.stats), exist_ok=True)
    with open(args.stats, "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "languages_top"}, indent=1))


if __name__ == "__main__":
    main()
