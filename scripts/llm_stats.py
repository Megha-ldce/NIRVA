#!/usr/bin/env python3
"""llm_stats.py - summarise the Nebius/Nemotron call log written by model_client.py.

Every model call is appended to ~/nirva_llm_calls.jsonl (override: NIRVA_LLM_LOG).
This prints per-kind latency, token use, chosen tools and errors, and can emit a
Markdown table to paste into the README / Devpost write-up.

    python3 scripts/llm_stats.py                 # default log
    python3 scripts/llm_stats.py path/to/log.jsonl --markdown
    python3 scripts/llm_stats.py --since 1760000000   # only calls after a unix time
"""
import argparse
import json
import os
import statistics
from collections import Counter, defaultdict


def pct(values, q):
    if not values:
        return None
    values = sorted(values)
    k = max(0, min(len(values) - 1, round(q * (len(values) - 1))))
    return values[k]


def load(path, since):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if since and r.get('ts', 0) < since:
                continue
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('log', nargs='?', default=os.environ.get(
        'NIRVA_LLM_LOG', os.path.expanduser('~/nirva_llm_calls.jsonl')))
    ap.add_argument('--since', type=float, default=0.0, help='unix timestamp lower bound')
    ap.add_argument('--markdown', action='store_true', help='print a Markdown table')
    args = ap.parse_args()

    if not os.path.exists(args.log):
        raise SystemExit(f'log not found: {args.log}')
    rows = load(args.log, args.since)
    if not rows:
        raise SystemExit('no calls in log')

    by_kind = defaultdict(list)
    for r in rows:
        by_kind[(r.get('kind'), r.get('model'))].append(r)

    table = []
    for (kind, model), rs in sorted(by_kind.items(), key=lambda kv: str(kv[0])):
        ok = [r for r in rs if not r.get('error')]
        lat = [r['latency_s'] for r in ok if r.get('latency_s') is not None]
        ptok = [r['prompt_tokens'] for r in ok if r.get('prompt_tokens') is not None]
        ctok = [r['completion_tokens'] for r in ok if r.get('completion_tokens') is not None]
        table.append({
            'kind': kind, 'model': model, 'calls': len(rs), 'errors': len(rs) - len(ok),
            'lat_med': statistics.median(lat) if lat else None, 'lat_p95': pct(lat, 0.95),
            'prompt_tok_med': statistics.median(ptok) if ptok else None,
            'completion_tok_med': statistics.median(ctok) if ctok else None,
            'total_tokens': sum(ptok) + sum(ctok),
        })

    def f(x, nd=2):
        return '-' if x is None else (f'{x:.{nd}f}' if isinstance(x, float) else str(x))

    if args.markdown:
        print('| Call | Model | Calls | Errors | Median latency (s) | p95 latency (s) | Median prompt tok | Median completion tok | Total tokens |')
        print('|---|---|---|---|---|---|---|---|---|')
        for t in table:
            print(f"| {t['kind']} | `{t['model']}` | {t['calls']} | {t['errors']} | {f(t['lat_med'])} | "
                  f"{f(t['lat_p95'])} | {f(t['prompt_tok_med'], 0)} | {f(t['completion_tok_med'], 0)} | {t['total_tokens']} |")
    else:
        for t in table:
            print(f"[{t['kind']}] {t['model']}: {t['calls']} calls, {t['errors']} errors, "
                  f"latency median {f(t['lat_med'])}s / p95 {f(t['lat_p95'])}s, "
                  f"tokens/call median prompt {f(t['prompt_tok_med'], 0)} completion {f(t['completion_tok_med'], 0)}, "
                  f"total {t['total_tokens']}")

    tools = Counter(r.get('tool') for r in rows if r.get('kind') == 'plan' and r.get('tool'))
    if tools:
        print('\nTool choices by the planner:')
        for name, n in tools.most_common():
            print(f'  {name:<22} {n}')
    errs = Counter(r.get('error') for r in rows if r.get('error'))
    if errs:
        print('\nErrors:')
        for e, n in errs.most_common(5):
            print(f'  {n}x {str(e)[:120]}')


if __name__ == '__main__':
    main()
