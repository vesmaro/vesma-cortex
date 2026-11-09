#!/usr/bin/env python3
"""D1-monitor reading (a-retrial spec 6.2) — window (D0, D1], read-only.
D0 = 2026-10-28T20:39:47Z, D1 = 2026-11-04T20:39:47Z (D0+7d).
cn(D0) = |N(a) AND N(b)| 1-hop undirected on edges created <= D0.
"""
import json, sqlite3, collections

STORE = "file:/var/home/abyss/.vesma/data/vesma.db?mode=ro"
D0, D1 = "2026-10-28T20:39:47", "2026-11-04T20:39:47"

con = sqlite3.connect(STORE, uri=True)
cur = con.cursor()
rows = cur.execute("SELECT from_memory_id, to_memory_id, kind, created_at FROM memory_edges").fetchall()
con.close()

d0_edges = [(a, b) for a, b, k, t in rows if t[:19] <= D0]
win_edges = [(a, b, k) for a, b, k, t in rows if D0 < t[:19] <= D1]
kinds_d0 = collections.Counter(k for a, b, k, t in rows if t[:19] <= D0)
kinds_win = collections.Counter(k for a, b, k in win_edges)

# undirected adjacency on D0
adj = collections.defaultdict(set)
for a, b in d0_edges:
    adj[a].add(b); adj[b].add(a)

def cn(a, b):
    return len(adj[a] & adj[b])

cn_vals = [cn(a, b) for a, b, _ in win_edges]
buckets = collections.Counter()
for v in cn_vals:
    buckets["cn=0" if v == 0 else ("cn=1-2" if v <= 2 else "cn>=3")] += 1

out = {
    "store": "/var/home/abyss/.vesma/data/vesma.db (mode=ro)",
    "read_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    "window": f"({D0}, {D1}]",
    "d0_edges_reconstructed": len(d0_edges),
    "d0_kind_dist": dict(kinds_d0),
    "window_count": len(win_edges),
    "window_kind_dist": dict(kinds_win),
    "edge_kind_freeze": "PASS" if set(kinds_win) <= set(kinds_d0) else "FAIL",
    "cn_d0_buckets": dict(buckets),
    "cn_total_sum": sum(cn_vals),
    "min_gap_minutes_from_d1_boundary": None,
}
print(json.dumps(out, indent=1))
