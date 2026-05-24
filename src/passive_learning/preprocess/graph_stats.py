"""
Amazon Reviews — Graph Statistics
==================================
Computes cardinality, degree distribution, density, sparsity,
rating distribution and temporal metrics on a bipartite user-item graph.

Usage:
    python graph_stats.py reviews.csv [labels_baby_products.json]

Output: JSON written to stdout (or optional file path).
"""

import sys
import json
import pandas as pd
import numpy as np


# ── 0. LOAD ───────────────────────────────────────────────────────────────────



def graph_stats(df: pd.DataFrame,dataset: str,TOP_N: int = 10,k:int = 1) :
    out_path = f"./data/summary/{dataset}_{k}.json"
    df.columns = df.columns.str.strip()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["user_id", "item_id", "rating", "timestamp"])

    # ── helpers ───────────────────────────────────────────────────────────────────
    def series_stats(s: pd.Series) -> dict:
        pct = [50, 75, 90, 95, 99]
        return {
            "min":    round(float(s.min()), 4),
            "max":    round(float(s.max()), 4),
            "mean":   round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std":    round(float(s.std()), 4),
            "skew":   round(float(s.skew()), 4),
            "percentiles": {f"p{p}": round(float(np.percentile(s, p)), 2) for p in pct},
        }

    def top_bottom(series: pd.Series, n: int = TOP_N) -> dict:
        """Return top-n and bottom-n entities by value."""
        top = series.nlargest(n)
        bot = series.nsmallest(n)
        return {
            "highest": [{"id": k, "value": int(v)} for k, v in top.items()],
            "lowest":  [{"id": k, "value": int(v)} for k, v in bot.items()],
        }

    result: dict = {}

    # ── 1. CARDINALITY ────────────────────────────────────────────────────────────
    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()
    n_edges = len(df)
    n_dup   = int(df.duplicated(subset=["user_id", "item_id"]).sum())

    result["cardinality"] = {
        "users":           n_users,
        "items":           n_items,
        "total_nodes":     n_users + n_items,
        "interactions":    n_edges,
        "unique_pairs":    n_edges - n_dup,
        "duplicate_pairs": n_dup,
    }

    # ── 2. DENSITY & SPARSITY ─────────────────────────────────────────────────────
    max_edges = n_users * n_items
    density   = n_edges / max_edges if max_edges else 0.0
    sparsity  = 1.0 - density

    result["density_sparsity"] = {
        "max_possible_edges": max_edges,
        "density":            round(density, 8),
        "sparsity":           round(sparsity, 8),
        "density_pct":        round(density * 100, 6),
        "sparsity_pct":       round(sparsity * 100, 6),
    }

    # ── 3. DEGREE DISTRIBUTION ────────────────────────────────────────────────────
    user_degree = df.groupby("user_id")["item_id"].count()
    item_degree = df.groupby("item_id")["user_id"].count()

    user_cold = int((user_degree == 1).sum())
    item_cold = int((item_degree == 1).sum())

    result["degree_distribution"] = {
        "user_degree": {
            **series_stats(user_degree),
            "cold_start_count": user_cold,
            "cold_start_pct":   round(user_cold / n_users * 100, 2),
            "top_bottom":       top_bottom(user_degree),
        },
        "item_degree": {
            **series_stats(item_degree),
            "cold_start_count": item_cold,
            "cold_start_pct":   round(item_cold / n_items * 100, 2),
            "top_bottom":       top_bottom(item_degree),
        },
    }

    # ── 4. RATING DISTRIBUTION ────────────────────────────────────────────────────
    rating_dist     = df["rating"].value_counts().sort_index()
    user_avg_rating = df.groupby("user_id")["rating"].mean()
    item_avg_rating = df.groupby("item_id")["rating"].mean()

    stars_breakdown = {
        str(int(star)): {
            "count": int(cnt),
            "pct":   round(cnt / n_edges * 100, 2),
        }
        for star, cnt in rating_dist.items()
    }

    bins   = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5.0001]
    labels = ["1.0-1.4","1.5-1.9","2.0-2.4","2.5-2.9",
              "3.0-3.4","3.5-3.9","4.0-4.4","4.5-5.0"]

    def avg_rating_bins(avg_series, total):
        binned = pd.cut(avg_series, bins=bins, labels=labels, right=False)
        counts = binned.value_counts().reindex(labels).fillna(0).astype(int)
        return {
            lbl: {"count": int(cnt), "pct": round(cnt / total * 100, 2)}
            for lbl, cnt in counts.items()
        }

    def only_extreme_check(group_col):
        return int(
            df.groupby(group_col)["rating"]
              .apply(lambda r: r.isin([1, 5]).all())
              .sum()
        )

    result["rating_distribution"] = {
        "global": {
            **series_stats(df["rating"]),
            "by_star": stars_breakdown,
        },
        "user_avg_rating": {
            "mean":                round(float(user_avg_rating.mean()), 4),
            "std":                 round(float(user_avg_rating.std()), 4),
            "below_2_stars":       int((user_avg_rating < 2).sum()),
            "above_4_stars":       int((user_avg_rating > 4).sum()),
            "only_extreme_ratings": only_extreme_check("user_id"),
            "by_bin":              avg_rating_bins(user_avg_rating, n_users),
            "top_bottom":          top_bottom(user_avg_rating.round(4)),
        },
        "item_avg_rating": {
            "mean":                round(float(item_avg_rating.mean()), 4),
            "std":                 round(float(item_avg_rating.std()), 4),
            "below_2_stars":       int((item_avg_rating < 2).sum()),
            "above_4_stars":       int((item_avg_rating > 4).sum()),
            "only_extreme_ratings": only_extreme_check("item_id"),
            "by_bin":              avg_rating_bins(item_avg_rating, n_items),
            "top_bottom":          top_bottom(item_avg_rating.round(4)),
        },
        "rating_by_year": {
            str(yr): {str(int(star)): int(cnt) for star, cnt in row.items()}
            for yr, row in (
                df.assign(year=df["timestamp"].dt.year)
                  .groupby(["year", "rating"])
                  .size()
                  .unstack(fill_value=0)
                  .iterrows()
            )
        },
    }

    # ── 5. TEMPORAL DISTRIBUTION ─────────────────────────────────────────────────
    t_min = df["timestamp"].min()
    t_max = df["timestamp"].max()
    span  = t_max - t_min
    weeks = max(span.days / 7, 1)

    df["_year"] = df["timestamp"].dt.year
    by_year = df["_year"].value_counts().sort_index()

    user_time_range  = df.groupby("user_id")["timestamp"].agg(
        lambda x: (x.max() - x.min()).days
    )
    single_day_users = int((user_time_range == 0).sum())

    result["temporal"] = {
        "first_interaction":           t_min.isoformat(),
        "last_interaction":            t_max.isoformat(),
        "time_span_days":              span.days,
        "time_span_years":             round(span.days / 365.25, 2),
        "avg_interactions_per_week":   round(n_edges / weeks, 2),
        "avg_interactions_per_month":  round(n_edges / (span.days / 30.44), 2) if span.days else None,
        "users_active_single_day":     single_day_users,
        "users_active_single_day_pct": round(single_day_users / n_users * 100, 2),
        "volume_by_year": {
            str(yr): {"count": int(cnt), "pct": round(cnt / n_edges * 100, 2)}
            for yr, cnt in by_year.items()
        },
    }
    df.drop(columns=["_year"], inplace=True, errors="ignore")

    # ── 6. CONNECTIVITY (Union-Find) ──────────────────────────────────────────────
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[a] = b

    for _, row in df.iterrows():
        union(f"U_{row['user_id']}", f"I_{row['item_id']}")

    components: dict = {}
    for node in parent:
        components.setdefault(find(node), []).append(node)

    comp_sizes = sorted([len(v) for v in components.values()], reverse=True)
    n_comp     = len(comp_sizes)
    giant_frac = comp_sizes[0] / (n_users + n_items)

    result["connectivity"] = {
        "num_connected_components": n_comp,
        "giant_component_size":     comp_sizes[0],
        "giant_component_pct":      round(giant_frac * 100, 2),
        "isolated_nodes":           sum(1 for s in comp_sizes if s == 1),
        "fully_connected":          n_comp == 1,
        "top5_component_sizes":     comp_sizes[:5],
    }

    # ── 7. SUMMARY ────────────────────────────────────────────────────────────────
    result["summary"] = {
        "users":                n_users,
        "items":                n_items,
        "interactions":         n_edges,
        "density":              round(density, 8),
        "sparsity":             round(sparsity, 8),
        "avg_user_degree":      round(float(user_degree.mean()), 4),
        "avg_item_degree":      round(float(item_degree.mean()), 4),
        "global_avg_rating":    round(float(df["rating"].mean()), 4),
        "pct_5star":            round(float((df["rating"] == 5).mean() * 100), 2),
        "time_span_days":       span.days,
        "connected_components": n_comp,
    }

    # ── OUTPUT ────────────────────────────────────────────────────────────────────
    json_str = json.dumps(result, indent=2, ensure_ascii=False)

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"[graph_stats] Written to {out_path}")
    else:
        print(json_str)