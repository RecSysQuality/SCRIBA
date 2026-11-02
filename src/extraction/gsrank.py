
from datetime import datetime
from collections import defaultdict
import argparse
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
if str(here) not in sys.path:
    sys.path.insert(0, str(here))
from features import prune, maximal_bicliques


GROUP_FEATURES = ["GTW", "GD", "GETF", "GSR", "GS", "GSUP"]
INDIV_FEATURES = ["IRD", "IETF", "IMC"]
POP_FEATURES = ["ITEM_POP", "USER_POP"]
ALL_FEATURES = GROUP_FEATURES + INDIV_FEATURES + POP_FEATURES

DEFAULT_TAU_DAYS = 2.87 * 30.44
DEFAULT_BETA_DAYS = 8.86 * 30.44

MAX_RATING_DEV = 4.0  # scala 1-5 -> deviazione massima 4


# ---------------------------------------------------------------- caricamento

def parse_day(s):
    """Data -> numero di giorni (float). Accetta ISO o timestamp unix."""
    s = s.strip()
    try:
        return float(s) / 86400.0
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s[:len(fmt) + 2].strip(), fmt).toordinal()
        except ValueError:
            continue
    raise ValueError(f"data non riconosciuta: {s!r}")


class ReviewData:

    def __init__(self):
        self.item_rev = defaultdict(dict)
        self.item_first = {}
        self.item_sum = defaultdict(float)
        self.item_cnt = defaultdict(int)
        self.user_cnt = defaultdict(int)

    def add(self, user, item, rating, day):
        self.item_rev[item][user] = (rating, day)
        self.item_sum[item] += rating
        self.item_cnt[item] += 1
        self.user_cnt[user] += 1

        if item not in self.item_first or day < self.item_first[item]:
            self.item_first[item] = day

    @classmethod
    def load(cls, path, sep="\t"):
        rd = cls()
        with open(path) as f:
            next(f)
            for ln, line in enumerate(f, 1):
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                parts = line.split(sep)
                if len(parts) < 4:
                    raise ValueError(f"riga {ln}: servono 4 colonne "
                                     f"(user, item, rating, data)")
                u, i, r, d = parts[:4]
                rd.add(u, i, float(r), parse_day(d))
        return rd


    def graph(self, exclude_items=None):
        """user2items / item2users per l'enumerazione, senza gli item esclusi."""
        exclude = exclude_items or set()
        item2users = {i: set(rev) for i, rev in self.item_rev.items()
                      if i not in exclude}
        user2items = defaultdict(set)
        for i, us in item2users.items():
            for u in us:
                user2items[u].add(i)
        return dict(user2items), item2users


    def group_days(self, users, item):
        return [self.item_rev[item][u][1] for u in users if u in self.item_rev[item]]

    def group_ratings(self, users, item):
        return [self.item_rev[item][u][0] for u in users if u in self.item_rev[item]]

    def others_avg_rating(self, users, item):
        """Media dei rating del prodotto data dai NON membri. None se non ce ne sono."""
        n_all, s_all = self.item_cnt[item], self.item_sum[item]
        g = self.group_ratings(users, item)
        n_oth, s_oth = n_all - len(g), s_all - sum(g)
        return (s_oth / n_oth) if n_oth > 0 else None


def popular_items(rd, pct):
    """Item con un numero di recensioni sopra il percentile `pct`."""
    if pct <= 0 or pct >= 100:
        return set()
    counts = sorted(rd.item_cnt.values())
    thr = counts[min(int(len(counts) * pct / 100), len(counts) - 1)]
    return {i for i, c in rd.item_cnt.items() if c >= thr}
def group_feature_values(users, items, rd, tau, beta, selected):
    """Feature di gruppo NON normalizzate globalmente (GS e GSUP restano grezze)."""
    out = {}
    need = lambda f: f in selected

    if need("GTW"):
        best = 0.0
        for p in items:
            days = rd.group_days(users, p)
            span = max(days) - min(days)
            best = max(best, 0.0 if span > tau else 1.0 - span / tau)
        out["GTW"] = best

    if need("GD"):
        best = 0.0
        for p in items:
            g = rd.group_ratings(users, p)
            oth = rd.others_avg_rating(users, p)
            if oth is None:      # nessun recensore esterno -> deviazione 0 (paper)
                continue
            best = max(best, abs(sum(g) / len(g) - oth) / MAX_RATING_DEV)
        out["GD"] = best

    if need("GETF"):
        best = 0.0
        for p in items:
            late = max(rd.group_days(users, p)) - rd.item_first[p]
            best = max(best, 0.0 if late > beta else 1.0 - late / beta)
        out["GETF"] = best

    if need("GSR"):
        out["GSR"] = sum(len(users) / rd.item_cnt[p] for p in items) / len(items)

    # grezzi: normalizzati dopo, sul massimo di tutti i gruppi trovati
    if need("GS"):
        out["_GS_raw"] = float(len(users))
    if need("GSUP"):
        out["_GSUP_raw"] = float(len(items))

    return out


def indiv_feature_values(users, items, rd, beta, selected):
    """
    Feature individuali per ogni membro. Restituisce {feature: {membro: valore}}.

    Nota su IMC (eq. 12): come scritta nel paper, vale ~0 quando il membro posta
    vicino alla media degli altri (accoppiamento STRETTO = sospetto) e cresce
    quando e' lontano. Usata dentro w3 premierebbe quindi i membri scollegati.
    Lascio la formula letterale e aggiungo IMC_coupling = 1 - IMC, che e' la
    versione "1 = sospetto" coerente con le altre feature: scegli tu quale usare.
    """
    res = {}

    if "IRD" in selected:
        d = {}
        for m in users:
            best = 0.0
            for p in items:
                oth = rd.others_avg_rating({m}, p)
                if oth is None:
                    continue
                r = rd.item_rev[p][m][0]
                best = max(best, abs(r - oth) / MAX_RATING_DEV)
            d[m] = best
        res["IRD"] = d

    if "IETF" in selected:
        d = {}
        for m in users:
            best = 0.0
            for p in items:
                late = rd.item_rev[p][m][1] - rd.item_first[p]
                best = max(best, 0.0 if late > beta else 1.0 - late / beta)
            d[m] = best
        res["IETF"] = d

    if "IMC" in selected:
        d = {}
        n = len(users)
        for m in users:
            vals = []
            for p in items:
                days = {u: rd.item_rev[p][u][1] for u in users}
                F, L = min(days.values()), max(days.values())
                span = L - F
                if span <= 0 or n < 2:
                    vals.append(0.0)     # tutti nello stesso giorno: coupling massimo
                    continue
                others_avg = sum(days[u] - F for u in users if u != m) / (n - 1)
                vals.append(abs((days[m] - F) - others_avg) / span)
            d[m] = sum(vals) / len(vals) if vals else 0.0
        res["IMC"] = d

    return res


def pop_feature_values(users, items, rd, selected):
    """
    Popolarita' di item e membri del gruppo: media, max e min del n. di
    recensioni nell'intero file (anche per gli item esclusi dall'enumerazione
    con --max-item-pct, che qui non compaiono comunque nei gruppi).
    """
    out = {}

    def agg3(prefix, vals):
        vals = [float(v) for v in vals]
        avg, mx, mn = (sum(vals) / len(vals), max(vals), min(vals)) if vals else (0.0, 0.0, 0.0)
        out[prefix + "_avg"], out[prefix + "_max"], out[prefix + "_min"] = avg, mx, mn

    if "ITEM_POP" in selected:
        agg3("ITEM_POP", (rd.item_cnt.get(p, 0) for p in items))
    if "USER_POP" in selected:
        agg3("USER_POP", (rd.user_cnt.get(u, 0) for u in users))
    return out

def aggregate(per_member):
    """Da {membro: valore} a (avg, max)."""
    v = list(per_member.values())
    return (sum(v) / len(v), max(v)) if v else (0.0, 0.0)



def score_groups(rd, min_users=3, min_items=2, features=None,
                 tau=DEFAULT_TAU_DAYS, beta=DEFAULT_BETA_DAYS,
                 max_groups=0, verbose=True, exclude_items=None):

    features = set(features or ALL_FEATURES)
    unknown = features - set(ALL_FEATURES)
    if unknown:
        raise ValueError(f"feature sconosciute: {sorted(unknown)}. "
                         f"Disponibili: {ALL_FEATURES}")


    u2i, i2u = rd.graph(exclude_items)
    u2i, i2u = prune(u2i, i2u, min_users, min_items, verbose=verbose)

    rows = []
    for users, items in maximal_bicliques(u2i, i2u, min_users, min_items):
        row = {"users": sorted(users), "items": sorted(items)}
        row.update(group_feature_values(users, items, rd, tau, beta, features))
        for fname, per_member in indiv_feature_values(users, items, rd, beta,
                                                      features).items():
            avg, mx = aggregate(per_member)
            row[fname + "_avg"] = avg
            row[fname + "_max"] = mx
            if fname == "IMC":
                row["IMC_coupling_avg"] = 1.0 - avg
        row.update(pop_feature_values(users, items, rd, features))
        rows.append(row)
        if verbose and len(rows) % 50000 == 0:
            print(f"[score] {len(rows)} gruppi...", file=sys.stderr)
        if max_groups and len(rows) >= max_groups:
            print("[score] limite gruppi raggiunto", file=sys.stderr)
            break


    for raw, name in (("_GS_raw", "GS"), ("_GSUP_raw", "GSUP")):
        if name not in features:
            continue
        mx = max((r[raw] for r in rows), default=1.0) or 1.0
        for r in rows:
            r[name] = r.pop(raw) / mx

    if verbose:
        print(f"[score] {len(rows)} gruppi valutati", file=sys.stderr)
    return rows


def feature_columns(rows):
    """Nomi delle colonne di feature presenti, in ordine canonico."""
    if not rows:
        return []
    order = (GROUP_FEATURES
             + [f + s for f in INDIV_FEATURES for s in ("_avg", "_max")]
             + ["IMC_coupling_avg"])
    present = set(rows[0]) - {"users", "items"}
    return [c for c in order if c in present]


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    #ap.add_argument("reviews", help="TSV: user<TAB>item<TAB>rating<TAB>data")
    ap.add_argument("--min-users", type=int, default=4) #4 #4
    ap.add_argument("--min-items", type=int, default=3) # 3 #5 yelp
    ap.add_argument("--features", default=",".join(ALL_FEATURES),
                    help=f"lista separata da virgole fra: {','.join(ALL_FEATURES)}")
    ap.add_argument("--tau-days", type=float, default=DEFAULT_TAU_DAYS,
                    help="finestra GTW in giorni (default: 2.87 mesi)")
    ap.add_argument("--beta-days", type=float, default=DEFAULT_BETA_DAYS,
                    help="soglia 'early' per GETF/IETF in giorni (default: 8.86 mesi)")
    ap.add_argument("--sort-by", default="",
                    help="colonna su cui ordinare in senso decrescente, "
                         "oppure 'SUM' per la somma delle feature selezionate")
    ap.add_argument("--max-groups", type=int, default=0)
    ap.add_argument("--sep", default=",")
    ap.add_argument("--out", default="-")
    ap.add_argument("--max-item-pct", type=float, default=80.0, # solo per beauty 95
                    help="esclude dall'enumerazione gli item sopra questo "
                         "percentile di recensioni (0 o 100 = nessun filtro)")
    args = ap.parse_args()
    datasets = [
        'Baby_Products',
        'Sports_and_Outdoors',
        'Office_Products',
        'Toys_and_Games',
        'Pet_Supplies',
        'Beauty_and_Personal_Care',
        'Books','yelp'
    ]
    datasets = ['Beauty_and_Personal_Care']
    datasets = ['Pet_Supplies']



    for dataset in datasets:
        print('DATASET', dataset)

        args.reviews = f"data/original/{dataset}/{dataset}.train.csv"
        args.out = f"data/original/{dataset}/{dataset}.group_ff.csv"

        feats = [f.strip() for f in args.features.split(",") if f.strip()]
        rd = ReviewData.load(args.reviews, args.sep)
        excl = popular_items(rd, args.max_item_pct)
        print(f"[filter] esclusi {len(excl)} item su {len(rd.item_cnt)} "
              f"(percentile {args.max_item_pct})", file=sys.stderr)

        rows = score_groups(rd, args.min_users, args.min_items, feats,
                            args.tau_days, args.beta_days, args.max_groups,
                            exclude_items=excl)



        # rows = score_groups(rd, args.min_users, args.min_items, feats,
        #                     args.tau_days, args.beta_days, args.max_groups)

        cols = feature_columns(rows)
        if args.sort_by == "SUM":
            rows.sort(key=lambda r: sum(r[c] for c in cols), reverse=True)
        elif args.sort_by:
            rows.sort(key=lambda r: r[args.sort_by], reverse=True)

        out = sys.stdout if args.out == "-" else open(args.out, "w")
        out.write("\t".join(["n_users", "n_items", "users", "items"] + cols) + "\n")
        for r in rows:
            out.write("\t".join(
                [str(len(r["users"])), str(len(r["items"])),
                 ",".join(map(str, r["users"])), ",".join(map(str, r["items"]))]
                + [f"{r[c]:.4f}" for c in cols]) + "\n")
        if out is not sys.stdout:
            out.close()


if __name__ == "__main__":
    main()