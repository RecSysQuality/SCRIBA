"""
Enumerazione di biclique massimali in un grafo bipartito user-item.

Una biclique (U, I) e' un gruppo di utenti U e item I tale che OGNI utente di U
ha recensito OGNI item di I. Cerchiamo quelle con |U| >= min_users e
|I| >= min_items, e solo quelle MASSIMALI (non estendibili ne' in utenti ne'
in item): altrimenti da un gruppo di 4 utenti usciresti anche tutti i suoi
sottogruppi, un'esplosione combinatoria inutile.

Algoritmo: LCM / prefix-preserving closure extension (enumerazione dei closed
itemset). Ogni biclique massimale viene emessa esattamente una volta, senza
deduplica e senza tenere i risultati in memoria.

Uso:
    python maximal_bicliques.py edges.tsv --min-users 3 --min-items 2
dove edges.tsv ha una riga per arco: user_id<TAB>item_id
"""

from collections import defaultdict, Counter
import argparse
import sys


# ---------------------------------------------------------------- caricamento

def load_edges(path, sep="\t"):
    """Legge il file archi e restituisce user2items, item2users (dict di set)."""
    user2items = defaultdict(set)
    item2users = defaultdict(set)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            u, i = line.split(sep)[:2]
            user2items[u].add(i)
            item2users[i].add(u)
    return dict(user2items), dict(item2users)


def build_from_dict(user_dict):
    """Costruisce le due mappe da un dict {user: iterable_di_item}."""
    user2items = {u: set(items) for u, items in user_dict.items()}
    item2users = defaultdict(set)
    for u, items in user2items.items():
        for i in items:
            item2users[i].add(u)
    return user2items, dict(item2users)


# ------------------------------------------------------------ pre-filtraggio

def prune(user2items, item2users, min_users, min_items, verbose=True):
    """
    Riduzione iterativa (tipo k-core bipartito): butta via item con meno di
    min_users recensori e utenti con meno di min_items item, fino a punto fisso.
    Su dati reali questo passaggio toglie spesso l'80-95% degli archi ed e' la
    ragione principale per cui l'enumerazione poi regge.
    """
    changed = True
    while changed:
        changed = False

        dead_items = [i for i, us in item2users.items() if len(us) < min_users]
        for i in dead_items:
            for u in item2users.pop(i):
                user2items[u].discard(i)
            changed = True

        dead_users = [u for u, its in user2items.items() if len(its) < min_items]
        for u in dead_users:
            for i in user2items.pop(u):
                item2users[i].discard(u)
            changed = True

    if verbose:
        n_edges = sum(len(v) for v in user2items.values())
        print(f"[prune] {len(user2items)} utenti, {len(item2users)} item, "
              f"{n_edges} archi rimasti", file=sys.stderr)
    return user2items, item2users


# ------------------------------------------------------------- enumerazione

def maximal_bicliques(user2items, item2users, min_users=3, min_items=2):
    """
    Generatore di (frozenset_utenti, frozenset_item), biclique massimali con
    |U| >= min_users e |I| >= min_items.

    L'ordine totale sugli item serve solo a garantire l'unicita': uso un
    ranking interno (item piu' frequenti = rank piu' basso), che di solito
    pota prima e meglio.
    """
    rank = {i: r for r, i in enumerate(
        sorted(item2users, key=lambda x: (-len(item2users[x]), str(x))))}

    def item_counts(tid):
        """Quante volte ogni item compare fra gli utenti di tid."""
        c = Counter()
        for u in tid:
            c.update(user2items[u])
        return c

    def rec(prefix, tid, tail):
        """
        prefix: closed itemset corrente (set di item)
        tid:    insieme di utenti che li hanno TUTTI recensiti
        tail:   rank dell'ultimo item aggiunto (solo estensioni con rank >)
        """
        counts = item_counts(tid)
        # candidati: item non gia' nel prefix, con rank > tail, ancora frequenti
        cands = sorted((rank[i] for i, c in counts.items()
                        if c >= min_users and rank[i] > tail and i not in prefix))

        for r in cands:
            e = rank_inv[r]
            new_tid = tid & item2users[e]
            if len(new_tid) < min_users:
                continue

            # chiusura: tutti gli item recensiti da TUTTI gli utenti di new_tid
            nc = item_counts(new_tid)
            n = len(new_tid)
            closure = {i for i, c in nc.items() if c == n}

            # prefix-preserving check: se la chiusura tira dentro un item con
            # rank minore di e e non gia' nel prefix, questo closed itemset
            # verra' (o e' gia' stato) generato da un altro ramo -> salta.
            if any(rank[j] < r for j in closure if j not in prefix):
                continue

            if len(closure) >= min_items:
                yield frozenset(new_tid), frozenset(closure)

            yield from rec(closure, new_tid, r)

    rank_inv = {r: i for i, r in rank.items()}

    all_users = set(user2items)
    if len(all_users) < min_users:
        return
    # chiusura della radice: item comuni a tutti gli utenti (di solito vuota)
    root_counts = Counter()
    for u in all_users:
        root_counts.update(user2items[u])
    root = {i for i, c in root_counts.items() if c == len(all_users)}
    if len(root) >= min_items:
        yield frozenset(all_users), frozenset(root)
    yield from rec(root, all_users, -1)


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edges", help="file TSV: user<TAB>item per riga")
    ap.add_argument("--min-users", type=int, default=3)
    ap.add_argument("--min-items", type=int, default=2)
    ap.add_argument("--sep", default="\t")
    ap.add_argument("--limit", type=int, default=0,
                    help="fermati dopo N gruppi (0 = nessun limite)")
    ap.add_argument("--out", default="-", help="file di output, '-' = stdout")
    args = ap.parse_args()

    u2i, i2u = load_edges(args.edges, args.sep)
    print(f"[load] {len(u2i)} utenti, {len(i2u)} item, "
          f"{sum(len(v) for v in u2i.values())} archi", file=sys.stderr)

    u2i, i2u = prune(u2i, i2u, args.min_users, args.min_items)

    out = sys.stdout if args.out == "-" else open(args.out, "w")
    n = 0
    for users, items in maximal_bicliques(u2i, i2u, args.min_users, args.min_items):
        n += 1
        out.write(f"{len(users)}\t{len(items)}\t"
                  f"{','.join(map(str, sorted(users)))}\t"
                  f"{','.join(map(str, sorted(items)))}\n")
        if n % 100000 == 0:
            print(f"[run] {n} gruppi...", file=sys.stderr)
        if args.limit and n >= args.limit:
            print("[run] limite raggiunto, mi fermo", file=sys.stderr)
            break
    if out is not sys.stdout:
        out.close()
    print(f"[done] {n} gruppi massimali", file=sys.stderr)


