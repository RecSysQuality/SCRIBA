#!/usr/bin/env python3
"""
Riprende il training di un modello "dopo la rimozione" dal suo checkpoint e lo porta avanti
fino al numero di epoche / early stopping della configurazione completa.

Uso (da src/recbole_f, su un nodo con GPU):
    python3 resume_train.py --dataset Books \
        --removed src/recbole_f/removed_defects_new_version_tndcg/<file removed_*.tsv> \
        [--epochs 300] [--stopping-step 10] [--rre-module run_removed_eval_1]

Cosa fa:
  1. ricostruisce il dataset ripulito ESATTAMENTE come il dump (staging dal file di rimozione attuale);
  2. carica il checkpoint esistente con trainer.resume_checkpoint (pesi, ottimizzatore, epoca,
     contatore dell'early stopping, best valid score);
  3. continua trainer.fit fino a --epochs (default: quelle della config) con lo stesso early stopping;
  4. se il training trova un modello migliore, il vecchio checkpoint viene rinominato in .pth.short,
     cosi' nella cartella resta un solo .pth (quello che il dump usera').
Il checkpoint di RecBole e' quello dell'epoca migliore in validazione: si riparte da li'.
"""
import argparse
import importlib
import re
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--removed", required=True, type=Path, help="file removed_*.tsv del metodo")
    p.add_argument("--epochs", type=int, default=None, help="epoche massime (default: config)")
    p.add_argument("--stopping-step", type=int, default=None, help="pazienza early stopping (default: config)")
    p.add_argument("--rre-module", default="run_removed_eval_1")
    a = p.parse_args()

    here = Path(__file__).resolve().parent
    for d in (here, here.parent):
        sys.path.insert(0, str(d))
    rre = importlib.import_module(a.rre_module)
    rc, mrd = rre.rc, rre.mrd
    from recbole.utils import get_trainer, init_seed
    from recbole.model.general_recommender import LightGCN  # noqa: F401  (solo per fallire subito se manca)

    ds = a.dataset
    sel = rre.selection_name(a.removed, ds)
    tag = re.sub(r"[^A-Za-z0-9_.]+", "_", f"{ds}__rm__{sel}")            # come run_removed_eval / dump
    saved_dir = rc.SAVED_DIR / tag / rre.MODEL
    old = rc.find_checkpoint(saved_dir) if saved_dir.exists() else None
    if old is None:
        sys.exit(f"nessun checkpoint in {saved_dir}")
    print(f"[{sel}] checkpoint da riprendere: {old}")

    # dataset ripulito dal file di rimozione attuale (come il dump)
    noise = rc.build_noise_registry(ds)["clean"]
    train_rows = rc._parse_inter(rc.DATASETS[ds] / noise["subdir"] / noise["levels"]["0.0"], want_rating=False)
    u2i = mrd.user_items(train_rows)
    try:
        defects = mrd.read_removed(a.removed, ds)
    except TypeError:
        defects = mrd.read_removed(a.removed)
    removed = set()
    for d in defects:
        removed.update(mrd.defect_edges(u2i, d))
    staging = Path(rc.STAGING_DIR)
    rre.stage_removed(ds, tag, [r for r in train_rows if (r[0], r[1]) not in removed], staging)
    env = rre.build_env(ds, tag, staging, saved_dir)
    cfg = env.cfg
    if a.epochs is not None:
        cfg["epochs"] = a.epochs
    if a.stopping_step is not None:
        cfg["stopping_step"] = a.stopping_step
    print(f"[{sel}] {len(removed)} archi rimossi; epochs={cfg['epochs']} stopping_step={cfg['stopping_step']}")

    init_seed(cfg["seed"], cfg["reproducibility"])
    model = rre.get_model(rre.MODEL)(cfg, env.train.dataset).to(cfg["device"])
    trainer = get_trainer(cfg["MODEL_TYPE"], cfg["model"])(cfg, model)
    trainer.resume_checkpoint(str(old))
    print(f"[{sel}] riparto dall'epoca {trainer.start_epoch} (best_valid={trainer.best_valid_score}, "
          f"epoche senza miglioramento={trainer.cur_step})")
    if trainer.start_epoch >= cfg["epochs"]:
        sys.exit(f"il checkpoint e' gia' all'epoca {trainer.start_epoch} >= epochs={cfg['epochs']}: "
                 "alza --epochs")
    best, _ = trainer.fit(env.train, env.valid, saved=True, show_progress=cfg["show_progress"])
    new = Path(trainer.saved_model_file)
    if new.exists() and new != old:
        old.rename(old.with_suffix(".pth.short"))
        print(f"[{sel}] nuovo checkpoint migliore: {new.name} (best_valid={best}); "
              f"vecchio rinominato in {old.name}.short")
    else:
        print(f"[{sel}] nessun miglioramento in validazione: resta {old.name}")


if __name__ == "__main__":
    main()