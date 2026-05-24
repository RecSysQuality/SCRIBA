"""
Injection Configs v2
====================
Ricalibrate sulle 8 funzioni effettivamente chiamate in run_injection:

    inject_f1_burst, inject_f1_camouflage, inject_f1_hetero, inject_f1_delayed
    inject_f2_extreme, inject_f2_bias, inject_f2_late, inject_f2_highprod

Parametri che contano davvero
------------------------------
F1:
    f1_*_groups           → numero di campagne (loop count)
    f1_delayed_group_size → utenti per campagna delayed (users_per_wave = size // waves)
    f1_delayed_waves      → numero di ondate
    (group_size e items degli altri F1 sono hardcoded come randint nel codice)

F2:
    f2_*_users            → numero di spammer (loop count)
    f2_highprod_items     → UNICO parametro item realmente usato
    f2_bias_magnitude     → deviazione dal rating medio
    f2_late_days_after    → soglia temporale per recensore tardivo
    (f2_extreme_items, f2_bias_items, f2_late_items ignorati: randint(2,5) hardcoded)

Non chiamate → parametri ininfluenti: f1_history_*, f2_camouflage_*, f3_*, f4_*, f5_*

Target v2
---------
    Fake users : ~3.0% (range 2–5%)
    Nuovi archi: ~7.0% (range 5–10%)

Calibrazione
------------
    Correction factor (formula → reale) calcolato per dataset:
        Baby 2.35 | Office 2.35 | Sports 2.48 | Toys 2.42 | Pet 2.16 | Books 1.77

Stime v2
--------
    baby_products    : ~3.0% users  | ~7.0% archi | ~81%  items
    office_products  : ~3.0% users  | ~7.0% archi | ~53%  items
    sports_outdoors  : ~3.0% users  | ~7.3% archi | ~51%  items
    toys_games       : ~3.0% users  | ~7.0% archi | ~54%  items
    pet_supplies     : ~3.0% users  | ~7.0% archi | ~100% items (catalogo saturo)
    books            : ~3.0% users  | ~7.0% archi | ~67%  items
"""
from dataclasses import dataclass


@dataclass
class InjectionConfig:
    """
    Tutti i parametri dell'injection in un unico posto.
    Modifica qui per scalare su dataset più grandi o più piccoli.
    """
    contested_pool_pct: float = 0.15

    # ratio per casistica (frazione di target da contested pool)
    f1_burst_contested:     float = 0.25
    f1_history_contested:   float = 0.20
    f1_camouflage_contested:float = 0.15
    f1_delayed_contested:   float = 0.25  # spesso stesso target del burst
    f1_hetero_contested:    float = 0.10

    f2_extreme_contested:   float = 0.30
    f2_bias_contested:      float = 0.20
    f2_highprod_contested:  float = 0.35
    f2_late_contested:      float = 0.10  # basso — timing diverso

    f3_echo_ext_contested:  float = 0.40  # solo sui link esterni
    f3_sybil_contested:     float = 0.15


    seed: int = 42  # seme per riproducibilità
    f1_burst_min_reviews_pct: float = 0.25  # P25 — sopra la coda rumorosa
    f1_burst_max_reviews_pct: float = 0.80  # P80 — sotto i best-seller
    # ── Famiglia 1 — Gruppi coordinati ────────────────────────────

    # Burst puro: gruppo che recensisce gli stessi item
    # nella stessa finestra temporale con rating simili.
    # Rilevato da: FraudAR (co-occorrenza), GSDB (burst temporale)
    f1_burst_groups:        int = 5    # quanti gruppi iniettare
    f1_burst_group_size:    int = 8    # utenti per gruppo
    f1_burst_items:         int = 4     # item target per gruppo
    f1_burst_window_hours:  int = 48    # finestra temporale del burst
    f1_burst_rating:        int = 5
    f2_bridge: int  = 40,
    f2_shilling: int =200,
    f2_high_dege: int =60,

    # History condivisa: il gruppo condivide uno storico di item
    # prima di attaccare il target — simula una community organica.
    # Rilevato da: FraudAR (Jaccard similarity alta tra utenti)
    f1_history_groups:        int = 5
    f1_history_group_size:    int = 15
    f1_history_shared_items:  int = 6   # item in comune nello storico
    f1_history_target_items:  int = 4   # item target del burst
    f1_history_target_rating: int = 5,
    f1_history_shared_rating: int = 4,
    # Gruppo con camouflage: come burst puro ma ogni membro aggiunge
    # recensioni casuali per abbassare i propri indicatori individuali.
    # Rilevato da: FraudAR, GSDB (segnale parzialmente diluito)
    f1_camouflage_groups:     int = 15
    f1_camouflage_group_size: int = 13
    f1_camouflage_target:     int = 5   # item target
    f1_camouflage_noise:      int = 12  # item casuali come cover
    f1_camouflage_target_rating: int = 5

    # Burst ritardato: il gruppo agisce in ondate separate invece
    # che tutte insieme — evita il picco temporale singolo.
    # Rilevato da: GSDB (burst multipli con stessa firma)
    f1_delayed_groups:      int = 10
    f1_delayed_group_size:  int = 15
    f1_delayed_waves:       int = 3     # numero di ondate separate
    f1_delayed_target_rating:       int = 5     # numero di ondate separate

    # Gruppo eterogeneo: rating diversi tra i membri (3-5 stelle)
    # sullo stesso item — simula disaccordo naturale ma mantiene
    # la co-occorrenza.
    # Rilevato da: FraudAR (co-occorrenza alta, rating variabili)
    f1_hetero_groups:       int = 10
    f1_hetero_group_size:   int = 10
    f1_hetero_target_items: int = 6

    # ── Famiglia 2 — Spammer individuali ──────────────────────────

    # Rating estremi: utente che dà solo 1 o 5 stelle.
    # Distribuzione bimodale — entropia bassissima.
    # Rilevato da: GSDB (distribuzione rating), REV2 (fairness score)
    f2_extreme_users: int = 500
    f2_extreme_items: int = 7   # item recensiti per utente

    # Bias sistematico: valuta sempre sopra/sotto la media dell'item
    # di una quantità fissa. Gonfia o affossa artificialmente la
    # reputazione degli item target.
    # Rilevato da: REV2 (deviazione costante dalla media ponderata)
    f2_bias_users:     int = 500
    f2_bias_items:     int = 10
    f2_bias_magnitude: float = 1.8  # entità della deviazione dalla media

    # Alta produttività: numero di recensioni anomalo rispetto alla
    # media — centinaia di review in poco tempo.
    # Rilevato da: GSDB (velocità recensioni), REV2 (grado alto)
    f2_highprod_users: int = 300
    f2_highprod_items: int = 35   # item recensiti (alto grado)
    f2_highprod_rating: int = 5
    # Spammer con camouflage: alterna review spam su item target
    # con review genuine. Rapporto spam:legit = 1:4.
    # Rilevato da: GSDB, REV2 (segnale diluito — hard case)
    f2_camouflage_users:  int = 200
    f2_camouflage_target: int = 5   # item target anomali
    f2_camouflage_legit:  int = 20  # item casuali come cover
    f2_camouflage_rating: int = 5
    # Recensore tardivo: recensisce item molto dopo il lancio,
    # quando le review genuine sono stabilizzate.
    # Rilevato da: GSDB (timestamp anomalo rispetto alla distribuzione)
    f2_late_users:        int = 200
    f2_late_items:        int = 7
    f2_late_days_after:   int = 180  # recensisce negli ultimi N giorni
    f2_late_rating: int = 5,

    # ── Famiglia 3 — Attacchi strutturali ─────────────────────────

    # Hub injection: nodo con grado altissimo connesso a moltissimi
    # item popolari — massimizza la propagazione dello spam nella GNN.
    # Specifico per GNN: centralità alta nel grafo bipartito
    f3_hub_users: int = 100
    f3_hub_items: int = 75    # item recensiti (grado molto alto)

    # Bridge injection: connette cluster di utenti altrimenti separati
    # — altera la struttura comunitaria del grafo.
    # Specifico per GNN: betweenness alta, structural hole
    f3_bridge_users:      int = 50
    f3_bridge_items_per_cl: int = 8  # item per cluster connesso

    # Sybil attack: tanti account falsi con pochissime recensioni
    # ciascuno — difficili individualmente ma impattanti collettivamente.
    # Rilevato da: FraudAR (tanti nodi a basso grado con stesso target)
    f3_sybil_groups:     int = 250
    f3_sybil_group_size: int = 2    # solo 2 utenti per gruppo
    f3_sybil_items:      int = 3    # pochissime review ciascuno
    f3_sybil_rating:     int = 5
    # Echo chamber: gruppo chiuso con alta densità interna e pochi
    # link verso utenti legittimi — si rinforza internamente.
    # Rilevato da: FraudAR (community detection)
    f3_echo_groups:       int = 10
    f3_echo_group_size:   int = 20
    f3_echo_shared_items: int = 10  # item condivisi internamente
    f3_echo_rating: int=5,
    # ── Famiglia 4 — Camouflage avanzato (SOLO TEST SET) ──────────
    # Questi casi vanno solo nel test set — sono gli hard cases
    # che misurano il limite superiore del regressor.

    # Near-legitimate: quasi identico a un utente legittimo —
    # solo il target item ha rating anomalo. Sfida tutti i metodi.
    f4_nearlegit_users: int = 100

    # Gradual drift: inizialmente legittimo, diventa spam
    # gradualmente nel tempo — nessun punto di rottura netto.
    # Parzialmente rilevato da: GSDB (deriva temporale)
    f4_drift_users: int = 100

    # Copy-cat reviewer: copia il pattern di un utente reale
    # e aggiunge solo un target anomalo.
    # Non rilevato da nessun metodo standard.
    f4_copycat_users: int = 100

    # ── Famiglia 5 — Nodi legittimi ───────────────────────────────
    # Campionati dal dataset reale — danno label negativa al regressor.
    # Senza questi il regressor non impara cosa significa un nodo utile.

    f5_poweruser_n:     int = 1000  # utenti molto attivi
    f5_niche_n:         int = 1000  # utenti di nicchia (poche review)
    f5_earlyadopter_n:  int = 500   # utenti con timestamp precoce

    # ── Parametri globali ─────────────────────────────────────────
    popular_top_pct: float = 0.10   # top 10% item per popolarità
# ══════════════════════════════════════════════════════════════════
# BABY PRODUCTS  — 151K users, 36K items, 1.25M edges
# users ≈ 4 535 (3.0%)  |  archi ≈ 7.0%  |  items ≈ 81%
# Leva: +82% gruppi F1, +82% users F2, highprod_items 18 → 29
# ══════════════════════════════════════════════════════════════════
cfg_baby_products = InjectionConfig(
    seed=42,

    # ── F1 ── parametri che contano: solo i *_groups e delayed size/waves
    f1_burst_groups         = 100,
    f1_burst_window_hours   = 24,
    f1_burst_rating         = 5,


    f1_camouflage_groups        = 100,

    f1_delayed_groups       = 100,
    f1_delayed_target_rating= 5,

    f1_hetero_groups        = 100,
    f1_hetero_group_size    = 10,   # non usato
    f1_hetero_target_items  = 3,    # non usato

    # ── F2 ── highprod_items è l'unico param item effettivo
    f2_bridge  = 20,
    f2_shilling=100,
    f2_high_dege=30,

    f2_bias_users     = 100,
    f2_bias_items     = 6,          # non usato
    f2_bias_magnitude = 2.0,        # USATO

    f2_highprod_users  = 100,
    f2_highprod_items  = 29,        # USATO: 18 → 29

    f2_late_users       = 100,
    f2_late_items       = 5,        # non usato
    f2_late_days_after  = 180,      # USATO
    f2_late_rating      = 5,

    # ── F3 / F4 / F5 — non chiamate ──────────────────────────────
    f3_hub_users=0, f3_hub_items=0,
    f3_bridge_users=0, f3_bridge_items_per_cl=0,
    f3_sybil_groups=0, f3_sybil_group_size=2, f3_sybil_items=0, f3_sybil_rating=5,
    f3_echo_groups=0, f3_echo_group_size=2, f3_echo_shared_items=0, f3_echo_rating=5,
    f4_nearlegit_users=0, f4_drift_users=0, f4_copycat_users=0,
    f5_poweruser_n=0, f5_niche_n=0, f5_earlyadopter_n=0,

    popular_top_pct = 0.10,
)


# ══════════════════════════════════════════════════════════════════
# OFFICE PRODUCTS — 226K users, 78K items, 1.80M edges
# users ≈ 6 786 (3.0%)  |  archi ≈ 7.0%  |  items ≈ 53%
# Leva: +83% gruppi F1, +81% users F2, highprod_items 22 → 27
# ══════════════════════════════════════════════════════════════════
cfg_office_products = InjectionConfig(
    seed=42,

    f1_burst_groups         = 30,
    f1_burst_group_size     = 10,
    f1_burst_items          = 3,
    f1_burst_window_hours   = 48,
    f1_burst_rating         = 5,

    f1_history_groups=0, f1_history_group_size=0,
    f1_history_shared_items=0, f1_history_target_items=0,
    f1_history_target_rating=5, f1_history_shared_rating=4,

    f1_camouflage_groups        = 60,
    f1_camouflage_group_size    = 10,
    f1_camouflage_target        = 4,
    f1_camouflage_noise         = 10,
    f1_camouflage_target_rating = 5,

    f1_delayed_groups       = 50,
    f1_delayed_group_size   = 10,
    f1_delayed_waves        = 3,
    f1_delayed_target_rating= 5,

    f1_hetero_groups        = 50,
    f1_hetero_group_size    = 10,
    f1_hetero_target_items  = 5,

    f2_extreme_users  = 1635,
    f2_extreme_items  = 6,

    f2_bias_users     = 1635,
    f2_bias_items     = 8,
    f2_bias_magnitude = 1.8,

    f2_highprod_users  = 981,
    f2_highprod_items  = 27,

    f2_camouflage_users=0, f2_camouflage_target=0, f2_camouflage_legit=0, f2_camouflage_rating=5,

    f2_late_users      = 654,
    f2_late_items      = 6,
    f2_late_days_after = 180,
    f2_late_rating     = 5,

    f3_hub_users=0, f3_hub_items=0,
    f3_bridge_users=0, f3_bridge_items_per_cl=0,
    f3_sybil_groups=0, f3_sybil_group_size=2, f3_sybil_items=0, f3_sybil_rating=5,
    f3_echo_groups=0, f3_echo_group_size=2, f3_echo_shared_items=0, f3_echo_rating=5,
    f4_nearlegit_users=0, f4_drift_users=0, f4_copycat_users=0,
    f5_poweruser_n=0, f5_niche_n=0, f5_earlyadopter_n=0,

    popular_top_pct = 0.10,
)


# ══════════════════════════════════════════════════════════════════
# SPORTS AND OUTDOORS — 412K users, 157K items, 3.49M edges
# users ≈ 12 361 (3.0%)  |  archi ≈ 7.3%  |  items ≈ 51%
# Leva: +77% gruppi F1, +77% users F2, highprod_items invariato (30)
# ══════════════════════════════════════════════════════════════════

cfg_all_beauty = InjectionConfig(
    seed=42,

    # ── F1 ── parametri che contano: solo i *_groups e delayed size/waves
    f1_burst_groups         = 5,
    f1_burst_group_size     = 10,   # non usato (randint nel codice)
    f1_burst_items          = 2,    # non usato
    f1_burst_window_hours   = 24,
    f1_burst_rating         = 5,

    f2_bridge=3,
    f2_shilling=5,
    f2_high_dege=5,

    f1_camouflage_groups        = 5,
    f1_camouflage_group_size    = 10,   # non usato
    f1_camouflage_target        = 3,    # non usato
    f1_camouflage_noise         = 8,    # non usato
    f1_camouflage_target_rating = 5,

    f1_delayed_groups       = 5,
    f1_delayed_group_size   = 10,   # USATO: users_per_wave = size // waves
    f1_delayed_waves        = 3,    # USATO
    f1_delayed_target_rating= 5,

    f1_hetero_groups        = 5,
    f1_hetero_group_size    = 10,   # non usato
    f1_hetero_target_items  = 3,    # non usato

    # ── F2 ── highprod_items è l'unico param item effettivo
    f2_extreme_users  = 5,
    f2_extreme_items  = 5,          # non usato

    f2_bias_users     = 5,
    f2_bias_items     = 6,          # non usato
    f2_bias_magnitude = 2.0,        # USATO

    f2_highprod_users  = 5,
    f2_highprod_items  = 29,        # USATO: 18 → 29

    f2_late_users       = 5,
    f2_late_items       = 5,        # non usato
    f2_late_days_after  = 180,      # USATO
    f2_late_rating      = 5,

    # ── F3 / F4 / F5 — non chiamate ──────────────────────────────
    f3_hub_users=0, f3_hub_items=0,
    f3_bridge_users=0, f3_bridge_items_per_cl=0,
    f3_sybil_groups=0, f3_sybil_group_size=2, f3_sybil_items=0, f3_sybil_rating=5,
    f3_echo_groups=0, f3_echo_group_size=2, f3_echo_shared_items=0, f3_echo_rating=5,
    f4_nearlegit_users=0, f4_drift_users=0, f4_copycat_users=0,
    f5_poweruser_n=0, f5_niche_n=0, f5_earlyadopter_n=0,

    popular_top_pct = 0.10,
)

cfg_sports_outdoors = InjectionConfig(
    seed=42,

    # ── F1 ── parametri che contano: solo i *_groups e delayed size/waves
    f1_burst_groups         = 200,
    f1_burst_group_size     = 10,   # non usato (randint nel codice)
    f1_burst_items          = 2,    # non usato
    f1_burst_window_hours   = 24,
    f1_burst_rating         = 5,


    f1_camouflage_groups        = 200,
    f1_camouflage_group_size    = 10,   # non usato
    f1_camouflage_target        = 3,    # non usato
    f1_camouflage_noise         = 8,    # non usato
    f1_camouflage_target_rating = 5,

    f1_delayed_groups       = 200,
    f1_delayed_group_size   = 10,   # USATO: users_per_wave = size // waves
    f1_delayed_waves        = 3,    # USATO
    f1_delayed_target_rating= 5,

    f2_bridge =  40,
    f2_shilling = 200,
    f2_high_dege = 60,

    f1_hetero_groups        = 200,
    f1_hetero_group_size    = 10,   # non usato
    f1_hetero_target_items  = 3,    # non usato

    # ── F2 ── highprod_items è l'unico param item effettivo
    f2_extreme_users  = 200,
    f2_extreme_items  = 5,          # non usato

    f2_bias_users     = 200,
    f2_bias_items     = 6,          # non usato
    f2_bias_magnitude = 2.0,        # USATO

    f2_highprod_users  = 200,
    f2_highprod_items  = 29,        # USATO: 18 → 29

    f2_late_users       = 200,
    f2_late_items       = 5,        # non usato
    f2_late_days_after  = 180,      # USATO
    f2_late_rating      = 5,

    # ── F3 / F4 / F5 — non chiamate ──────────────────────────────
    f3_hub_users=0, f3_hub_items=0,
    f3_bridge_users=0, f3_bridge_items_per_cl=0,
    f3_sybil_groups=0, f3_sybil_group_size=2, f3_sybil_items=0, f3_sybil_rating=5,
    f3_echo_groups=0, f3_echo_group_size=2, f3_echo_shared_items=0, f3_echo_rating=5,
    f4_nearlegit_users=0, f4_drift_users=0, f4_copycat_users=0,
    f5_poweruser_n=0, f5_niche_n=0, f5_earlyadopter_n=0,

    popular_top_pct = 0.10,
)


# ══════════════════════════════════════════════════════════════════
# TOYS AND GAMES — 435K users, 162K items, 3.88M edges
# users ≈ 13 055 (3.0%)  |  archi ≈ 7.0%  |  items ≈ 54%
# Leva: +78% gruppi F1, +78% users F2, highprod_items 30 → 32
# ══════════════════════════════════════════════════════════════════
cfg_toys_games = InjectionConfig(
    seed=42,

    f1_burst_groups         = 41,
    f1_burst_group_size     = 10,
    f1_burst_items          = 5,
    f1_burst_window_hours   = 48,
    f1_burst_rating         = 5,

    f1_history_groups=0, f1_history_group_size=0,
    f1_history_shared_items=0, f1_history_target_items=0,
    f1_history_target_rating=5, f1_history_shared_rating=4,

    f1_camouflage_groups        = 124,
    f1_camouflage_group_size    = 12,
    f1_camouflage_target        = 5,
    f1_camouflage_noise         = 14,
    f1_camouflage_target_rating = 5,

    f1_delayed_groups       = 82,
    f1_delayed_group_size   = 12,
    f1_delayed_waves        = 3,
    f1_delayed_target_rating= 5,

    f1_hetero_groups        = 82,
    f1_hetero_group_size    = 10,
    f1_hetero_target_items  = 7,

    f2_extreme_users  = 3090,
    f2_extreme_items  = 6,

    f2_bias_users     = 3090,
    f2_bias_items     = 10,
    f2_bias_magnitude = 1.6,

    f2_highprod_users  = 1856,
    f2_highprod_items  = 32,

    f2_camouflage_users=0, f2_camouflage_target=0, f2_camouflage_legit=0, f2_camouflage_rating=5,

    f2_late_users      = 1236,
    f2_late_items      = 6,
    f2_late_days_after = 180,
    f2_late_rating     = 5,

    f3_hub_users=0, f3_hub_items=0,
    f3_bridge_users=0, f3_bridge_items_per_cl=0,
    f3_sybil_groups=0, f3_sybil_group_size=2, f3_sybil_items=0, f3_sybil_rating=5,
    f3_echo_groups=0, f3_echo_group_size=2, f3_echo_shared_items=0, f3_echo_rating=5,
    f4_nearlegit_users=0, f4_drift_users=0, f4_copycat_users=0,
    f5_poweruser_n=0, f5_niche_n=0, f5_earlyadopter_n=0,

    popular_top_pct = 0.10,
)


# ══════════════════════════════════════════════════════════════════
# PET SUPPLIES — 600K users, 115K items, 5.31M edges
# users ≈ 18 009 (3.0%)  |  archi ≈ 7.0%  |  items ≈ 100% (saturo)
# Leva: +77% gruppi F1, +77% users F2, highprod_items 28 → 39
# Nota: item space (115K) viene coperto interamente a questi volumi
# ══════════════════════════════════════════════════════════════════
cfg_pet_supplies = InjectionConfig(
    seed = 42,
    # ── F1 ── parametri che contano: solo i *_groups e delayed size/waves
    f1_burst_groups         = 250,
    f1_burst_window_hours   = 24,
    f1_burst_rating         = 5,


    f1_camouflage_groups        = 250,

    f1_delayed_groups       = 100,
    f1_delayed_target_rating= 5,

    f1_hetero_groups        = 100,
    f1_hetero_group_size    = 10,   # non usato
    f1_hetero_target_items  = 3,    # non usato

    # ── F2 ── highprod_items è l'unico param item effettivo
    f2_bridge  = 100,
    f2_shilling=200,
    f2_high_dege=100,

    f2_bias_users     = 100,
    f2_bias_items     = 6,          # non usato
    f2_bias_magnitude = 2.0,        # USATO

    f2_highprod_users  = 100,
    f2_highprod_items  = 29,        # USATO: 18 → 29

    f2_late_users       = 100,
    f2_late_items       = 5,        # non usato
    f2_late_days_after  = 180,      # USATO
    f2_late_rating      = 5,

    # ── F3 / F4 / F5 — non chiamate ──────────────────────────────
    f3_hub_users=0, f3_hub_items=0,
    f3_bridge_users=0, f3_bridge_items_per_cl=0,
    f3_sybil_groups=0, f3_sybil_group_size=2, f3_sybil_items=0, f3_sybil_rating=5,
    f3_echo_groups=0, f3_echo_group_size=2, f3_echo_shared_items=0, f3_echo_rating=5,
    f4_nearlegit_users=0, f4_drift_users=0, f4_copycat_users=0,
    f5_poweruser_n=0, f5_niche_n=0, f5_earlyadopter_n=0,

    popular_top_pct = 0.10,
)

# ══════════════════════════════════════════════════════════════════
# BOOKS — 776K users, 495K items, 9.50M edges
# users ≈ 23 278 (3.0%)  |  archi ≈ 7.0%  |  items ≈ 67%
# Leva: +77% gruppi F1, +77% users F2, highprod_items 55 → 85
# late_days_after 365: ciclo di vita lungo dei libri
# ══════════════════════════════════════════════════════════════════
cfg_books = InjectionConfig(
    seed=42,

    # ── F1 ── parametri che contano: solo i *_groups e delayed size/waves
    f1_burst_groups         = 300,
    f1_burst_group_size     = 10,   # non usato (randint nel codice)
    f1_burst_items          = 2,    # non usato
    f1_burst_window_hours   = 24,
    f1_burst_rating         = 5,
    f2_bridge=300,
    f2_shilling=400,
    f2_high_dege=300,

    f1_camouflage_groups        = 300,
    f1_camouflage_group_size    = 10,   # non usato
    f1_camouflage_target        = 3,    # non usato
    f1_camouflage_noise         = 8,    # non usato
    f1_camouflage_target_rating = 5,

    f1_delayed_groups       = 300,
    f1_delayed_group_size   = 10,   # USATO: users_per_wave = size // waves
    f1_delayed_waves        = 3,    # USATO
    f1_delayed_target_rating= 5,

    f1_hetero_groups        = 500,
    f1_hetero_group_size    = 10,   # non usato
    f1_hetero_target_items  = 3,    # non usato

    # ── F2 ── highprod_items è l'unico param item effettivo
    f2_extreme_users  = 500,
    f2_extreme_items  = 5,          # non usato

    f2_bias_users     = 500,
    f2_bias_items     = 6,          # non usato
    f2_bias_magnitude = 2.0,        # USATO

    f2_highprod_users  = 500,
    f2_highprod_items  = 29,        # USATO: 18 → 29

    f2_late_users       = 500,
    f2_late_items       = 5,        # non usato
    f2_late_days_after  = 180,      # USATO
    f2_late_rating      = 5,

    # ── F3 / F4 / F5 — non chiamate ──────────────────────────────
    f3_hub_users=0, f3_hub_items=0,
    f3_bridge_users=0, f3_bridge_items_per_cl=0,
    f3_sybil_groups=0, f3_sybil_group_size=2, f3_sybil_items=0, f3_sybil_rating=5,
    f3_echo_groups=0, f3_echo_group_size=2, f3_echo_shared_items=0, f3_echo_rating=5,
    f4_nearlegit_users=0, f4_drift_users=0, f4_copycat_users=0,
    f5_poweruser_n=0, f5_niche_n=0, f5_earlyadopter_n=0,

    popular_top_pct = 0.10,
)

# ══════════════════════════════════════════════════════════════════
# REGISTRO
# ══════════════════════════════════════════════════════════════════
CONFIGS = {
    "baby_products":   cfg_baby_products,
    "office_products": cfg_office_products,
    "sports_outdoors": cfg_sports_outdoors,
    "toys_games":      cfg_toys_games,
    "pet_supplies":    cfg_pet_supplies,
    "books":           cfg_books,
}