"""EDA audit for ESCI Task 1: near-duplicate queries, shortcut checks, single-feature ranking table,
samples for checking the text rules by hand.

Usage: python eda/eda_audit.py  (run from the project root; needs only the raw data files in the root)
Writes eda/audit_stats.json, eda/manifest.json, eda/tables/*.csv, eda/figures/a*.png.
"""
import hashlib
import json
import math
import re
from collections import Counter, defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common import (GAIN, INK, INK2, NEG_RE, NUM_RE, OUT, ROOT, SERIES, SURFACE, TABLES, colors_of, missing, save,
                    title_negates, tokens)

SEED = 42
VAL_SHARE = 0.10
stats = {}

ex = pd.read_parquet(ROOT / "examples_us_small.parquet")
pr = pd.read_parquet(ROOT / "products_us_small.parquet")
src = pd.read_csv(ROOT / "sources_us_small.csv")
pr["catalog_pos"] = np.arange(len(pr)) / len(pr)  # row order of the product file

#  K. near-duplicate queries
def norm_tokens(q):
    out = []
    for t in tokens(q):
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return set(out)


qdf = ex.drop_duplicates("query_id")[["query_id", "query", "split"]].reset_index(drop=True)
qsets = [norm_tokens(q) for q in qdf["query"]]
keys = [" ".join(sorted(s)) for s in qsets]
dfreq = Counter(t for s in qsets for t in s)

parent = list(range(len(qdf)))


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def union(i, j):
    a, b = find(i), find(j)
    if a != b:
        parent[b] = a


edges = set()
by_key = defaultdict(list)
for i, k in enumerate(keys):
    if k:
        by_key[k].append(i)
for members in by_key.values():
    for j in members[1:]:
        edges.add((members[0], j, 1.0))

blocks = defaultdict(list)  # block on the two rarest tokens of each query
for i, s in enumerate(qsets):
    for t in sorted(s, key=lambda t: (dfreq[t], t))[:2]:
        blocks[t].append(i)
for members in blocks.values():
    if len(members) < 2 or len(members) > 3000:
        continue
    for a_pos, a in enumerate(members):
        sa = qsets[a]
        for b in members[a_pos + 1:]:
            if a == b or keys[a] == keys[b]:
                continue
            sb = qsets[b]
            inter = len(sa & sb)
            if inter and inter / len(sa | sb) >= 0.8:
                edges.add((min(a, b), max(a, b), round(inter / len(sa | sb), 3)))
for a, b, _ in edges:
    union(a, b)
qdf["dup_group"] = [find(i) for i in range(len(qdf))]
grp_size = qdf.groupby("dup_group").query_id.transform("size")
grp_splits = qdf.groupby("dup_group").split.transform("nunique")

e = pd.DataFrame(sorted(edges), columns=["a", "b", "jaccard"])
e["split_a"], e["split_b"] = qdf.split.values[e.a], qdf.split.values[e.b]
e["query_a"], e["query_b"] = qdf["query"].values[e.a], qdf["query"].values[e.b]
e["qid_a"], e["qid_b"] = qdf.query_id.values[e.a], qdf.query_id.values[e.b]
cross = e[e.split_a != e.split_b].copy()

# product overlap and label agreement for train-test near-duplicate pairs
prod_sets = ex.groupby("query_id").product_id.agg(set)
lab = ex.set_index(["query_id", "product_id"]).esci_label
over, agree, shared_n = [], [], []
for qa, qb in zip(cross.qid_a, cross.qid_b):
    pa, pb = prod_sets[qa], prod_sets[qb]
    sh = pa & pb
    over.append(len(sh) / min(len(pa), len(pb)))
    shared_n.append(len(sh))
    agree.append(np.mean([lab[(qa, p)] == lab[(qb, p)] for p in sh]) if sh else np.nan)
cross["product_overlap"], cross["shared_products"], cross["label_agreement"] = over, shared_n, agree

K = {
    "queries": len(qdf),
    "edges_exact_after_normalization": int((e.jaccard == 1.0).sum()),
    "edges_jaccard_ge_0.8": int((e.jaccard < 1.0).sum()),
    "groups_with_2plus_queries": int((qdf.groupby("dup_group").size() > 1).sum()),
    "queries_in_such_groups": int((grp_size > 1).sum()),
    "groups_spanning_train_and_test": int(qdf.loc[grp_splits > 1, "dup_group"].nunique()),
    "test_queries_with_near_dup_in_train": int(qdf[(qdf.split == "test") & (grp_splits > 1)].shape[0]),
    "train_test_pairs": len(cross),
    "train_test_pairs_mean_product_overlap": round(float(cross.product_overlap.mean()), 4),
    "train_test_pairs_with_shared_products": int((cross.shared_products > 0).sum()),
    "shared_product_label_agreement": round(float(np.nanmean(cross.label_agreement)), 4),
}
K["examples"] = cross.sort_values("jaccard", ascending=False)[["query_a", "query_b", "jaccard", "product_overlap"]] \
    .drop_duplicates("query_a").head(10).to_dict("records")
stats["K_near_duplicates"] = K
cross[["qid_a", "query_a", "split_a", "qid_b", "query_b", "split_b", "jaccard", "product_overlap",
       "shared_products", "label_agreement"]].to_csv(TABLES / "near_dup_pairs_train_test.csv", index=False)

# group-aware validation proposal: 10% of train dup groups (groups touching test stay in train)
train_groups = qdf[(qdf.split == "train") & (grp_splits == 1)].dup_group.unique()
rng = np.random.default_rng(SEED)
n_val_target = int(round(VAL_SHARE * (qdf.split == "train").sum()))
val_groups, n_val = set(), 0
for g in rng.permutation(train_groups):
    if n_val >= n_val_target:
        break
    val_groups.add(g)
    n_val += int((qdf.dup_group == g).sum())
qdf["split_eda"] = np.where(qdf.dup_group.isin(val_groups), "val", qdf.split)
qdf[["query_id", "split", "split_eda", "dup_group"]].to_csv(TABLES / "query_groups_val_proposal.csv", index=False)
stats["val_proposal"] = {"seed": SEED, "val_queries": int((qdf.split_eda == "val").sum()),
                         "train_queries_left": int((qdf.split_eda == "train").sum())}

# pair-level features
keep = ["product_id", "product_title", "product_brand", "product_color", "product_bullet_point",
        "product_description", "catalog_pos"]
df = ex.merge(pr[keep], on="product_id", how="left").merge(src, on="query_id", how="left")
df = df.merge(qdf[["query_id", "split_eda"]], on="query_id")
df["gain"] = df.esci_label.map(GAIN)

title_tok = df.product_title.map(tokens)
q_tok = df["query"].map(tokens)
q_set = q_tok.map(set)
main_text = (df.product_title.fillna("") + " " + df.product_brand.fillna("") + " " + df.product_color.fillna("")
             + " " + df.product_bullet_point.fillna(""))
main_set = main_text.map(lambda s: set(tokens(s)))

# BM25 on titles, IDF over the whole product catalogue
cat_tok = pr.product_title.map(tokens)
N = len(cat_tok)
title_df = Counter(t for ts in cat_tok for t in set(ts))
avgdl = float(np.mean([len(ts) for ts in cat_tok]))
k1, b = 1.2, 0.75


def bm25(qt, dt):
    if not dt:
        return 0.0
    tf = Counter(dt)
    norm = k1 * (1 - b + b * len(dt) / avgdl)
    s = 0.0
    for t in set(qt):
        f = tf.get(t, 0)
        if f:
            n = title_df.get(t, 0)
            s += math.log(1 + (N - n + 0.5) / (n + 0.5)) * f * (k1 + 1) / (f + norm)
    return s


df["bm25_title"] = [bm25(q, d) for q, d in zip(q_tok, title_tok)]
df["overlap_title"] = [len(q & set(t)) / len(q) if q else 0.0 for q, t in zip(q_set, title_tok)]
df["overlap_main"] = [len(q & m) / len(q) if q else 0.0 for q, m in zip(q_set, main_set)]

q_nums = df["query"].map(lambda s: set(re.findall(NUM_RE, s)))
t_nums = df.product_title.fillna("").map(lambda s: set(re.findall(NUM_RE, s)))
df["num_match"] = [0.5 if not q else float(q <= t) for q, t in zip(q_nums, t_nums)]
qc = df["query"].map(colors_of)
pc = (df.product_title.fillna("") + " " + df.product_color.fillna("")).map(colors_of)
df["color_match"] = [0.5 if not a or not p else float(bool(a & p)) for a, p in zip(qc, pc)]

qjoin = df["query"].map(lambda s: " " + " ".join(tokens(s)) + " ")
bjoin = df.product_brand.map(lambda s: " ".join(tokens(s)))
df["brand_in_query"] = [float(len(x) >= 3 and f" {x} " in q) for x, q in zip(bjoin, qjoin)]
df["_b"] = df.product_brand.fillna("").str.lower().str.strip()
cnt = df[df._b != ""].groupby(["query_id", "_b"]).size().rename("bcnt").reset_index()
df = df.merge(cnt, on=["query_id", "_b"], how="left")
df["n_cand"] = df.groupby("query_id").product_id.transform("size")
df["brand_share"] = df.bcnt.fillna(0) / df.n_cand
top = cnt.sort_values("bcnt", ascending=False).drop_duplicates("query_id").set_index("query_id")
top_map = dict(zip(top.index, zip(top._b, top.bcnt)))
df["top_brand"] = [float(bb != "" and top_map.get(q, ("", 0))[0] == bb and top_map.get(q, ("", 0))[1] > 1)
                   for q, bb in zip(df.query_id, df._b)]

# negation: crude rule (negated term appears in title) and refined rule (title does not negate it itself)
neg_terms = df["query"].str.lower().map(lambda s: set(re.findall(NEG_RE, s)))
title_low = df.product_title.fillna("").str.lower()
crude, refined = [], []
for terms, ttl, tt in zip(neg_terms, title_low, title_tok):
    if not terms:
        crude.append(0.5)
        refined.append(0.5)
        continue
    hit = [t for t in terms if t in set(tt)]
    crude.append(float(bool(hit)))
    refined.append(float(any(not title_negates(ttl, t) for t in hit)))
df["neg_term_in_title"] = crude
df["neg_violation_refined"] = refined

df["title_words"] = df.product_title.fillna("").str.split().str.len()
df["bullet_words"] = df.product_bullet_point.fillna("").str.split().str.len()
for f in ["product_description", "product_bullet_point", "product_brand", "product_color"]:
    df["miss_" + f.replace("product_", "")] = missing(df[f]).astype(float)
df["html_in_description"] = df.product_description.fillna("").str.contains(r"<[a-zA-Z/][^>]{0,60}>", regex=True).astype(float)
df["isbn_like_id"] = df.product_id.str[0].str.isdigit().astype(float)

# product prior: mean gain of the same product in other training queries (memorization shortcut)
fit_part = df[df.split_eda == "train"]
psum = fit_part.groupby("product_id").gain.sum()
pcnt = fit_part.groupby("product_id").gain.size()
s_ = df.product_id.map(psum).fillna(0.0)
c_ = df.product_id.map(pcnt).fillna(0).astype(int)
own = (df.split_eda == "train")
num = s_ - np.where(own, df.gain, 0.0)
den = c_ - own.astype(int)
df["product_seen"] = (den > 0).astype(float)
global_mean = float(fit_part.gain.mean())
df["product_prior"] = np.where(den > 0, num / np.maximum(den, 1), global_mean)

#  I/J. single-feature ranking table
FEATURES = [
    ("bm25_title", "relevance", "BM25 on title"),
    ("overlap_title", "relevance", "Share of query words found in title"),
    ("overlap_main", "relevance", "Share of query words found in title+brand+color+bullets"),
    ("num_match", "relevance", "All numbers in the query appear in the title"),
    ("color_match", "relevance", "Query color matches product color"),
    ("brand_in_query", "relevance", "Product brand appears in the query"),
    ("neg_term_in_title", "relevance", "Title contains the negated term (raw rule)"),
    ("neg_violation_refined", "relevance", "Title contains the negated term (refined rule)"),
    ("top_brand", "list", "Most frequent brand in the list"),
    ("brand_share", "list", "Share of candidates with the same brand"),
    ("title_words", "product", "Number of words in title"),
    ("bullet_words", "product", "Number of words in bullet points"),
    ("miss_bullet_point", "product", "Bullet points missing"),
    ("miss_brand", "product", "Brand missing"),
    ("miss_color", "product", "Color missing"),
    ("miss_description", "product", "Description missing"),
    ("html_in_description", "artifact", "Description contains HTML"),
    ("product_prior", "artifact", "Mean gain of the product in other queries (product memorization)"),
    ("catalog_pos", "artifact", "Row position in the products file"),
    ("isbn_like_id", "artifact", "ISBN-like product_id (starts with a digit)"),
    ("n_cand", "artifact", "Number of candidates of the query (constant within a query)"),
]

parts = {"train": df[df.split_eda == "train"], "val": df[df.split_eda == "val"]}
idcg = {}
for name, part in parts.items():
    g = part[["query_id", "gain"]].copy()
    g["r"] = g.groupby("query_id").gain.rank(method="first", ascending=False) - 1
    idcg[name] = (g.gain / np.log2(g.r + 2)).groupby(g.query_id).sum()


def ndcg(part, name, score, seeds=3):
    vals = []
    for sd in range(seeds):
        f = pd.DataFrame({"q": part.query_id.values, "g": part.gain.values, "s": score,
                          "r": np.random.default_rng(sd).random(len(part))})
        f = f.sort_values(["q", "s", "r"], ascending=[True, False, True])
        pos = f.groupby("q").cumcount().values
        dcg = pd.Series(f.g.values / np.log2(pos + 2)).groupby(f.q.values).sum()
        vals.append(float((dcg / idcg[name].loc[dcg.index]).mean()))
    return float(np.mean(vals))


def e_over_s(part, score):
    f = pd.DataFrame({"q": part.query_id.values, "l": part.esci_label.values, "s": score})
    f = f[f.l.isin(["E", "S"])]
    f["rk"] = f.groupby("q").s.rank(method="average")  # Mann-Whitney U: ties count one half
    f["isE"] = f.l == "E"
    agg = f.groupby("q").agg(nE=("isE", "sum"), n=("isE", "size"))
    agg["rE"] = f[f.isE].groupby("q").rk.sum().reindex(agg.index).fillna(0.0)
    agg["nS"] = agg.n - agg.nE
    agg = agg[(agg.nE > 0) & (agg.nS > 0)]
    u = agg.rE - agg.nE * (agg.nE + 1) / 2
    return float(u.sum() / (agg.nE * agg.nS).sum())


rows = []
rand = {n: ndcg(p, n, np.zeros(len(p))) for n, p in parts.items()}
for col, group, desc in FEATURES:
    tr_s = parts["train"][col].astype(float).values
    sign = 1.0 if ndcg(parts["train"], "train", tr_s, seeds=1) >= ndcg(parts["train"], "train", -tr_s, seeds=1) else -1.0
    row = {"feature": col, "group": group, "description": desc, "direction": "+" if sign > 0 else "−"}
    for n, p in parts.items():
        s = sign * p[col].astype(float).values
        row[f"ndcg_{n}"] = round(ndcg(p, n, s), 4)
        row[f"gain_vs_random_{n}"] = round(row[f"ndcg_{n}"] - rand[n], 4)
        row[f"e_over_s_{n}"] = round(e_over_s(p, s), 4)
    row["val_minus_train"] = round(row["gain_vs_random_val"] - row["gain_vs_random_train"], 4)
    rows.append(row)
feat = pd.DataFrame(rows)
feat.to_csv(TABLES / "single_feature_ranking.csv", index=False)
stats["random_ndcg"] = {k: round(v, 4) for k, v in rand.items()}
stats["J_single_feature"] = feat.to_dict("records")

# shortcut details
seen_val = parts["val"][parts["val"].product_seen == 1]
I = {
    "share_pairs_product_seen_in_train": {
        "val": round(float(parts["val"].product_seen.mean()), 4),
        "test": round(float(df[df.split_eda == "test"].product_id.isin(fit_part.product_id).mean()), 4),
    },
    "isbn_like_share": round(float(df.isbn_like_id.mean()), 4),
    "isbn_like_label_share": df[df.isbn_like_id == 1].esci_label.value_counts(normalize=True).round(3).to_dict(),
}
maj = fit_part.groupby("product_id").esci_label.agg(lambda s: s.value_counts().index[0])
I["val_seen_pairs"] = int(len(seen_val))
I["val_seen_label_agreement_with_train_majority"] = round(
    float((seen_val.product_id.map(maj) == seen_val.esci_label).mean()), 4)
stats["I_shortcuts"] = I

#  L. samples for checking rules by hand
# Written once; the judgement columns are added by hand, so existing files are never overwritten.
def write_sample(frame, n, cols, name):
    path = TABLES / name
    if not path.exists():
        frame.sample(n, random_state=0)[cols].to_csv(path, index=False)


nq = qdf[qdf["query"].str.lower().map(lambda s: bool(re.search(NEG_RE, s)))].copy()
nq["captured"] = nq["query"].str.lower().map(lambda s: ", ".join(re.findall(NEG_RE, s)))
write_sample(nq, 40, ["query_id", "query", "captured"], "rule_check_negation_queries.csv")
nt = df[(df.neg_term_in_title == 1)].copy()
nt["captured"] = nt["query"].str.lower().map(lambda s: ", ".join(re.findall(NEG_RE, s)))
write_sample(nt, 30, ["query", "captured", "product_title", "esci_label", "neg_violation_refined"],
             "rule_check_negation_titles.csv")
cq = qdf[qdf["query"].map(lambda s: bool(colors_of(s)))].copy()
cq["colors"] = cq["query"].map(lambda s: ", ".join(sorted(colors_of(s))))
write_sample(cq, 40, ["query_id", "query", "colors"], "rule_check_color_queries.csv")
neg_ref = df[df.neg_term_in_title == 1]
L = {
    "negation_title_pairs": int(len(neg_ref)),
    "refined_violation_share": round(float((neg_ref.neg_violation_refined == 1).mean()), 4),
    "P_E_refined": {
        "violates": round(float((df[df.neg_violation_refined == 1].esci_label == "E").mean()), 4),
        "no_violation": round(float((df[(df.neg_violation_refined == 0)].esci_label == "E").mean()), 4),
    },
}
stats["L_rules"] = L

(OUT / "audit_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

# manifest (data hashes, outputs)
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


manifest = {
    "data": {p: sha256(ROOT / p) for p in ["examples_us_small.parquet", "products_us_small.parquet",
                                             "sources_us_small.csv"]},
    "scripts": ["eda/eda_analysis.py", "eda/eda_audit.py", "eda/common.py"],
    "outputs": sorted(str(p.relative_to(ROOT)).replace("\\", "/")
                      for p in list(OUT.glob("*.json")) + list(TABLES.glob("*.csv")) + list((OUT / "figures").glob("*.png"))),
    "seed_val_proposal": SEED,
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

#  figure: single-feature strength
GROUP_NAME = {"relevance": "query–product match", "list": "list context", "product": "product information",
              "artifact": "artifact / shortcut"}
GROUP_COLOR = {"relevance": SERIES[0], "list": "#1baf7a", "product": "#b5b3ab", "artifact": SERIES[1]}
order = feat.sort_values("gain_vs_random_val", ascending=True)
fig, ax = plt.subplots(figsize=(6.6, 5.6))
y = np.arange(len(order))
ax.barh(y, order.gain_vs_random_val, color=[GROUP_COLOR[g] for g in order.group], height=0.62,
        edgecolor=SURFACE, linewidth=1.2)
ax.scatter(order.gain_vs_random_train, y, marker="|", s=90, color=INK, zorder=3)
for yi, v, t in zip(y, order.gain_vs_random_val, order.gain_vs_random_train):
    ax.text(max(v, t, 0) + 0.0015, yi, f"{v:+.3f}", va="center", fontsize=7, color=INK2)
ax.set_yticks(y, [f"{d} ({s})" for d, s in zip(order.description, order.direction)], fontsize=7.5)
ax.axvline(0, color=INK, linewidth=0.8)
ax.set_xlim(right=float(order[["gain_vs_random_val", "gain_vs_random_train"]].max().max()) + 0.012)
ax.set_xlabel("val nDCG using this feature alone, minus random order")
ax.grid(axis="x")
ax.set_axisbelow(True)
handles = [Patch(color=GROUP_COLOR[g], label=GROUP_NAME[g]) for g in GROUP_NAME]
handles.append(Line2D([0], [0], marker="|", color=INK, linestyle="none", markersize=9, label="train value"))
ax.legend(handles=handles, ncol=3, loc="lower left", bbox_to_anchor=(-0.55, 1.0), fontsize=7.5)
ax.set_title("Strength of each feature used alone", pad=34, x=-0.55)
save(fig, "a1_single_feature.png")

print(json.dumps({k: v for k, v in stats.items() if k != "J_single_feature"}, ensure_ascii=False, indent=1, default=str))
print(feat[["feature", "direction", "ndcg_train", "ndcg_val", "gain_vs_random_val", "e_over_s_train", "e_over_s_val",
            "val_minus_train"]].to_string(index=False))
