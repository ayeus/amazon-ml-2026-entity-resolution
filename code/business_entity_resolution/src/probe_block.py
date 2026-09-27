import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK, NF, HALF
import blocking as bl
from blocking import load_split, make_ctx

ms = pd.read_pickle(f"{WORK}/val_missed2.pkl"); ms = ms[(~ms.in_block) & (ms.cat == "other")]
data = load_split("train"); df1 = data[1][0]
rows = []
for country in ("US", "India"):
    sub = ms[ms.country == country].sample(40, random_state=5)
    ctx = make_ctx(data, country)
    g = bl._G
    wn = g["wn"].diagonal(); wa = g["wa"].diagonal()
    for _, r in sub.iterrows():
        x1 = g["X1all"][int(r.i1)]
        Bx = data[int(r.src)][1][int(r.j)]
        sn = x1.multiply(Bx).multiply(wn).sum(); sa = x1.multiply(Bx).multiply(wa).sum()
        PT = g["PT2"] if r.src == 2 else g["PT3"]
        Sn = (x1 @ sp.diags(wn) @ PT).toarray().ravel(); Sa = (x1 @ sp.diags(wa) @ PT).toarray().ravel(); St = Sn + Sa
        kth = lambda S, K: np.sort(S)[::-1][K - 1]
        rows.append(dict(country=country, src=int(r.src), true_all=sn + sa, kth_all=kth(St, 15), true_name=sn, kth_name=kth(Sn, 12),
                         true_addr=sa, kth_addr=kth(Sa, 12),
                         rank_all=int((St > sn + sa).sum()) + 1, rank_name=int((Sn > sn).sum()) + 1, rank_addr=int((Sa > sa).sum()) + 1,
                         shared_tokens=int(x1.multiply(Bx).nnz), shared_weighted=int((x1.multiply(Bx).multiply(wn + wa) != 0).sum())))
R = pd.DataFrame(rows)
print(R.groupby("country")[["shared_tokens", "shared_weighted"]].describe().T.round(1).to_string())
print("\nrank of true B in each channel (pool ranks; retrieval needs <=15 all, <=12 name/addr):")
print(R[["rank_all", "rank_name", "rank_addr"]].describe(percentiles=[.25, .5, .75, .9]).round(0).to_string())
print("\ntrue B score / K-th score  (all/name/addr):")
print((R[["true_all", "true_name", "true_addr"]].values / R[["kth_all", "kth_name", "kth_addr"]].values).round(2)[:15])
print("\nfraction with zero weighted shared tokens:", (R.shared_weighted == 0).mean().round(3), "| with zero shared tokens:", (R.shared_tokens == 0).mean().round(3))
R.to_pickle(f"{WORK}/probe.pkl")
