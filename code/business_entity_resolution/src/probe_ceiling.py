"""For a sample of never-blocked true pairs, the rank of the true record in every blocking channel (how far outside top-K)."""
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK
import blocking as bl
from blocking import load_split, make_ctx

ms = pd.read_pickle(f"{WORK}/ceiling_misses.pkl"); ms = ms[ms.stage == "never blocked"]
data = load_split("train"); ctry = data[1][0].country.to_numpy(dtype=object)
rows = []
for c in ("US", "India"):
    ctx = make_ctx(data, c); g = bl._G
    for t in ("name ok, address ok", "no address, unique name", "alias/junk name, address ok"):
        sub = ms[(ctry[ms.i1.values] == c) & (ms.type == t)]
        for _, r in sub.sample(min(40, len(sub)), random_state=0).iterrows():
            x1 = g["X1all"][int(r.i1)]; s = int(r.src); PT = g["PT2"] if s == 2 else g["PT3"]
            jj = int(np.flatnonzero(ctx["pools"][s][0] == r.j)[0])
            Sn = (x1 @ g["wn"] @ PT).toarray().ravel(); Sa = (x1 @ g["wa"] @ PT).toarray().ravel(); Sx = (x1 @ g["wx"] @ PT).toarray().ravel()
            Sc = Sn * (g["invB"][s][0].diagonal()); Sy = Sx * (g["invB"][s][2].diagonal())
            rk = lambda S: int((S > S[jj]).sum()) + 1 if S[jj] > 0 else -1
            rows.append(dict(country=c, type=t, r_all=rk(Sn + Sa), r_name=rk(Sn), r_addr=rk(Sa), r_x=rk(Sx), r_cos=rk(Sc), r_y=rk(Sy),
                             n_tied_name=int((Sn == Sn[jj]).sum()) if Sn[jj] > 0 else 0))
R = pd.DataFrame(rows)
print("ranks of the TRUE record per channel (K: all 15, name 12, addr 12, composite 10, cosine 6; -1 = zero score)")
for t, g_ in R.groupby("type"):
    print(f"\n== {t} (n={len(g_)})")
    print(g_[["r_all", "r_name", "r_addr", "r_x", "r_cos", "r_y", "n_tied_name"]].describe(percentiles=[.25, .5, .75]).loc[["25%", "50%", "75%"]].round(0).to_string())
    print("  zero name score:", round((g_.r_name == -1).mean(), 2), "| best rank over channels median:", g_[["r_all", "r_name", "r_addr", "r_x", "r_cos", "r_y"]].replace(-1, 10**9).min(axis=1).median())
R.to_pickle(f"{WORK}/probe_ceiling.pkl")
