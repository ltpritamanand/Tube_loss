"""
Condition C1 — BERT + all 10 LLM scores as features (raw uncertainty vector)
"""

import os, json, re, time, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from groq import Groq
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════
DATA_PATH     = "data/model_annotations.aligned/paired/model_annotations.aligned.paired.jsonl"
CACHE_LLM     = "cache_llm_v2_c1.json"
CACHE_MULTI   = "cache_llm_multi.json"    # separate cache for 10-shot scores
CACHE_EMBED   = "cache_embed_v2_c1.json"
OUT_DIR       = "bmp_results_C1"
os.makedirs(OUT_DIR, exist_ok=True)

MAX_SAMPLES   = 200
ALPHA         = 0.10
T_CONF        = 1.0 - ALPHA
R_SHIFT       = 0.0
HIDDEN        = 256
EPOCHS        = 150
LR            = 3e-4
BATCH         = 16
SEED          = 42
DIMS          = ["coherence", "consistency", "fluency", "relevance"]
PER_DIMENSION = True
N_LLM_RUNS   = 5    # number of repeated LLM calls per sample

Q     = 1.0 - ALPHA
R     = 0.5
DELTA = 0.05

COLORS = {"coherence":"#4C72B0","consistency":"#DD8452",
          "fluency":"#55A868","relevance":"#C44E52","combined":"#8172B2"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED)

print("=" * 62)
print("  Condition C1 — BERT + 10 raw LLM scores as features")
print(f"  TubeNet + Tube Loss  |  t={T_CONF}  DELTA={DELTA}")
print(f"  device={DEVICE}  α={ALPHA}  N_LLM_RUNS={N_LLM_RUNS}")
print("=" * 62)

# ══════════════════════════════════════════════════════════
#  CACHE
# ══════════════════════════════════════════════════════════
def load_json(p):    return json.load(open(p)) if os.path.exists(p) else {}
def save_json(p, o): json.dump(o, open(p, "w"))

llm_cache   = load_json(CACHE_LLM)
multi_cache = load_json(CACHE_MULTI)
embed_cache = load_json(CACHE_EMBED)

# ══════════════════════════════════════════════════════════
#  LLM
# ══════════════════════════════════════════════════════════
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

PROMPT_TMPL = (
    "You are an expert evaluator of text summaries.\n"
    "Rate the summary quality on a scale from 1 to 5:\n"
    "1 = very poor, 2 = poor, 3 = average, 4 = good, 5 = excellent.\n\n"
    "Guidelines:\n"
    "- Use the full range of scores (1 to 5).\n"
    "- Score 3 only if the summary is truly average.\n"
    "- Give 1 or 2 for clearly bad summaries.\n"
    "- Give 4 or 5 for strong, high-quality summaries.\n"
    "- Be consistent and objective.\n\n"
    "Article: {article}\nSummary: {summary}\n\n"
    "Return ONLY one number (1, 2, 3, 4, or 5)."
)

def llm_judge_score(article, summary):
    """Single call, temp=0. Used only for demo display."""
    key = summary[:150]
    if key in llm_cache: return llm_cache[key]
    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":PROMPT_TMPL.format(
                article=article[:500], summary=summary)}],
            temperature=0, max_tokens=5)
        m = re.search(r"[1-5]", r.choices[0].message.content.strip())
        score = float(m.group()) if m else 3.0
    except Exception: score = 3.0
    llm_cache[key] = score; save_json(CACHE_LLM, llm_cache)
    return score

def llm_judge_scores_n(article, summary, n=N_LLM_RUNS):
    """n calls with temp=0.7 — captures LLM score variability."""
    key = f"n{n}_{summary[:150]}"
    if key in multi_cache: return multi_cache[key]
    scores = []
    for _ in range(n):
        try:
            r = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role":"user","content":PROMPT_TMPL.format(
                    article=article[:500], summary=summary)}],
                temperature=0.2, max_tokens=5)
            m = re.search(r"[1-5]", r.choices[0].message.content.strip())
            scores.append(float(m.group()) if m else 3.0)
        except Exception: scores.append(3.0)
        time.sleep(0.05)
    multi_cache[key] = scores; save_json(CACHE_MULTI, multi_cache)
    return scores

# ══════════════════════════════════════════════════════════
#  BERT EMBEDDINGS
# ══════════════════════════════════════════════════════════
print("[Init] Loading SentenceTransformer …")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_bert_embedding(article, summary):
    key = summary[:150]
    if key in embed_cache: return np.array(embed_cache[key], dtype=np.float32)
    a = embedder.encode(article[:512], show_progress_bar=False, normalize_embeddings=True)
    s = embedder.encode(summary[:256], show_progress_bar=False, normalize_embeddings=True)
    vec = np.concatenate([a, s])
    embed_cache[key] = vec.tolist(); save_json(CACHE_EMBED, embed_cache)
    return vec.astype(np.float32)

# ══════════════════════════════════════════════════════════
#  TUBE LOSS
# ══════════════════════════════════════════════════════════
def tube_loss(mu1, mu2, y):
    f1, f2 = mu1, mu2
    c1 = (1-Q)*(f2-y); c2 = (1-Q)*(y-f1)
    c3 =    Q *(f1-y); c4 =    Q *(y-f2)
    threshold  = R*f2 + (1-R)*f1
    loss_part1 = torch.where(y >= threshold, c1, c2)
    loss_part2 = torch.where(f1 > y, c3, c4)
    inside     = (y >= f1) & (y <= f2)
    return (torch.where(inside, loss_part1, loss_part2)
            + DELTA * torch.abs(f1-f2)).mean()

# ══════════════════════════════════════════════════════════
#  LOAD DATA  — C1: X = BERT(768) + 10 raw LLM scores (778-dim)
# ══════════════════════════════════════════════════════════
print("\n[Data] Loading SummEval …")
raw = [json.loads(l)
       for l in open(DATA_PATH, encoding="utf-8", errors="ignore")][:MAX_SAMPLES]

X_list, Y_dims_list, llm_scores_list = [], [], []

for sample in tqdm(raw, desc="Features"):
    art  = sample["text"][:800]; summ = sample["decoded"]
    anns = sample.get("expert_annotations") or sample.get("turker_annotations", [])
    if not anns: continue
    dim_scores = [float(np.mean([a[d] for a in anns if d in a])) for d in DIMS]
    z         = get_bert_embedding(art, summ)              # 768-dim
    yh_single = llm_judge_score(art, summ)                 # for demo display
    yh_multi  = llm_judge_scores_n(art, summ)              # 10 scores

    # ── C1 FEATURE BLOCK ──────────────────────────────────
    llm_feats = np.array(yh_multi, dtype=np.float32)       # shape (10,) — all raw scores
    # ──────────────────────────────────────────────────────

    X_list.append(np.concatenate([z, llm_feats]))
    Y_dims_list.append(dim_scores)
    llm_scores_list.append(yh_single)
    time.sleep(0.03)

X          = np.array(X_list,      dtype=np.float32)
Y_dims     = np.array(Y_dims_list, dtype=np.float32)
Y_med      = np.median(Y_dims, axis=1)
llm_scores = np.array(llm_scores_list, dtype=np.float32)
N          = len(Y_med)
print(f"[Data] N={N}  X={X.shape}  (BERT 768 + {N_LLM_RUNS} LLM scores)")

idx = np.arange(N)
idx_tr, idx_tmp = train_test_split(idx, test_size=0.30, random_state=SEED)
idx_cal, idx_te = train_test_split(idx_tmp, test_size=0.33, random_state=SEED)
X_tr  = X[idx_tr]; X_cal = X[idx_cal]; X_te = X[idx_te]
print(f"[Split] train={len(idx_tr)}  cal={len(idx_cal)}  test={len(idx_te)}")

# ══════════════════════════════════════════════════════════
#  TUBENET
# ══════════════════════════════════════════════════════════
class TubeNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN), nn.LayerNorm(HIDDEN), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(HIDDEN, HIDDEN//2), nn.LayerNorm(HIDDEN//2), nn.GELU(), nn.Dropout(0.1),
        )
        self.head_mid  = nn.Linear(HIDDEN//2, 1)
        self.head_half = nn.Linear(HIDDEN//2, 1)

    def forward(self, x):
        h    = self.net(x)
        mid  = self.head_mid(h).squeeze(-1)
        half = torch.nn.functional.softplus(self.head_half(h)).squeeze(-1)
        return mid - half, mid + half

class TabDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):        return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

IN_DIM = X.shape[1]   # 778 for C1

# ══════════════════════════════════════════════════════════
#  TRAIN + CONFORMAL EVAL
# ══════════════════════════════════════════════════════════
def train_and_eval(y_tr, y_cal, y_te, label="combined"):
    model = TubeNet(IN_DIM).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ldr   = DataLoader(TabDS(X_tr, y_tr), batch_size=BATCH, shuffle=True)
    loss_hist = []
    for ep in range(1, EPOCHS+1):
        model.train(); ep_loss = 0.0
        for xb, yb in ldr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            m1, m2 = model(xb)
            loss   = tube_loss(m1, m2, yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ep_loss += loss.item()*len(yb)
        sch.step()
        loss_hist.append(ep_loss/len(y_tr))

    model.eval()
    @torch.no_grad()
    def pred(Xnp):
        xt = torch.tensor(Xnp, dtype=torch.float32).to(DEVICE)
        m1, m2 = model(xt); return m1.cpu().numpy(), m2.cpu().numpy()

    m1c, m2c = pred(X_cal)
    nc    = np.maximum(m1c - y_cal, y_cal - m2c)
    n_cal = len(y_cal)
    qlev  = min((n_cal+1)*(1-ALPHA)/n_cal, 1.0)
    q_hat = max(float(np.quantile(nc, qlev, method="higher")), 0.0)

    m1t, m2t = pred(X_te)
    clo = np.clip(m1t - q_hat + R_SHIFT, 1.0, 5.0)
    chi = np.clip(m2t + q_hat + R_SHIFT, 1.0, 5.0)
    mid = (clo + chi) / 2.0

    cov_c = float(np.mean((clo <= y_te) & (y_te <= chi)))
    wid_c = float(np.mean(chi - clo))
    mae_m = float(np.mean(np.abs(mid - y_te)))
    mae_l = float(np.mean(np.abs(llm_scores[idx_te] - y_te)))

    print(f"  [{label:12s}]  cov={cov_c:.3f}  wid={wid_c:.3f}  "
          f"MAE_mid={mae_m:.3f}  MAE_llm={mae_l:.3f}  q̂={q_hat:.3f}")
    return dict(label=label, q_hat=q_hat, cov_c=cov_c, wid_c=wid_c,
                mae_mid=mae_m, mae_llm=mae_l,
                clo=clo, chi=chi, mid=mid, y_te=y_te,
                loss_hist=loss_hist, model=model)

# ── Run ───────────────────────────────────────────────────
results = {}
if PER_DIMENSION:
    print(f"\n[Train] Per-dimension  (Condition C1 — BERT + 10 raw LLM scores)")
    for i, dim in enumerate(DIMS):
        results[dim] = train_and_eval(
            Y_dims[idx_tr,i], Y_dims[idx_cal,i], Y_dims[idx_te,i], label=dim)
results["combined"] = train_and_eval(
    Y_med[idx_tr], Y_med[idx_cal], Y_med[idx_te], label="combined")

# ══════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ══════════════════════════════════════════════════════════
print("\n"+"═"*65)
print(f"  {'Dimension':<14}{'Coverage':>9}{'Width':>8}{'MAE_mid':>9}{'MAE_llm':>9}{'q̂':>7}")
print("─"*65)
for k, r in results.items():
    print(f"  {k:<14}{r['cov_c']:>9.3f}{r['wid_c']:>8.3f}"
          f"{r['mae_mid']:>9.3f}{r['mae_llm']:>9.3f}{r['q_hat']:>7.3f}")
print("═"*65)

# ══════════════════════════════════════════════════════════
#  DEMO
# ══════════════════════════════════════════════════════════
best   = results["combined"]; q_demo = best["q_hat"]; m_demo = best["model"]
ds     = raw[idx_te[0]]
da     = ds["text"][:800]; dsu = ds["decoded"]
daa    = ds.get("expert_annotations") or ds.get("turker_annotations",[])
d_sc   = [float(np.mean([a[k] for a in daa if k in a])) for k in DIMS]
d_true = float(np.median(d_sc))
dyh    = llm_judge_score(da, dsu)
dyh_m  = llm_judge_scores_n(da, dsu)

dz     = get_bert_embedding(da, dsu)

# ── C1 DEMO FEATURE BLOCK ─────────────────────────────────
d_llm_feats = np.array(dyh_m, dtype=np.float32)           # 10 raw scores
# ──────────────────────────────────────────────────────────

dx = torch.tensor(
    np.concatenate([dz, d_llm_feats]).reshape(1,-1),
    dtype=torch.float32).to(DEVICE)

m_demo.eval()
with torch.no_grad(): dm1, dm2 = m_demo(dx)
dlo  = float(np.clip(dm1.item()-q_demo+R_SHIFT, 1, 5))
dhi  = float(np.clip(dm2.item()+q_demo+R_SHIFT, 1, 5))
dmid = (dlo+dhi)/2.0

print("\n"+"═"*55)
print("  DEMO — Condition C1 (BERT + 10 raw LLM scores)")
print("═"*55)
print(f"  Human ground truth          : {d_true:.2f}")
print(f"  LLM scores (10 runs)        : {[round(s,1) for s in dyh_m]}")
print(f"  LLM mean / std              : {np.mean(dyh_m):.2f} / {np.std(dyh_m):.2f}")
print(f"  Continuous interval  [L, U] : [{dlo:.2f}, {dhi:.2f}]")
print(f"  Midpoint                    : {dmid:.2f}")
print(f"  Confidence level            : {int((1-ALPHA)*100)}%")
print(f"  Ground truth covered?       : {'YES ✓' if dlo<=d_true<=dhi else 'NO ✗'}")
if PER_DIMENSION:
    print(f"\n  Per-dimension:")
    for i, dim in enumerate(DIMS):
        dr=results[dim]; dq=dr["q_hat"]; dm=dr["model"]; dm.eval()
        with torch.no_grad(): di1,di2 = dm(dx)
        lo_i=float(np.clip(di1.item()-dq+R_SHIFT,1,5))
        hi_i=float(np.clip(di2.item()+dq+R_SHIFT,1,5))
        print(f"    {dim:<14}: true={d_sc[i]:.2f}  [{lo_i:.2f},{hi_i:.2f}]"
              f"  mid={(lo_i+hi_i)/2:.2f}  {'✓' if lo_i<=d_sc[i]<=hi_i else '✗'}")
print("═"*55)

# ══════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════
def sorted_interval_plot(ax, r, title):
    y=r["y_te"]; lo=r["clo"]; hi=r["chi"]; mid=r["mid"]
    order=np.argsort(y); ys,ls,hs,ms=y[order],lo[order],hi[order],mid[order]
    xs=np.arange(len(ys))
    ax.fill_between(xs,ls,hs,alpha=0.25,color="#4C72B0",label="90% interval")
    ax.plot(xs,ms,color="#4C72B0",lw=1.5,label="Midpoint")
    cov=(ls<=ys)&(ys<=hs)
    ax.scatter(xs[cov],ys[cov],s=18,color="green",zorder=5,label="Covered")
    ax.scatter(xs[~cov],ys[~cov],s=28,color="red",zorder=5,marker="x",label="Missed")
    ax.set_ylim(0.8,5.2); ax.set_xlabel("Sample (sorted)",fontsize=9)
    ax.set_ylabel("Score",fontsize=9); ax.set_title(title,fontsize=10,fontweight="bold")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

def coverage_width_bar(ax_cov, ax_wid):
    keys=list(results.keys()); cols=[COLORS.get(k,"#888") for k in keys]
    covs=[results[k]["cov_c"] for k in keys]; wids=[results[k]["wid_c"] for k in keys]
    x=np.arange(len(keys))
    ax_cov.bar(x,covs,color=cols,alpha=0.85)
    ax_cov.axhline(1-ALPHA,color="red",ls="--",lw=1.5,label=f"{int((1-ALPHA)*100)}% target")
    ax_cov.set_xticks(x); ax_cov.set_xticklabels(keys,fontsize=8)
    ax_cov.set_ylabel("Coverage"); ax_cov.set_ylim(0.5,1.05)
    ax_cov.set_title("Coverage by Dimension",fontsize=10,fontweight="bold")
    ax_cov.legend(fontsize=8); ax_cov.grid(axis="y",alpha=0.3)
    ax_wid.bar(x,wids,color=cols,alpha=0.85)
    ax_wid.set_xticks(x); ax_wid.set_xticklabels(keys,fontsize=8)
    ax_wid.set_ylabel("Avg Width")
    ax_wid.set_title("Interval Width by Dimension",fontsize=10,fontweight="bold")
    ax_wid.grid(axis="y",alpha=0.3)

def mae_bar(ax):
    keys=list(results.keys())
    mae_m=[results[k]["mae_mid"] for k in keys]; mae_l=[results[k]["mae_llm"] for k in keys]
    x=np.arange(len(keys)); w=0.35
    ax.bar(x-w/2,mae_l,w,label="Raw LLM",color="#E57373",alpha=0.9)
    ax.bar(x+w/2,mae_m,w,label="Midpoint (C1)",color="#4DB6AC",alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(keys,fontsize=8)
    ax.set_ylabel("MAE vs Human")
    ax.set_title("MAE: Raw LLM vs C1 Midpoint",fontsize=10,fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y",alpha=0.3)

def loss_curve(ax, r, title):
    ax.plot(r["loss_hist"],color="#4C72B0",lw=1.5)
    ax.set_xlabel("Epoch",fontsize=9); ax.set_ylabel("Tube Loss",fontsize=9)
    ax.set_title(title,fontsize=10,fontweight="bold"); ax.grid(alpha=0.3)

def midpoint_vs_truth(ax, r, title):
    y=r["y_te"]; m=r["mid"]
    ax.scatter(y,m,alpha=0.6,s=25,color="#4C72B0")
    ax.plot([0.8,5.2],[0.8,5.2],"r--",lw=1.5,label="Perfect")
    ax.set_xlabel("Human Ground Truth",fontsize=9); ax.set_ylabel("Midpoint",fontsize=9)
    ax.set_title(title,fontsize=10,fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

def width_vs_error(ax, r, title):
    w=r["chi"]-r["clo"]; e=np.abs(r["mid"]-r["y_te"])
    ax.scatter(w,e,alpha=0.5,s=22,color="#DD8452")
    corr=float(np.corrcoef(w,e)[0,1])
    ax.text(0.05,0.92,f"r={corr:.2f}",transform=ax.transAxes,fontsize=9)
    ax.set_xlabel("Interval Width",fontsize=9); ax.set_ylabel("|Mid-Truth|",fontsize=9)
    ax.set_title(title,fontsize=10,fontweight="bold"); ax.grid(alpha=0.3)

fig1 = plt.figure(figsize=(16,10))
fig1.suptitle(f"Condition C1 — BERT + 10 LLM scores  |  TubeNet + CP",
              fontsize=12,fontweight="bold",y=0.98)
gs = gridspec.GridSpec(2,3,figure=fig1,hspace=0.45,wspace=0.35)
sorted_interval_plot(fig1.add_subplot(gs[0,:2]),results["combined"],
                     "Sorted Prediction Intervals — Combined (90%)")
loss_curve(fig1.add_subplot(gs[0,2]),results["combined"],"TubeNet Training Loss")
coverage_width_bar(fig1.add_subplot(gs[1,0]),fig1.add_subplot(gs[1,1]))
mae_bar(fig1.add_subplot(gs[1,2]))
fig1.savefig(os.path.join(OUT_DIR,"fig1_main.png"),dpi=150,bbox_inches="tight")

if PER_DIMENSION:
    fig2,axes=plt.subplots(2,2,figsize=(14,9))
    fig2.suptitle(f"C1 Per-Dimension Intervals (α={ALPHA})",fontsize=12,fontweight="bold")
    for ax,dim in zip(axes.flat,DIMS):
        sorted_interval_plot(ax,results[dim],
            f"{dim.capitalize()}  cov={results[dim]['cov_c']:.2f}  wid={results[dim]['wid_c']:.2f}")
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR,"fig2_per_dim.png"),dpi=150,bbox_inches="tight")

fig3,axes=plt.subplots(1,2,figsize=(12,5))
fig3.suptitle("C1 Diagnostics",fontsize=12,fontweight="bold")
midpoint_vs_truth(axes[0],results["combined"],"Midpoint vs Human Ground Truth")
width_vs_error(axes[1],results["combined"],"Width vs Prediction Error")
fig3.tight_layout()
fig3.savefig(os.path.join(OUT_DIR,"fig3_diagnostics.png"),dpi=150,bbox_inches="tight")

if PER_DIMENSION:
    fig4,axes=plt.subplots(1,4,figsize=(16,4))
    fig4.suptitle("C1 — Tube Loss Curves per Dimension",fontsize=12,fontweight="bold")
    for ax,dim in zip(axes,DIMS): loss_curve(ax,results[dim],dim.capitalize())
    fig4.tight_layout()
    fig4.savefig(os.path.join(OUT_DIR,"fig4_loss_curves.png"),dpi=150,bbox_inches="tight")

plt.close("all")
print(f"\n[Plots] Saved to ./{OUT_DIR}/")

save_json(os.path.join(OUT_DIR,"results_C1.json"),{
    k:{"coverage":round(r["cov_c"],4),"avg_width":round(r["wid_c"],4),
       "midpoint_mae":round(r["mae_mid"],4),"raw_llm_mae":round(r["mae_llm"],4),
       "q_hat":round(r["q_hat"],4),"condition":"C1_bert+10llm_raw",
       "alpha":ALPHA,"t":T_CONF,"delta":DELTA}
    for k,r in results.items()})
print(f"[Saved] {OUT_DIR}/results_C1.json\nDone.")
