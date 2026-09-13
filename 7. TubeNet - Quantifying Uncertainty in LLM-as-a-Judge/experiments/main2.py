"""
LLM-as-a-Judge Uncertainty — BMP Project
BERT + TubeNet (exact Tube Loss from Anand et al. 2024) + Conformal Prediction

═══════════════════════════════════════════════════════════════
  DEFAULT CONFIGURATION (after removing extreme tweaks)
═══════════════════════════════════════════════════════════════
"""
import os, json, re, time, math, warnings
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
#  CONFIG (DEFAULT VALUES)
# ══════════════════════════════════════════════════════════
DATA_PATH    = "data/model_annotations.aligned/paired/model_annotations.aligned.paired.jsonl"
CACHE_LLM    = "cache_llm_v2.json"
CACHE_EMBED  = "cache_embed_v2.json"
OUT_DIR      = "bmp_results"
os.makedirs(OUT_DIR, exist_ok=True)

MAX_SAMPLES  = 250
ALPHA        = 0.10          # fixed — 90% confidence
HIDDEN       = 256
EPOCHS       = 150
LR           = 3e-4
BATCH        = 16
SEED         = 42
DIMS         = ["coherence", "consistency", "fluency", "relevance"]
PER_DIMENSION = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED)

# ══════════════════════════════════════════════════════════
#  TUBE LOSS HYPERPARAMETERS (DEFAULT – balanced)
# ══════════════════════════════════════════════════════════
Q           = 0.92          # target coverage inside tube (slightly lower to encourage narrower intervals)
R_INIT      = 0.5
DELTA_MIN   = 0.01
DELTA_MAX   = 0.35          # width penalty – strong but not extreme
DELTA_RAMP  = 80
SOFT_TEMP   = 1.5           # smooth switch for r gradient

# Conformal: uncapped q_hat
R_GRID      = np.linspace(0.05, 0.95, 31)   # grid for post-hoc r refinement

print("=" * 66)
print("  LLM-as-a-Judge Uncertainty  |  BMP Project (v2 — Learnable r)")
print(f"  Tube Loss (Anand et al. 2024)  q={Q}  r_init={R_INIT}")
print(f"  Adaptive δ: {DELTA_MIN}→{DELTA_MAX} over {DELTA_RAMP} epochs")
print(f"  device={DEVICE}  α={ALPHA} (fixed)  r_grid={len(R_GRID)} pts  q_hat=uncapped")
print("=" * 66)

# ══════════════════════════════════════════════════════════
#  CACHE, LLM JUDGE, BERT (same as before)
# ══════════════════════════════════════════════════════════
def load_json(p):    return json.load(open(p)) if os.path.exists(p) else {}
def save_json(p, o): json.dump(o, open(p, "w"))
llm_cache   = load_json(CACHE_LLM)
embed_cache = load_json(CACHE_EMBED)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def llm_judge_score(article: str, summary: str) -> float:
    key = article[:100] + summary[:100]
    if key in llm_cache: return llm_cache[key]
    prompt = (
        "You are an expert evaluator of text summaries.\n"
        "Rate the summary quality on a scale from 1 to 5:\n"
        "1 = very poor, 2 = poor, 3 = average, 4 = good, 5 = excellent.\n\n"
        "Guidelines:\n"
        "- Use the full range of scores (1 to 5).\n"
        "- Score 3 only if the summary is truly average.\n"
        "- Give 1 or 2 for clearly bad summaries.\n"
        "- Give 4 or 5 for strong, high-quality summaries.\n"
        "- Be consistent and objective.\n\n"
        f"Article: {article[:500]}\n"
        f"Summary: {summary}\n\n"
        "Return ONLY one number (1, 2, 3, 4, or 5)."
    )
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=5)
        m = re.search(r"[1-5]", resp.choices[0].message.content.strip())
        score = float(m.group()) if m else 3.0
    except Exception:
        score = 3.0
    llm_cache[key] = score; save_json(CACHE_LLM, llm_cache)
    return score

print("[Init] Loading SentenceTransformer …")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_bert_embedding(article: str, summary: str) -> np.ndarray:
    key = summary[:150]
    if key in embed_cache: return np.array(embed_cache[key], dtype=np.float32)
    a = embedder.encode(article[:512], show_progress_bar=False, normalize_embeddings=True)
    s = embedder.encode(summary[:256], show_progress_bar=False, normalize_embeddings=True)
    vec = np.concatenate([a, s])
    embed_cache[key] = vec.tolist(); save_json(CACHE_EMBED, embed_cache)
    return vec.astype(np.float32)

# ══════════════════════════════════════════════════════════
#  TUBE LOSS (exact – no extra multipliers)
# ══════════════════════════════════════════════════════════
def tube_loss(f1: torch.Tensor, f2: torch.Tensor, y: torch.Tensor,
              r: torch.Tensor, delta: float) -> torch.Tensor:
    c1 = (1 - Q) * (f2 - y)
    c2 = (1 - Q) * (y - f1)
    c3 = Q * (f1 - y)
    c4 = Q * (y - f2)

    w_shift = torch.sigmoid(SOFT_TEMP * (y - r * (f1 + f2)))
    loss_part1 = w_shift * c1 + (1.0 - w_shift) * c2
    loss_part2 = torch.where(f1 > y, c3, c4)
    inside = (y >= f1) & (y <= f2)
    final_loss = torch.where(inside, loss_part1, loss_part2)
    final_loss = final_loss + delta * torch.abs(f1 - f2)
    return final_loss.mean()

def get_delta(epoch: int) -> float:
    if epoch >= DELTA_RAMP:
        return DELTA_MAX
    t = epoch / DELTA_RAMP
    return DELTA_MIN + (DELTA_MAX - DELTA_MIN) * t

# ══════════════════════════════════════════════════════════
#  LOAD DATA (no aggressive filtering)
# ══════════════════════════════════════════════════════════
print("\n[Data] Loading SummEval …")
all_data = [json.loads(l) for l in open(DATA_PATH, encoding="utf-8", errors="ignore")]
np.random.seed(SEED)
np.random.shuffle(all_data)

raw = all_data[:MAX_SAMPLES]
X_list, Y_dims_list, llm_scores_list = [], [], []

for sample in tqdm(raw, desc="Features"):
    art = sample["text"][:800]; summ = sample["decoded"]
    anns = sample.get("expert_annotations") or sample.get("turker_annotations", [])
    if not anns: continue
    dim_scores = [float(np.mean([a[d] for a in anns if d in a])) for d in DIMS]
    # Removed aggressive std filter (was >0.5)
    z = get_bert_embedding(art, summ)
    yh = llm_judge_score(art, summ)
    X_list.append(np.concatenate([z, [yh]]))
    Y_dims_list.append(dim_scores)
    llm_scores_list.append(yh)
    time.sleep(0.03)

X = np.array(X_list, dtype=np.float32)
Y_dims = np.array(Y_dims_list, dtype=np.float32)
Y_med = np.median(Y_dims, axis=1)
N = len(Y_med)
print(f"[Data] N={N}  X={X.shape}")

idx = np.arange(N)
idx_tr, idx_tmp = train_test_split(idx, test_size=0.40, random_state=SEED)
idx_cal, idx_te = train_test_split(idx_tmp, test_size=0.50, random_state=SEED)
X_tr = X[idx_tr]; X_cal = X[idx_cal]; X_te = X[idx_te]
print(f"[Split] train={len(idx_tr)}  cal={len(idx_cal)}  test={len(idx_te)}")

# ══════════════════════════════════════════════════════════
#  TUBENET
# ══════════════════════════════════════════════════════════
class TubeNet(nn.Module):
    def __init__(self, in_dim, r_init=R_INIT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN), nn.LayerNorm(HIDDEN), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(HIDDEN, HIDDEN // 2), nn.LayerNorm(HIDDEN // 2), nn.GELU(), nn.Dropout(0.1),
        )
        self.head_mid = nn.Linear(HIDDEN // 2, 1)
        self.head_half = nn.Linear(HIDDEN // 2, 1)
        nn.init.constant_(self.head_half.bias, -2.0)
        r_logit_init = math.log(r_init / (1.0 - r_init))
        self.r_logit = nn.Parameter(torch.tensor(r_logit_init))

    @property
    def r(self):
        return torch.sigmoid(self.r_logit)

    def forward(self, x):
        h = self.net(x)
        mid = self.head_mid(h).squeeze(-1)
        half = nn.functional.softplus(self.head_half(h)).squeeze(-1)
        return mid - half, mid + half

class TabDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

IN_DIM = X.shape[1]

# ══════════════════════════════════════════════════════════
#  CALIBRATION r REFINEMENT
# ══════════════════════════════════════════════════════════
def refine_r_on_cal(model, X_cal_np, y_cal, alpha=ALPHA):
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X_cal_np, dtype=torch.float32).to(DEVICE)
        f1, f2 = model(xt)
        f1 = f1.cpu().numpy(); f2 = f2.cpu().numpy()
    width_raw = f2 - f1
    best_shift = 0.0
    best_r_val = float(model.r.item())
    best_width = np.mean(width_raw) * 100
    target_cov = 1.0 - alpha
    for r_cand in R_GRID:
        shift = (r_cand - 0.5) * np.mean(width_raw) * 0.5
        f1_s = f1 + shift
        f2_s = f2 + shift
        cov = np.mean((f1_s <= y_cal) & (y_cal <= f2_s))
        wid = np.mean(f2_s - f1_s)
        if cov >= target_cov and wid < best_width:
            best_width = wid
            best_shift = shift
            best_r_val = r_cand
    return best_shift, best_r_val

# ══════════════════════════════════════════════════════════
#  TRAIN + EVAL
# ══════════════════════════════════════════════════════════
def train_and_eval(y_tr, y_cal, y_te, label="combined"):
    model = TubeNet(IN_DIM).to(DEVICE)
    r_params = [model.r_logit]
    net_params = [p for n, p in model.named_parameters() if n != "r_logit"]
    opt = torch.optim.Adam([
        {"params": net_params, "lr": LR, "weight_decay": 1e-4},
        {"params": r_params,   "lr": LR * 50, "weight_decay": 0.0},
    ])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ldr = DataLoader(TabDS(X_tr, y_tr), batch_size=BATCH, shuffle=True)

    loss_hist, r_hist, delta_hist = [], [], []
    for ep in range(1, EPOCHS + 1):
        model.train()
        ep_loss = 0.0
        delta_ep = get_delta(ep)
        for xb, yb in ldr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            f1, f2 = model(xb)
            loss = tube_loss(f1, f2, yb, r=model.r, delta=delta_ep)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * len(yb)
        sch.step()
        loss_hist.append(ep_loss / len(y_tr))
        r_hist.append(float(model.r.item()))
        delta_hist.append(delta_ep)

    learned_r = float(model.r.item())
    model.eval()

    @torch.no_grad()
    def pred(Xnp):
        xt = torch.tensor(Xnp, dtype=torch.float32).to(DEVICE)
        f1, f2 = model(xt)
        return f1.cpu().numpy(), f2.cpu().numpy()

    cal_shift, refined_r = refine_r_on_cal(model, X_cal, y_cal)

    m1c, m2c = pred(X_cal)
    m1c += cal_shift; m2c += cal_shift
    nc = np.maximum(m1c - y_cal, y_cal - m2c)
    n_cal = len(y_cal)
    qlev = min(math.ceil((n_cal + 1) * (1 - ALPHA)) / n_cal, 1.0)
    q_hat = float(np.quantile(nc, qlev))

    m1t, m2t = pred(X_te)
    m1t += cal_shift; m2t += cal_shift
    clo = np.clip(m1t - q_hat, 1.0, 5.0)
    chi = np.clip(m2t + q_hat, 1.0, 5.0)
    mid = (clo + chi) / 2.0

    cov_c = float(np.mean((clo <= y_te) & (y_te <= chi)))
    wid_c = float(np.mean(chi - clo))
    mae_m = float(np.mean(np.abs(mid - y_te)))
    mae_l = float(np.mean(np.abs(X_te[:, -1] - y_te)))

    print(f"  [{label:12s}]  cov={cov_c:.3f}  wid={wid_c:.3f}  "
          f"MAE_mid={mae_m:.3f}  MAE_llm={mae_l:.3f}  "
          f"q̂={q_hat:.3f}  r_learned={learned_r:.3f}  r_refined={refined_r:.3f}  "
          f"shift={cal_shift:+.3f}")
    return dict(label=label, q_hat=q_hat, cov_c=cov_c, wid_c=wid_c,
                mae_mid=mae_m, mae_llm=mae_l,
                clo=clo, chi=chi, mid=mid, y_te=y_te,
                loss_hist=loss_hist, r_hist=r_hist, delta_hist=delta_hist,
                learned_r=learned_r, refined_r=refined_r, cal_shift=cal_shift,
                model=model)

# ── Run ───────────────────────────────────────────────────
results = {}
if PER_DIMENSION:
    print(f"\n[Train] Per-dimension TubeNet (q={Q}, r=learnable, δ={DELTA_MIN}→{DELTA_MAX})")
    for i, dim in enumerate(DIMS):
        results[dim] = train_and_eval(
            Y_dims[idx_tr, i], Y_dims[idx_cal, i], Y_dims[idx_te, i], label=dim)

results["combined"] = train_and_eval(
    Y_med[idx_tr], Y_med[idx_cal], Y_med[idx_te], label="combined")

# ══════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ══════════════════════════════════════════════════════════
print("\n" + "═" * 90)
print(f"  {'Dimension':<14} {'Coverage':>9} {'Width':>8} {'MAE_mid':>9} "
      f"{'MAE_llm':>9} {'q̂':>7} {'r_learn':>8} {'r_refin':>8} {'shift':>7}")
print("─" * 90)
for k, r in results.items():
    print(f"  {k:<14} {r['cov_c']:>9.3f} {r['wid_c']:>8.3f} "
          f"{r['mae_mid']:>9.3f} {r['mae_llm']:>9.3f} {r['q_hat']:>7.3f} "
          f"{r['learned_r']:>8.3f} {r['refined_r']:>8.3f} {r['cal_shift']:>+7.3f}")
print("═" * 90)

# ══════════════════════════════════════════════════════════
#  DEMO
# ══════════════════════════════════════════════════════════
best   = results["combined"]
q_demo = best["q_hat"]
m_demo = best["model"]
s_demo = best["cal_shift"]

ds  = raw[idx_te[0]]
da  = ds["text"][:800]; dsu = ds["decoded"]
daa = ds.get("expert_annotations") or ds.get("turker_annotations", [])
d_sc   = [float(np.mean([a[k] for a in daa if k in a])) for k in DIMS]
d_true = float(np.median(d_sc))
dz     = get_bert_embedding(da, dsu)
dyh    = llm_judge_score(da, dsu)
dx     = torch.tensor(np.concatenate([dz, [dyh]]).reshape(1, -1),
                      dtype=torch.float32).to(DEVICE)

m_demo.eval()
with torch.no_grad():
    dm1, dm2 = m_demo(dx)

dlo  = float(np.clip(dm1.item() + s_demo - q_demo, 1, 5))
dhi  = float(np.clip(dm2.item() + s_demo + q_demo, 1, 5))
dmid = (dlo + dhi) / 2.0

print("\n" + "═" * 60)
print("  DEMO — single sample end-to-end")
print("═" * 60)
print(f"  Human ground truth          : {d_true:.2f}")
print(f"  LLM raw score  ŷ            : {dyh:.0f}")
print(f"  Tube Loss params            : q={Q}  r_learned={best['learned_r']:.3f}")
print(f"  Calibration shift           : {s_demo:+.3f}  (r_refined={best['refined_r']:.3f})")
print(f"  Continuous interval  [L, U] : [{dlo:.2f}, {dhi:.2f}]")
print(f"  Midpoint (calibrated score) : {dmid:.2f}")
print(f"  Confidence level            : {int((1-ALPHA)*100)}%  (fixed α={ALPHA})")
print(f"  Ground truth covered?       : {'YES ✓' if dlo <= d_true <= dhi else 'NO ✗'}")

if PER_DIMENSION:
    print(f"\n  Per-dimension (q={Q}, learnable r):")
    for i, dim in enumerate(DIMS):
        dr = results[dim]; dq = dr["q_hat"]; dm = dr["model"]; ds_i = dr["cal_shift"]
        dm.eval()
        with torch.no_grad(): di1, di2 = dm(dx)
        lo_i  = float(np.clip(di1.item() + ds_i - dq, 1, 5))
        hi_i  = float(np.clip(di2.item() + ds_i + dq, 1, 5))
        mid_i = (lo_i + hi_i) / 2.0
        print(f"    {dim:<14}: true={d_sc[i]:.2f}  interval=[{lo_i:.2f},{hi_i:.2f}]  "
              f"mid={mid_i:.2f}  r={dr['learned_r']:.3f}  "
              f"{'✓' if lo_i <= d_sc[i] <= hi_i else '✗'}")
print("═" * 60)

# ══════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════
COLORS = {"coherence": "#4C72B0", "consistency": "#DD8452",
          "fluency": "#55A868", "relevance": "#C44E52", "combined": "#8172B2"}


def sorted_interval_plot(ax, r, title):
    y = r["y_te"]; lo = r["clo"]; hi = r["chi"]; mid = r["mid"]
    order = np.argsort(y)
    ys, ls, hs, ms = y[order], lo[order], hi[order], mid[order]
    xs = np.arange(len(ys))
    ax.fill_between(xs, ls, hs, alpha=0.25, color="#4C72B0", label="90% interval")
    ax.plot(xs, ms, color="#4C72B0", lw=1.5, label="Midpoint")
    cov = (ls <= ys) & (ys <= hs)
    ax.scatter(xs[cov],  ys[cov],  s=18, color="green", zorder=5, label="Covered")
    ax.scatter(xs[~cov], ys[~cov], s=28, color="red",   zorder=5, marker="x", label="Missed")
    ax.set_ylim(0.8, 5.2); ax.set_xlabel("Sample (sorted)", fontsize=9)
    ax.set_ylabel("Score", fontsize=9); ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)


def coverage_width_bar(ax_cov, ax_wid):
    keys = list(results.keys())
    cols = [COLORS.get(k, "#888") for k in keys]
    covs = [results[k]["cov_c"] for k in keys]
    wids = [results[k]["wid_c"] for k in keys]
    x = np.arange(len(keys))
    ax_cov.bar(x, covs, color=cols, alpha=0.85)
    ax_cov.axhline(1 - ALPHA, color="red", ls="--", lw=1.5, label=f"{int((1-ALPHA)*100)}% target")
    ax_cov.set_xticks(x); ax_cov.set_xticklabels(keys, fontsize=8)
    ax_cov.set_ylabel("Coverage"); ax_cov.set_ylim(0.5, 1.05)
    ax_cov.set_title("Coverage by Dimension", fontsize=10, fontweight="bold")
    ax_cov.legend(fontsize=8); ax_cov.grid(axis="y", alpha=0.3)
    ax_wid.bar(x, wids, color=cols, alpha=0.85)
    ax_wid.set_xticks(x); ax_wid.set_xticklabels(keys, fontsize=8)
    ax_wid.set_ylabel("Avg Width")
    ax_wid.set_title("Interval Width by Dimension", fontsize=10, fontweight="bold")
    ax_wid.grid(axis="y", alpha=0.3)


def mae_bar(ax):
    keys  = list(results.keys())
    mae_m = [results[k]["mae_mid"] for k in keys]
    mae_l = [results[k]["mae_llm"] for k in keys]
    x = np.arange(len(keys)); w = 0.35
    ax.bar(x - w / 2, mae_l, w, label="Raw LLM",            color="#E57373", alpha=0.9)
    ax.bar(x + w / 2, mae_m, w, label="Midpoint (TubeNet)", color="#4DB6AC", alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(keys, fontsize=8)
    ax.set_ylabel("MAE vs Human")
    ax.set_title("MAE: Raw LLM vs Calibrated Midpoint", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)


def loss_curve(ax, r, title):
    ax.plot(r["loss_hist"], color="#4C72B0", lw=1.5)
    ax.set_xlabel("Epoch", fontsize=9); ax.set_ylabel("Tube Loss", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold"); ax.grid(alpha=0.3)


def r_evolution_plot(ax, r, title):
    """NEW: show how the learned r evolves during training."""
    ax.plot(r["r_hist"], color="#C44E52", lw=1.5, label="Learned r")
    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.5, label="Symmetric (r=0.5)")
    ax.axhline(r["refined_r"], color="#55A868", ls=":", lw=1.5,
               label=f"Refined r={r['refined_r']:.2f}")
    ax.set_xlabel("Epoch", fontsize=9); ax.set_ylabel("r value", fontsize=9)
    ax.set_ylim(0, 1); ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)


def delta_schedule_plot(ax, r, title):
    """NEW: show the adaptive delta schedule."""
    ax.plot(r["delta_hist"], color="#DD8452", lw=1.5)
    ax.set_xlabel("Epoch", fontsize=9); ax.set_ylabel("δ (width penalty)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold"); ax.grid(alpha=0.3)


def midpoint_vs_truth(ax, r, title):
    y = r["y_te"]; m = r["mid"]
    ax.scatter(y, m, alpha=0.6, s=25, color="#4C72B0")
    mn, mx = 0.8, 5.2
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect")
    ax.set_xlabel("Human Ground Truth", fontsize=9); ax.set_ylabel("Midpoint", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)


def width_vs_error(ax, r, title):
    w = r["chi"] - r["clo"]; e = np.abs(r["mid"] - r["y_te"])
    ax.scatter(w, e, alpha=0.5, s=22, color="#DD8452")
    corr = float(np.corrcoef(w, e)[0, 1])
    ax.text(0.05, 0.92, f"r = {corr:.2f}", transform=ax.transAxes, fontsize=9)
    ax.set_xlabel("Interval Width (uncertainty)", fontsize=9)
    ax.set_ylabel("|Midpoint - Truth|", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold"); ax.grid(alpha=0.3)


# ── Fig 1 — main ─────────────────────────────────────────
fig1 = plt.figure(figsize=(16, 10))
fig1.suptitle(
    f"LLM-as-a-Judge Uncertainty  |  BERT + TubeNet "
    f"(q={Q}, learnable r, adaptive δ) + Conformal Prediction",
    fontsize=12, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig1, hspace=0.45, wspace=0.35)
sorted_interval_plot(fig1.add_subplot(gs[0, :2]), results["combined"],
                     "Sorted Prediction Intervals — Combined (90% Guarantee)")
loss_curve(fig1.add_subplot(gs[0, 2]), results["combined"], "TubeNet Training Loss")
coverage_width_bar(fig1.add_subplot(gs[1, 0]), fig1.add_subplot(gs[1, 1]))
mae_bar(fig1.add_subplot(gs[1, 2]))
fig1.savefig(os.path.join(OUT_DIR, "fig1_main.png"), dpi=150, bbox_inches="tight")

# ── Fig 2 — per-dimension intervals ──────────────────────
if PER_DIMENSION:
    fig2, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig2.suptitle(
        f"Per-Dimension Intervals (q={Q}, learnable r, adaptive δ, α={ALPHA})",
        fontsize=12, fontweight="bold")
    for ax, dim in zip(axes.flat, DIMS):
        sorted_interval_plot(ax, results[dim],
            f"{dim.capitalize()}  cov={results[dim]['cov_c']:.2f}  "
            f"wid={results[dim]['wid_c']:.2f}  r={results[dim]['learned_r']:.2f}")
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, "fig2_per_dim.png"), dpi=150, bbox_inches="tight")

# ── Fig 3 — diagnostics ──────────────────────────────────
fig3, axes = plt.subplots(1, 2, figsize=(12, 5))
fig3.suptitle("Diagnostics", fontsize=12, fontweight="bold")
midpoint_vs_truth(axes[0], results["combined"], "Midpoint vs Human Ground Truth")
width_vs_error(axes[1], results["combined"], "Width vs Prediction Error")
fig3.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, "fig3_diagnostics.png"), dpi=150, bbox_inches="tight")

# ── Fig 4 — loss curves per dim ──────────────────────────
if PER_DIMENSION:
    fig4, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig4.suptitle("Tube Loss Training Curves per Dimension", fontsize=12, fontweight="bold")
    for ax, dim in zip(axes, DIMS):
        loss_curve(ax, results[dim], dim.capitalize())
    fig4.tight_layout()
    fig4.savefig(os.path.join(OUT_DIR, "fig4_loss_curves.png"), dpi=150, bbox_inches="tight")

# ── Fig 5 — NEW: r evolution + delta schedule ─────────────
fig5, axes = plt.subplots(1, 3, figsize=(16, 5))
fig5.suptitle("Shifting Property: Learned r Evolution & Adaptive δ Schedule",
              fontsize=12, fontweight="bold")
r_evolution_plot(axes[0], results["combined"], "Combined: r Evolution")
delta_schedule_plot(axes[1], results["combined"], "Adaptive δ Schedule")
# r values bar chart across dimensions
dim_keys = [k for k in results.keys()]
r_learned = [results[k]["learned_r"] for k in dim_keys]
r_refined = [results[k]["refined_r"] for k in dim_keys]
x = np.arange(len(dim_keys)); w = 0.35
axes[2].bar(x - w / 2, r_learned, w, label="Learned r", color="#4C72B0", alpha=0.85)
axes[2].bar(x + w / 2, r_refined, w, label="Refined r", color="#55A868", alpha=0.85)
axes[2].axhline(0.5, color="gray", ls="--", lw=1, alpha=0.5)
axes[2].set_xticks(x); axes[2].set_xticklabels(dim_keys, fontsize=8)
axes[2].set_ylabel("r value"); axes[2].set_ylim(0, 1)
axes[2].set_title("Learned vs Refined r per Dimension", fontsize=10, fontweight="bold")
axes[2].legend(fontsize=8); axes[2].grid(axis="y", alpha=0.3)
fig5.tight_layout()
fig5.savefig(os.path.join(OUT_DIR, "fig5_shifting.png"), dpi=150, bbox_inches="tight")

plt.close("all")
print(f"\n[Plots] Saved to ./{OUT_DIR}/")

# ══════════════════════════════════════════════════════════
#  SAVE JSON
# ══════════════════════════════════════════════════════════
save_json(os.path.join(OUT_DIR, "results.json"), {
    k: {"coverage": round(r["cov_c"], 4), "avg_width": round(r["wid_c"], 4),
        "midpoint_mae": round(r["mae_mid"], 4), "raw_llm_mae": round(r["mae_llm"], 4),
        "q_hat": round(r["q_hat"], 4), "alpha": ALPHA,
        "q": Q, "r_init": R_INIT, "r_learned": round(r["learned_r"], 4),
        "r_refined": round(r["refined_r"], 4), "cal_shift": round(r["cal_shift"], 4),
        "delta_min": DELTA_MIN, "delta_max": DELTA_MAX, "q_hat": "uncapped"}
    for k, r in results.items()})

print(f"[Saved] {OUT_DIR}/results.json\nDone.")
