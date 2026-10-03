"""Fig 2 (screen results, both HCP constructions) and Fig 4 (coefficient metric fails).
Primary metric throughout: mean_ratio_pairs. IEEE double-column sizing."""
import os, json
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt

R="/workspace/meridian-identifiability/causalbench/results/screen"
OUT="/workspace/meridian-identifiability/framework/figures"; os.makedirs(OUT,exist_ok=True)
mpl.rcParams.update({"font.size":9,"axes.labelsize":9,"xtick.labelsize":9,
    "ytick.labelsize":8,"legend.fontsize":7.5,"axes.linewidth":0.8,
    "pdf.fonttype":42,"ps.fonttype":42})
def g(p,*k):
    d=json.load(open(os.path.join(R,p)))
    for x in k: d=d[x]
    return float(d)

# ---------------- FIG 2 ----------------
v={"K562":g("k562_filt0_n200_d10.json","mean_ratio_pairs"),
   "Norman":g("norman.json","summary","primary_mean_ratio_pairs"),
   "RPE1":g("rpe1_filt0_n200_d10.json","mean_ratio_pairs")}
HCP_MATCHED, HCP_OWN = 17.3, 0.69
gate=(g("rpe1_filt0_n200_d10_STEP0.json","mean_ratio_pairs"),
      g("k562_filt0_n200_d10_STEP0.json","mean_ratio_pairs"))
gate=(min(gate),max(gate))
print("FIG2:", {k:round(x,3) for k,x in v.items()},
      f"HCP {HCP_OWN}/{HCP_MATCHED}  gate {gate[0]:.3f}-{gate[1]:.3f}")

fig,ax=plt.subplots(figsize=(7.16,3.1))
names=["K562","Norman","RPE1","HCP\n(own null)","HCP\n(matched null)"]
vals=[v["K562"],v["Norman"],v["RPE1"],HCP_OWN,HCP_MATCHED]
cols=["#0072B2","#009E73","#E69F00","#D55E00","#D55E00"]
b=ax.bar(names,vals,color=cols,edgecolor="black",lw=0.8,width=0.62)
b[4].set_hatch("///"); b[4].set_alpha(0.75)
ax.axhspan(*gate,color="0.75",alpha=0.5,zorder=0,label=f"step-0 calibration ({gate[0]:.2f}–{gate[1]:.2f})")
ax.axhline(1.0,color="k",ls="--",lw=1.0,label="within-condition null (1.0)")
for r,val in zip(b,vals):
    ax.text(r.get_x()+r.get_width()/2,r.get_height()+0.3,f"{val:.2f}",
            ha="center",va="bottom",fontweight="bold",fontsize=9)
ax.annotate("same data, different null:\nCRL fails in both cases",
            xy=(4,HCP_MATCHED),xytext=(2.75,14.5),fontsize=7,ha="center",
            arrowprops=dict(arrowstyle="->",lw=0.8,color="0.35"))
ax.set_ylabel("mean-shift ratio\n(between-environment / within-environment)")
ax.set_ylim(0,20); ax.legend(loc="upper left",frameon=False)
ax.spines[["top","right"]].set_visible(False); fig.tight_layout()
for e in("pdf","png"): fig.savefig(f"{OUT}/goyal-fig2.{e}",dpi=300,bbox_inches="tight")

# ---------------- FIG 4 ----------------
# coefficient-based metric: near-null in BOTH domains -> cannot discriminate
K={5:1.309,10:1.178}                      # K562  -- replace from JSON if keys exist
H={10:1.22,20:1.142,30:1.098,50:1.065}    # HCP LR
H2={10:1.21,20:1.130,30:1.093,50:1.060}   # HCP RL
print("FIG4: K562",K,"HCP",H)
fig,ax=plt.subplots(figsize=(3.5,2.8))
ax.axhspan(0.9,1.3,color="0.88",zorder=0,label="near-null band")
ax.axhline(1.0,color="k",ls="--",lw=1.0,label="null (1.0)")
ax.plot(list(K),list(K.values()),"o-",color="#0072B2",ms=5,label="K562")
ax.plot(list(H),list(H.values()),"D-",color="#D55E00",ms=5,label="HCP (LR)")
ax.plot(list(H2),list(H2.values()),"D--",color="#D55E00",ms=5,mfc="white",label="HCP (RL)")
ax.set_xlabel("latent dim $d$"); ax.set_ylabel("coefficient-shift ratio")
ax.set_xscale("log"); ax.set_xticks([5,10,20,30,50]); ax.set_xticklabels([5,10,20,30,50])
ax.set_ylim(0.85,1.45); ax.legend(frameon=False,fontsize=6.5,loc="upper right")
ax.spines[["top","right"]].set_visible(False); fig.tight_layout()
for e in("pdf","png"): fig.savefig(f"{OUT}/goyal-fig4.{e}",dpi=300,bbox_inches="tight")
print(f"\nwrote {OUT}/goyal-fig2 and goyal-fig4 (.pdf/.png)")
