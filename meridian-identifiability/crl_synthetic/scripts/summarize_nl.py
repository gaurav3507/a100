import json, glob, collections, numpy as np
rows = collections.defaultdict(list)
for f in glob.glob("/workspace/results_nonlinear_mixing/seeds/*.json"):
    r = json.load(open(f))
    rows[(r["strength"], r["cond"])].append(r)

def ms(rs, k):
    v = np.array([x[k] for x in rs], float)
    return float(np.nanmean(v)), float(np.nanstd(v))

hdr = ("str", "cond", "n", "CF-MCC", "cf-init", "rand", "MSE floor", "cos", "rank1", "t_fav")
print("{:>5} {:>22} {:>3} {:>13} {:>13} {:>13} {:>13} {:>6} {:>9} {:>6}".format(*hdr))
print("-" * 118)
for k in sorted(rows, key=lambda x: (x[0], x[1])):
    rs = rows[k]
    cf, ci, rd, mm = ms(rs,"cf_mcc"), ms(rs,"cfinit_mcc"), ms(rs,"rand_mcc"), ms(rs,"mse_mcc")
    co, r1, tf = ms(rs,"cos_mean"), ms(rs,"rank1_mean"), ms(rs,"true_favored_frac")
    print("{:>5} {:>22} {:>3} {:>5.3f}±{:<7.3f} {:>5.3f}±{:<7.3f} {:>5.3f}±{:<7.3f} {:>5.3f}±{:<7.3f} {:>6.3f} {:>9.2e} {:>6.2f}".format(
        k[0], k[1], len(rs), cf[0],cf[1], ci[0],ci[1], rd[0],rd[1], mm[0],mm[1], co[0], r1[0], tf[0]))
print("\nCF-MCC vs MSE floor is the verdict. cond='independent_recursive' is the transfer-relevant rung.")
