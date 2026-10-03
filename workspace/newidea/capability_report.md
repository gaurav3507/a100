# Capability Guards

## B1 G-CaRL substrate: PASS

- Pinned commit: `0020bfce34736d61d70ab8175f061d02951a7ed4`
- Exact Sim1 configuration: 3 groups, 10 dimensions per group, 65536 samples, DAG, Laplace, 2 inter-group neighbors, 1 intra-group neighbor, no latent confounders
- True Z: second return `s` from `generate_dataset(...)`, shape `(65536, 3, 10)`
- Inter-group truth: `lam1`, converted by group-pair order to parent-row, child-column adjacency
- Intra-group truth: `lamin1`, block diagonal, always `EXPLORATORY_ONLY`
- Load-bearing target: inter-group graph only

## B2 causal discovery: PASS

| method | implementation | version | import | smoke fit | graph returned | included |
|---|---|---:|---|---|---|---|
| PC-Pearson | causal-learn | 0.1.4.8 | PASS | PASS | PASS | YES |
| PC-nonparanormal-preproc | causal-learn | 0.1.4.8 | PASS | PASS | PASS | YES |
| DirectLiNGAM | lingam | 1.12.2 | PASS | PASS | PASS | YES |
| NOTEARS-linear | xunzheng/notears, exact cached covariance objective | 4a9ab19fe502e392503da331773c5223d82d3666 | PASS | PASS | PASS | YES |
| NOTEARS-MLP | gCastle | 1.0.4 | PASS | PASS | PASS | YES |
| GOLEM | gCastle | 1.0.4 | PASS | PASS | PASS | YES |
| CAM | R CAM | unavailable | FAIL | FAIL | FAIL | NO |

CAM was excluded because an R runtime was unavailable. The final set contains six usable estimators.

## B3 graph semantics and scoring: PASS

- G-CaRL truth: parent-row, child-column
- DirectLiNGAM raw adjacency: effect-row, cause-column; adapter transposes
- gCastle raw causal matrix: parent-row, child-column; adapter preserves orientation
- causal-learn: endpoint pair `(-1, 1)` is row to column; `(-1, -1)` is undirected
- Directed scoring: inter-group directed precision, recall, F1, and pairwise SHD
- PC scoring: primary skeleton precision, recall, F1, and symmetric-difference SHD
- PC orientation: secondary, eligibility gated separately at oracle orientation F1 0.80
- Known-answer adapter and node-index tests: PASS
