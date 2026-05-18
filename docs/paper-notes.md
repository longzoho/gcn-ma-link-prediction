 # Paper notes — Mei & Zhao 2024

DOI: 10.1038/s41598-023-50977-6

## Equations

### NRNAE
- Clustering coefficient: `CC(i) = 2·R(i) / (K(i)·(K(i)-1))`
- Aggregation strength: `AS(i) = degree(i) · CC(i)`
- Pairwise aggregation: `S(i,j) = |N(i) ∩ N(j)| · AS(i)`
- Enhanced adjacency: `Ŝ = A + β·S + I`, β ∈ [0.7, 0.9] reported optimal

### Spectral GCN
`H^t = σ(D̂^(-1/2) · Ŝ^t · D̂^(-1/2) · X^t · W^t)`

### LSTM weight evolution
`W^t = LSTMCell(W^{t-1}, state^{t-1})`

### Attention
`Z^t = MultiHeadSelfAttn(H^t)`

### Decoder
`P^t = σ(MLP([Z^t[u] ⊕ Z^t[v]]))`

### Loss
`L = -1/|B| · Σ [Y·log(P) + (1-Y)·log(1-P)]`

## Reported scores (Table 2)

| Dataset | AUC | AP |
|---|---|---|
| Mooc-action | 98.80 | 98.63 |
| CollegeMsg | 91.49 | 89.26 |
| EUT | 92.22 | 90.82 |
| Bitcoinotc | 91.20 | 89.43 |
| LastFM | 87.57 | 87.04 |
| Wikipedia | 87.42 | 85.75 |

## Things the paper omits

- Learning rate, optimizer, batch size, epochs.
- Embedding/hidden dim.
- Number of attention heads.
- Negative sampling strategy and ratio.
- Initialization of W^0.
- Code repository.
