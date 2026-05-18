# Slide deck outline — Bảo vệ luận văn

Ngôn ngữ: tiếng Việt. Thời lượng: ~20 phút. Mục tiêu: 22 slide chính + 6 phụ lục.
Trục story: temporal graph evolution → dataset-as-evolving-graph → 5 chiến lược bám thời gian → kết quả → diagnosis topology ↔ winner.

---

## Slide 1 — Tái hiện và phân tích so sánh GCN_MA cho dự đoán liên kết trên đồ thị động

- Đề tài luận văn tốt nghiệp
- Tác giả: [Tên]
- Giảng viên hướng dẫn: [Tên GVHD]
- Ngày bảo vệ: [Ngày]
- IMAGE: results/report/plots/dataset_snapshots_grid.png (3 snapshot làm hình nền)

## Slide 2 — Bài toán: dự đoán liên kết trên đồ thị động

- Cho chuỗi snapshot $G^1, G^2, \dots, G^t$, dự đoán cạnh nào sẽ xuất hiện ở $G^{t+1}$
- Khác static link prediction: phải bám đồng thời cấu trúc và thời gian
- Ứng dụng: gợi ý bạn, dự đoán giao dịch tài chính, phát hiện gian lận, gợi ý nội dung
- Câu hỏi cốt lõi: làm sao biểu diễn được sự tiến hóa của cấu trúc mạng?

## Slide 3 — Ba câu hỏi nghiên cứu

- Câu 1: Tái hiện GCN_MA (Mei & Zhao 2024) có khớp số trong paper không?
- Câu 2: So với baseline hiện đại (HTGN, DyGNN, EvolveGCN, DGCN), GCN_MA đứng ở đâu?
- **Câu 3 (trục chính):** Cấu trúc tiến hóa của mạng quyết định mô hình nào thắng như thế nào?

## Slide 4 — Sáu mạng, sáu câu chuyện tiến hóa

- CollegeMsg: 1,899 node, 59K cạnh, 47 snapshot, unipartite — tin nhắn sinh viên
- Bitcoinotc: 5,881 node, 36K cạnh, 62 snapshot, unipartite — đánh giá tin cậy
- EUT: 986 node, 332K cạnh, 127 snapshot, unipartite — email tổ chức
- Mooc-actions: 7,144 node, 412K cạnh, 72 snapshot, bipartite — học trực tuyến
- LastFM: 1,980 node, 1.29M cạnh, 41 snapshot, bipartite — nghe nhạc
- Wikipedia: 7,474 node, 110K cạnh, 42 snapshot, bipartite — chỉnh sửa wiki

## Slide 5 — Trực quan: mạng tiến hóa qua snapshot

- 3 dataset đại diện × 3 mốc thời gian (t=0, T/2, T)
- CollegeMsg sparse-unipartite: tăng trưởng tuần tự theo học kỳ
- EUT dense-unipartite: bão hòa nhanh, mọi người email lẫn nhau
- LastFM dense-bipartite: tăng trưởng siêu tuyến tính (1.3M cạnh!)
- IMAGE: results/report/plots/dataset_snapshots_grid.png

## Slide 6 — Động học của mạng theo thời gian

- (a) Số cạnh per snapshot — 6 đường, log scale
- (b) Mật độ $\rho^t = 2 E^t / N(N-1)$ — 6 đường, log scale
- LastFM tăng dốc, CollegeMsg bậc thang, EUT bão hòa sớm
- IMAGE: results/report/plots/edge_growth_density.png

## Slide 7 — Bản đồ 6 mạng theo 2 trục cấu trúc

- Trục X: mật độ trung bình $\bar{\rho}$ (log scale)
- Trục Y: degree-distribution Gini (proxy cho phân cấp)
- Marker: ◆ bipartite / ● unipartite
- Đây là khung phân tích → sẽ overlay winner ở Slide 18
- IMAGE: results/report/plots/topology_map_2d.png

## Slide 8 — GCN_MA (Mei & Zhao 2024)

- Pipeline: A^t → NRNAE → spectral GCN(W^t) → H^t → multi-head attention → Z^t → decoder
- NRNAE: $\hat{S} = A + \beta S + I$ với $S(i,j) = |N(i) \cap N(j)| \cdot \deg(i) \cdot CC(i)$
- Cơ chế bám thời gian: $W^t = \mathrm{LSTM}(W^{t-1})$ — tiến hóa qua **trọng số**, không qua trạng thái node
- Hyperparameter cố định: $\beta = 0.8$ (grid search trên Bitcoinotc validation)

## Slide 9 — EvolveGCN-O (Pareja et al. 2020)

- Pipeline: X^t → GCN(W^t) → H^t, với $W^t = \mathrm{GRU}(W^{t-1})$
- Cơ chế bám thời gian: GRU tiến hóa trọng số — phiên bản tối giản của ý tưởng "evolve-the-weights"
- 2 lớp GRCU; dùng `nn.Embedding` thay one-hot identity để tiết kiệm RAM
- Vị trí trong bức tranh: cùng họ GCN_MA, nhưng không có NRNAE và không có attention

## Slide 10 — HTGN (Yang et al. 2021) — đối thủ mạnh nhất

- Pipeline: X^t → HGCN trên Poincaré ball (c=1.0) → log map về Euclidean → Z^t
- Thêm Hyperbolic Temporal Attention qua snapshot
- Cơ chế bám thời gian: temporal attention trên không gian hyperbolic
- Lý do mạnh: đa tạp cong biểu diễn cây phân cấp với độ chính xác cao hơn Euclidean
- Forshadowing: top-1 hoặc top-2 ở 6/6 dataset

## Slide 11 — DyGNN (Ma et al. 2020) — bộ nhớ per-node

- Pipeline: mỗi cạnh đến → GRU update memory cho cả src và dst → propagate cho neighbors
- Cơ chế bám thời gian: per-node memory, event-driven thay vì snapshot-driven
- Vectorized variant (vendored) thay per-edge loop để fit GPU
- Lưu ý: N/A trên LastFM (OOM ở 1.3M cạnh) — sẽ hiển thị "—" trong bảng kết quả

## Slide 12 — DGCN (Manessi et al. 2020) — baseline đơn giản

- Pipeline: GCN per snapshot → ghép embedding qua thời gian → LSTM trên trục thời gian → decoder
- Cơ chế bám thời gian: LSTM trên chuỗi embedding (không phải trọng số)
- WD-GCN variant: chia sẻ tham số GCN giữa các snapshot
- Forshadowing: mạnh bất ngờ ở EUT, đứng top-1 (0.9847 AUC)
- Tóm tắt 5 chiến lược: GCN_MA = LSTM trên W; EvolveGCN = GRU trên W; HTGN = attention + hyperbolic; DyGNN = memory per-node; DGCN = LSTM trên embedding

## Slide 13 — Thiết lập thực nghiệm: minh bạch và tái lập

- Chia tập temporal: ~70% snapshot đầu train, ~15% val, ~15% test
- Negative sampling 1:1 (mỗi cạnh dương ghép 1 cạnh âm random)
- Hyperparameter: lấy paper khi có ($\beta$, $c$); grid-search khi paper không nêu (lr, hidden, heads, epochs)
- 3 seeds {42, 123, 2024}; Adam; early stopping theo val AUC
- Hardware: NVIDIA RTX 3060 12GB, CUDA 12.1, PyTorch 2.4.0
- Metric: AUC + AP, báo cáo mean ± std qua 3 seeds

## Slide 14 — Bảng tổng hợp 5 mô hình × 6 datasets

- GCN_MA tái hiện đúng paper ở 3/6 dataset trong ±1.5 điểm AUC, 4/6 trong ±2.5 điểm
- LastFM là outlier với gap ~7.5 điểm — sẽ thảo luận riêng ở Slide 19
- Top-1 phân bố: HTGN (collegemsg, bitcoinotc), DyGNN (mooc, wikipedia), DGCN (eut), EvolveGCN-O (lastfm)
- DyGNN×LastFM: N/A (OOM)

## Slide 15 — So sánh AUC & AP trực quan

- HTGN dẫn đầu hoặc top-2 ở 6/6 dataset — consistent strong baseline
- DyGNN bùng nổ trên mooc-actions + wikipedia (cả 2 bipartite + dense)
- GCN_MA bám sát paper nhưng tụt mạnh ở LastFM
- IMAGE: results/report/plots/auc_comparison.png
- IMAGE: results/report/plots/ap_comparison.png

## Slide 16 — Ranking heatmap: ai thắng ở đâu

- Trục dọc: 5 mô hình; trục ngang: 6 dataset; ô đậm = rank cao
- 3 điểm chú ý: HTGN top trên CollegeMsg; DyGNN top trên Wiki/Mooc; DGCN top trên EUT (hơn HTGN ~0.001)
- DyGNN×LastFM masked grey (em-dash) — OOM, không so sánh được
- IMAGE: results/report/plots/ranking_heatmap.png

## Slide 17 — Learning curves chọn lọc

- CollegeMsg (sparse, unipartite): HTGN hội tụ nhanh, GCN_MA ổn định nhưng plateau thấp hơn
- Mooc-actions (dense, bipartite): DyGNN hội tụ nhanh nhất, GCN_MA bám sát top
- 4 dataset còn lại đặt ở Appendix A2 (sẵn sàng nếu hội đồng hỏi)
- IMAGE: results/report/plots/learning_curves_collegemsg.png
- IMAGE: results/report/plots/learning_curves_mooc_actions.png

## Slide 18 — Diagnosis: cấu trúc tiến hóa ↔ mô hình thắng

- Mệnh đề 1: Cấu trúc phân cấp ẩn → hyperbolic ăn điểm (HTGN top-1 ở CollegeMsg, Bitcoinotc)
- Mệnh đề 2: Sự kiện dồn dập trên bipartite dày → bộ nhớ per-node ăn điểm (DyGNN top-1 ở Mooc-actions, Wikipedia)
- Mệnh đề 3: Mạng dày không cấu trúc đặc thù → baseline đơn giản đủ tốt (DGCN top-1 ở EUT)
- Cấu trúc tiến hóa là tín hiệu thiết kế — không có mô hình một-cỡ-vừa-cho-tất-cả
- IMAGE: results/report/plots/topology_map_2d_with_winners.png

## Slide 19 — Nơi GCN_MA tỏa sáng và tụt hậu

- Tỏa sáng: Mooc-actions gap 0.4, Wikipedia 0.5, CollegeMsg 1.4 — 3/6 trong ±1.5 điểm AUC
- EUT (gap 2.1) và Bitcoinotc (gap 5.6) lệch xa hơn nhưng vẫn hợp lý qua 3 seeds
- Tụt hậu: LastFM gap ~7.5 — giả thuyết NRNAE bão hòa khi mạng siêu dense (1.3M cạnh)
- Thua HTGN/DyGNN ở 4/6 dataset (top-1)
- β = 0.8 là validation choice, không cherry-picked
- IMAGE: results/report/plots/beta_sensitivity.png

## Slide 20 — Runtime & trade-off: bức tranh đầy đủ

- GCN_MA thuộc nhóm nhẹ (LSTM trên W, NRNAE tính 1 lần)
- HTGN chậm 3–5× vì hyperbolic ops (exp_map, log_map)
- DyGNN chậm nhất do per-edge update tuần tự
- Bảo vệ paper gốc: GCN_MA không SOTA accuracy, nhưng là sweet spot accuracy/cost
- IMAGE: results/report/plots/runtime_comparison.png

## Slide 21 — Đóng góp & bài học

- Đóng góp 1: Tái hiện thành công GCN_MA — 3/6 dataset trong ±1.5 điểm AUC, code mở
- Đóng góp 2: Mở rộng so sánh với 4 baseline hiện đại trên cùng pipeline → fair-comparison mà paper gốc thiếu
- Đóng góp 3: Khung diagnosis cấu trúc-tiến-hóa ↔ mô hình để chọn baseline phù hợp
- Bài học: cấu trúc đồ thị là tín hiệu chọn mô hình; GCN_MA là sweet spot; paper gốc thiếu reproducibility
- Hướng mở rộng: NRNAE + LSTM-W kết hợp với hyperbolic hoặc per-node memory

## Slide 22 — Xin cảm ơn — Q&A

- Tên + email + tên GVHD
- QR code → GitHub repo (code + reproduction-log + thesis chapter PDF)
- Sẵn sàng nhận câu hỏi
- IMAGE: results/report/plots/dataset_snapshots_grid.png (hình nền lặp lại để đóng khung visual)

---

## Appendix (6 slide phụ — không trình bày mặc định, dùng cho Q&A)

## Slide A1 — Hyperparameter chi tiết 5 mô hình

- GCN_MA: hidden=128, β=0.8, heads=4, lr=1e-3, epochs=100, early-stop patience=20
- EvolveGCN-O: 2 lớp GRCU, hidden=128, lr=1e-3
- HTGN: c=1.0, hidden=128, lr=1e-3, patience=20
- DyGNN: memory_dim=128, lr=1e-3, vectorized variant
- DGCN: 2 lớp GCN per snapshot, LSTM hidden=128, WD-GCN variant
- Toàn bộ chi tiết trong `results/configs_runtime/`

## Slide A2 — Learning curves còn lại

- IMAGE: results/report/plots/learning_curves_bitcoinotc.png
- IMAGE: results/report/plots/learning_curves_eut.png
- IMAGE: results/report/plots/learning_curves_lastfm.png
- IMAGE: results/report/plots/learning_curves_wikipedia.png

## Slide A3 — Deviation chi tiết so với paper gốc

- collegemsg: gap −1.44 điểm AUC (within ±1.5)
- bitcoinotc: gap −5.60
- eut: gap −2.14
- mooc_actions: gap −0.35 (within ±1.5)
- lastfm: gap −7.53 (outlier, NRNAE saturation hypothesis)
- wikipedia: gap −0.46 (within ±1.5)

## Slide A4 — Chi tiết NRNAE với ví dụ tính toán

- Ví dụ: graph 4 node, cạnh {(1,2), (2,3), (3,1), (3,4)}
- $\deg = [2, 2, 3, 1]$
- $CC(1) = 1.0$ (hai hàng xóm 2,3 nối nhau), $CC(2)=1.0$, $CC(3)=0.33$, $CC(4)=0$
- $AS = [2, 2, 1, 0]$
- $S(1,2) = |\{3\}| \cdot AS(1) = 2$; tương tự cho các cặp khác
- $\hat{S} = A + 0.8 \cdot S + I$

## Slide A5 — Notes triển khai khó

- DyGNN OOM trên LastFM (1.3M cạnh) → bỏ dataset này khỏi run
- HTGN: upstream gọi argparse khi import → workaround reset sys.argv
- HTGN: hidden_initial không phải nn.Parameter → override .to() để rebuild trên target device
- EvolveGCN: PyTorch 2.4 patch nâng GRCU_layers thành nn.ModuleList; restore _parameters = {}
- Adjacency symmetrize để bipartite không bị triệt tiêu về zero embedding

## Slide A6 — Link tài nguyên

- GitHub repo: [URL]
- Reproduction log: docs/reproduction-log.md
- Thesis chapter PDF: docs/thesis_chapter.pdf
- Strengths/weaknesses analysis: docs/gcn-ma-strengths-weaknesses.pdf
