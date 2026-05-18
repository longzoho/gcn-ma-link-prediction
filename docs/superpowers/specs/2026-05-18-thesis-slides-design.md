# Thiết kế slide bảo vệ luận văn — GCN_MA reproduction + cross-baseline comparison

**Ngày:** 2026-05-18
**Tác giả:** long.huynh
**Mục đích:** Spec đề cương slide bảo vệ luận văn thesis, để chuyển tiếp sang giai đoạn writing-plans (lập kế hoạch thực thi).

---

## 1. Bối cảnh & mục tiêu

**Bối cảnh trình bày**
- Thesis defense (bảo vệ luận văn) trước hội đồng giảng viên
- Thời lượng: ~20 phút (Q&A riêng)
- Ngôn ngữ: **Tiếng Việt** (đồng bộ với `docs/thesis_chapter.md` đã viết)
- Platform: **Gamma AI** (sinh tự động từ outline, xuất PPTX/PDF, có thể chỉnh sau)

**Trục câu chuyện được chọn (Hướng B — Dataset-as-evolving-graph)**

Trục chính của talk: *"Cấu trúc tiến hóa của mạng (temporal graph evolution) quyết định mô hình nào thắng như thế nào?"*

6 dataset là **nhân vật chính**; 5 mô hình là **5 cách bám theo thời gian** khác nhau; kết quả + diagnosis là **lập luận liên kết hai bên**.

**Mục tiêu deck**
1. Chứng minh GCN_MA (Mei & Zhao 2024) đã được tái hiện đúng — 3/6 dataset (mooc, wikipedia, collegemsg) trong ±1.5 điểm AUC, 4/6 trong ±2.5 điểm; LastFM là outlier với gap ~7.5 điểm cần thảo luận
2. Định vị GCN_MA giữa 4 baseline hiện đại (EvolveGCN-O, HTGN, DyGNN, DGCN) trên cùng pipeline
3. **Đóng góp lập luận chính:** khung diagnosis "cấu trúc-tiến-hóa ↔ mô hình thắng" — không có mô hình một-cỡ-vừa-cho-tất-cả

---

## 2. Cấu trúc tổng thể (22 slides chính + 6 Appendix)

| Phần | # | Tên cụm | Slides | Ước lượng thời gian |
|---|---|---|---|---|
| I | 1–3 | Mở đầu: bối cảnh & câu hỏi nghiên cứu | 3 | ~2 phút |
| II | 4–7 | 6 mạng tiến hóa — các "nhân vật" | 4 | ~3.5 phút |
| III | 8–12 | 5 mô hình — 5 cách bám theo thời gian | 5 | ~4 phút |
| IV | 13 | Thiết lập thực nghiệm | 1 | ~1 phút |
| V | 14–17 | Kết quả định lượng | 4 | ~3.5 phút |
| VI | 18–20 | Phân tích: topology ↔ winner | 3 | ~4 phút |
| VII | 21–22 | Kết bài & Q&A | 2 | ~2 phút |
| | A1–A6 | Appendix (dự phòng Q&A) | 6 | ad-hoc |
| | **Tổng** | | **22 + 6** | **~20 phút** |

---

## 3. Đề cương chi tiết từng slide

### Phần I — Mở đầu (3 slides)

**Slide 1 — Tiêu đề**
- Tên đề tài: "Tái hiện và phân tích so sánh GCN_MA cho dự đoán liên kết trên đồ thị động"
- Tác giả, GVHD, ngày bảo vệ
- Hình nền: 3 snapshot nhỏ của một đồ thị tiến hóa (t=0, t=T/2, t=T)

**Slide 2 — Bài toán dynamic link prediction**
- Định nghĩa: cho chuỗi snapshot $G^1, G^2, \dots, G^t$ → dự đoán cạnh ở $G^{t+1}$
- Sơ đồ trực quan: 3 ô đồ thị liên tiếp + mũi tên "?"
- Khác static link prediction: cấu trúc + thời gian
- Ứng dụng: gợi ý bạn, dự đoán giao dịch, phát hiện gian lận

**Slide 3 — Câu hỏi nghiên cứu**
- 3 câu hỏi:
  1. Tái hiện GCN_MA có khớp số trong paper không?
  2. So với baseline hiện đại (HTGN, DyGNN, EvolveGCN, DGCN), GCN_MA đứng ở đâu?
  3. **Cấu trúc tiến hóa của mạng** quyết định mô hình nào thắng như thế nào?
- Câu 3 in đậm → trục chính của talk

### Phần II — 6 mạng tiến hóa: các "nhân vật" (4 slides)

**Slide 4 — Sáu mạng, sáu câu chuyện tiến hóa**
- Bảng tóm tắt (nguồn: `results/report/dataset_stats.md`):

| Dataset | N | E | T | Bipartite | Lĩnh vực |
|---|---|---|---|---|---|
| collegemsg | 1,899 | 59,835 | 47 | False | Tin nhắn sinh viên |
| bitcoinotc | 5,881 | 35,592 | 62 | False | Đánh giá tin cậy |
| eut | 986 | 332,334 | 127 | False | Email tổ chức |
| mooc_actions | 7,144 | 411,749 | 72 | True | Học trực tuyến |
| lastfm | 1,980 | 1,293,103 | 41 | True | Nghe nhạc |
| wikipedia | 7,474 | 110,218 | 42 | True | User–page edit |

- Mỗi dòng kèm icon nhỏ minh họa lĩnh vực

**Slide 5 — Trực quan: mạng tiến hóa qua snapshot** *(plot mới cần render)*
- Small multiples 3×3: 3 dataset đại diện × 3 mốc thời gian (t=0, T/2, T)
  - CollegeMsg (sparse–unipartite)
  - EUT (dense–unipartite)
  - LastFM (dense–bipartite)
- Mỗi ô: node-link diagram nhỏ (NetworkX `spring_layout` + matplotlib)
- Caption: "Mỗi dataset là một quỹ đạo cấu trúc khác nhau"

**Slide 6 — Động học của mạng theo thời gian** *(plot mới cần render)*
- 2 line chart cạnh nhau:
  - (a) Số cạnh per snapshot — 6 đường
  - (b) Mật độ $\rho = 2E/(N(N-1))$ theo snapshot — 6 đường
- Highlight: LastFM tăng dốc, CollegeMsg bậc thang, EUT bão hòa sớm

**Slide 7 — Bản đồ 6 mạng theo 2 trục cấu trúc** *(plot mới cần render)*
- Scatter 2D: trục X = mật độ trung bình $\rho$, trục Y = chỉ số tree-likeness (proxy cho phân cấp — gợi ý: $\delta$-hyperbolicity Gromov, hoặc nếu quá đắt tính trên N lớn thì dùng "average shortest-path / log N" hoặc degree-distribution Gini coefficient)
- Marker: ◆ bipartite, ● unipartite
- Đặt làm **khung phân tích** → tái sử dụng ở Slide 18
- Câu chốt: "Cấu trúc tiến hóa đa dạng → đòi hỏi cơ chế bám thời gian khác nhau"
- **Lưu ý implementation:** chọn metric Y cụ thể sẽ được xác định trong giai đoạn render plot — yêu cầu: rẻ tính trên N≤8K, phân biệt rõ 6 datasets, có ý nghĩa với hyperbolic geometry (HTGN). Avg clustering coefficient KHÔNG dùng được vì CC cao = nhiều tam giác = đối lập với hierarchy

### Phần III — 5 mô hình: 5 cách bám theo thời gian (5 slides)

Template chung: sơ đồ luồng dữ liệu trên + 1 dòng "cơ chế bám thời gian" + 2 dòng "điểm mạnh / hạn chế".

**Slide 8 — GCN_MA (Mei & Zhao 2024)**
- Sơ đồ: `A^t → NRNAE (Ŝ = A + βS + I) → GCN(W^t) → H^t → Multi-head Attn → Z^t → decoder`
- Cơ chế thời gian: **LSTMCell tiến hóa trọng số** $W^t = \mathrm{LSTM}(W^{t-1})$
- Điểm độc đáo: NRNAE — tăng cường ma trận kề bằng đặc trưng cấu trúc cục bộ (CC, AS)
- β = 0.8 (grid search trên Bitcoinotc validation)

**Slide 9 — EvolveGCN-O (Pareja et al. 2020)**
- Sơ đồ: `X^t → GCN(W^t) → H^t`, $W^t = \mathrm{GRU}(W^{t-1})$
- Cơ chế thời gian: **GRU tiến hóa trọng số** — phiên bản tối giản của ý tưởng evolve-the-weights
- 2 lớp GRCU, `nn.Embedding` thay one-hot identity

**Slide 10 — HTGN (Yang et al. 2021) — đối thủ mạnh nhất**
- Sơ đồ: `X^t → HGCN (Poincaré ball, c=1.0) → log_map_origin → Z^t`, + Hyperbolic Temporal Attention
- Cơ chế thời gian: **temporal attention trên không gian hyperbolic**
- Highlight visual: hình quả cầu Poincaré
- Forshadowing: HTGN sẽ thắng nhiều dataset

**Slide 11 — DyGNN (Ma et al. 2020) — bộ nhớ per-node**
- Sơ đồ: mỗi cạnh đến → GRU update cho src + dst → propagation cho neighbors
- Cơ chế thời gian: **per-node memory, event-driven** thay vì snapshot-driven
- Vectorized variant (vendored) thay per-edge loop
- **Cần nhấn:** N/A trên LastFM (OOM ở 1.3M cạnh)

**Slide 12 — DGCN (Manessi et al. 2020) — baseline đơn giản**
- Sơ đồ: stack GCN per snapshot → ghép embedding qua thời gian → LSTM trên trục thời gian
- Cơ chế thời gian: **LSTM trên chuỗi embedding** (không phải trọng số)
- WD-GCN variant: chia sẻ tham số GCN giữa snapshot
- Forshadowing: mạnh bất ngờ ở EUT

Mini-table tóm tắt 5 chiến lược ở chân slide 12:

| Mô hình | Bám thời gian qua... |
|---|---|
| GCN_MA | Trọng số (LSTM) + NRNAE cấu trúc cục bộ |
| EvolveGCN-O | Trọng số (GRU) |
| HTGN | Temporal attention + hyperbolic embedding |
| DyGNN | Bộ nhớ per-node (GRU theo cạnh) |
| DGCN | Embedding stack + LSTM trục thời gian |

### Phần IV — Thiết lập thực nghiệm (1 slide)

**Slide 13 — Thiết lập thực nghiệm: minh bạch và tái lập**

Layout 2 cột:
- **Trái — Dữ liệu & chia tập**: 6 datasets; chia temporal ~70/15/15 theo trục thời gian; negative sampling 1:1
- **Phải — Hyperparameter (Hybrid policy)**: lấy paper khi có (β, c); grid-search khi paper không nêu (lr, hidden, heads, epochs); Adam; 3 seeds; early stopping theo val AUC; NVIDIA GPU + CUDA 12.1 + PyTorch 2.4
- Chân slide: metrics = AUC + AP (mean ± std qua 3 seeds); raw metrics có ở `results/metrics.jsonl`

### Phần V — Kết quả định lượng (4 slides)

Plots đã có sẵn ở `results/report/plots/`.

**Slide 14 — Bảng tổng hợp 5 mô hình × 6 datasets**
- Bảng AUC + AP từ `results/report/baselines_summary.md`, format mean ± std
- Highlight màu: xanh đậm = top-1; xanh nhạt = top-2; xám gạch ngang = DyGNN×LastFM (OOM)
- Cột "Paper GCN_MA" đặt cuối → đối chiếu trực quan
- Câu chốt: **"GCN_MA tái hiện đúng paper (3/6 dataset trong ±1.5 điểm AUC, 4/6 trong ±2.5 điểm), nhưng không phải mô hình mạnh nhất khi so với baseline hiện đại"**

**Slide 15 — So sánh AUC & AP trực quan**
- Trái: `results/report/plots/auc_comparison.png`
- Phải: `results/report/plots/ap_comparison.png`
- Đường kẻ ngang trên mỗi cụm = AUC paper báo cáo cho GCN_MA → tham chiếu
- Phát hiện đáng chú ý:
  - HTGN top-1 hoặc top-2 ở 6/6 dataset — consistent strong baseline
  - DyGNN bùng nổ trên mooc-actions + wikipedia (bipartite + dense)
  - GCN_MA bám sát paper nhưng tụt mạnh ở LastFM (0.80 vs paper 0.87)

**Slide 16 — Ranking heatmap: ai thắng ở đâu**
- Plot: `results/report/plots/ranking_heatmap.png` (đã render, ô DyGNN×LastFM đã masked grey với em-dash)
- Trục: 5 mô hình × 6 dataset; ô tô màu theo rank
- Annotation đè: 3 mũi tên trỏ vào 3 ô nổi bật (HTGN top trên CollegeMsg, DyGNN top trên Wiki/Mooc, DGCN top trên EUT)
- Slide bản lề giữa kết quả thô và diễn giải

**Slide 17 — Learning curves chọn lọc**
- 2 dataset đại diện:
  - `results/report/plots/learning_curves_collegemsg.png` (sparse, unipartite)
  - `results/report/plots/learning_curves_mooc_actions.png` (dense, bipartite)
- 4 dataset còn lại để Appendix A2
- Đọc: tốc độ hội tụ, hiện tượng overfit (early stopping kicks in)

### Phần VI — Phân tích: topology ↔ winner (3 slides)

Phần đắt giá nhất — kích hoạt trục "biểu đồ mạng" đã setup ở Phần II.

**Slide 18 — Diagnosis: cấu trúc tiến hóa ↔ mô hình thắng**
- Layout: tái sử dụng bản đồ topology 2D ở Slide 7, overlay tên mô hình thắng tại vị trí từng dataset
- 3 mệnh đề diagnosis (1 dòng/mệnh đề):
  1. **Cấu trúc phân cấp ẩn → hyperbolic ăn điểm** (HTGN top-1 ở CollegeMsg, Bitcoinotc; top-2 ở Wikipedia, LastFM, EUT — consistent strong baseline)
  2. **Sự kiện dồn dập trên bipartite dày → bộ nhớ per-node ăn điểm** (DyGNN top-1 ở Mooc-actions, Wikipedia)
  3. **Mạng dày không có cấu trúc đặc thù → baseline đơn giản đủ tốt** (DGCN top-1 ở EUT, hơn HTGN ~0.001 AUC — chứng minh khi mọi node ai-cũng-kết-nối-ai, stack-GCN-rồi-LSTM là vừa đủ)
- Câu chốt: "Cấu trúc tiến hóa là tín hiệu thiết kế — không có mô hình một-cỡ-vừa-cho-tất-cả"

**Slide 19 — Nơi GCN_MA tỏa sáng và tụt hậu**
- **Tỏa sáng**:
  - Tái hiện rất sát ở Mooc-actions (gap 0.4 điểm), Wikipedia (0.5), CollegeMsg (1.4) — 3/6 trong ±1.5 điểm AUC
  - EUT (gap 2.1) và Bitcoinotc (gap 5.6) lệch xa hơn nhưng vẫn nằm trong vùng hợp lý qua 3 seeds
- **Tụt hậu**:
  - LastFM gap lớn nhất (0.80 vs paper 0.87, gap ~7.5 điểm) — giả thuyết: 1.3M cạnh siêu dense → NRNAE bão hòa, $\beta$-weighting hết phân biệt
  - Thua HTGN/DyGNN ở 4/6 dataset (top-1)
- Embed nhỏ `results/report/plots/beta_sensitivity.png` (góc dưới-phải) → β=0.8 là validation choice
- **Bài học**: NRNAE + LSTM-weight-evolve là idea hay, nhưng chưa khai thác phân cấp như HTGN, chưa bắt sự kiện mịn như DyGNN

**Slide 20 — Runtime & trade-off: bức tranh đầy đủ**
- Plot: `results/report/plots/runtime_comparison.png` (training time per epoch)
- 2 quan sát:
  - GCN_MA nhóm nhẹ (LSTM trên W chứ không trên embedding, NRNAE tính 1 lần)
  - HTGN chậm 3–5× (hyperbolic ops); DyGNN chậm nhất (per-edge update)
- Câu chốt **bảo vệ paper gốc**: "GCN_MA không phải SOTA accuracy, nhưng là sweet spot accuracy/cost"

### Phần VII — Kết bài & Q&A (2 slides)

**Slide 21 — Đóng góp & bài học**

Layout 2 khối:

**Đóng góp**
- Tái hiện thành công GCN_MA ở 3/6 dataset trong ±1.5 điểm AUC, 4/6 trong ±2.5 điểm AUC so với paper — đầy đủ reproduction log, code mở
- Mở rộng so sánh với 4 baseline hiện đại trên cùng pipeline → bức tranh fair-comparison mà paper gốc thiếu
- Cung cấp **khung diagnosis cấu trúc-tiến-hóa ↔ mô hình** để chọn baseline phù hợp

**Bài học**
- Cấu trúc đồ thị là **tín hiệu chọn mô hình** — không nên benchmark mù
- GCN_MA là sweet spot accuracy/cost, không phải SOTA
- Paper gốc thiếu reproducibility (không công bố code, thiếu hyperparameter) → cộng đồng cần chuẩn cao hơn

**Hướng mở rộng** (chân slide):
> "NRNAE + LSTM-weight-evolve có thể kết hợp với hyperbolic embedding hoặc per-node memory để khai thác đồng thời cả phân cấp lẫn sự kiện mịn"

**Slide 22 — Cảm ơn & Q&A**
- "Xin cảm ơn — Q&A"
- Tên + email + tên GVHD
- QR code → GitHub repo (code + reproduction-log + thesis chapter PDF)
- Hình nền: lặp lại 3 snapshot đồ thị tiến hóa của slide tiêu đề (đóng khung visual)

### Appendix (6 slides dự phòng Q&A)

- **A1** — Bảng hyperparameter chi tiết cho cả 5 mô hình
- **A2** — 4 learning curve còn lại (bitcoinotc, eut, lastfm, wikipedia) từ `results/report/plots/`
- **A3** — Bảng deviation chi tiết: số chúng tôi vs số paper với % chênh
- **A4** — Sơ đồ chi tiết NRNAE (CC, AS, S, Ŝ với ví dụ tính toán nhỏ)
- **A5** — Notes triển khai khó: DyGNN OOM trên LastFM, HTGN `sys.argv` hack, EvolveGCN PyTorch 2.4 patch
- **A6** — Link reproduction-log + thesis chapter PDF

---

## 4. Plot artifact map (ai có sẵn, ai cần tạo)

**Có sẵn** (từ `results/report/plots/`):
- `auc_comparison.png` → Slide 15
- `ap_comparison.png` → Slide 15
- `ranking_heatmap.png` → Slide 16
- `learning_curves_collegemsg.png` → Slide 17
- `learning_curves_mooc_actions.png` → Slide 17
- `learning_curves_bitcoinotc.png`, `_eut.png`, `_lastfm.png`, `_wikipedia.png` → Appendix A2
- `beta_sensitivity.png` → Slide 19 (embed nhỏ)
- `runtime_comparison.png` → Slide 20

**Cần tạo mới** (3 plot):

| Plot | Mục đích | Slide | Script gợi ý |
|---|---|---|---|
| `dataset_snapshots_grid.png` | 3 dataset × 3 mốc thời gian, node-link diagram | Slide 5 | NetworkX `spring_layout` + matplotlib subplots |
| `edge_growth_density.png` | Edge count + density per snapshot, 6 đường | Slide 6 | matplotlib, đọc từ snapshots đã preprocess |
| `topology_map_2d.png` | Scatter mật độ × clustering, marker bipartite/unipartite, overlay winner ở Slide 18 | Slide 7 + 18 | matplotlib scatter, 2 phiên bản (Slide 7 không overlay, Slide 18 có overlay) |

Script cần thêm: `scripts/plot_dataset_topology.py` (gói cả 3 plot mới ở trên).

---

## 5. Pipeline tạo slide

**Bước 1 — Render 3 plot mới**
- Viết `scripts/plot_dataset_topology.py`
- Output vào `results/report/plots/` cùng các plot có sẵn
- Verify visual: mở từng plot, kiểm tra label/legend đọc được ở size slide

**Bước 2 — Soạn outline Gamma-friendly**
- Tạo file `docs/slides/thesis_defense_outline.md` chứa outline cô đọng (slide title + 3–5 bullets + image reference) — đây là input cho Gamma
- Format đặc trưng: mỗi slide dùng `## Slide N — Title` và bullets ngắn (Gamma cắt theo heading)

**Bước 3 — Sinh deck qua Gamma MCP**
- Gọi `mcp__claude_ai_Gamma__generate` với input là outline ở Bước 2
- Tham số quan trọng: số slide ~22, theme phù hợp academic (clean, ít màu), aspect 16:9
- Lưu lại Gamma file ID để chỉnh sửa sau

**Bước 4 — Chỉnh tay trong Gamma editor (do user thực hiện)**
- Upload plot images vào các slide cần (Gamma không tự pull từ filesystem)
- Tinh chỉnh layout, màu, font
- Xuất PDF cuối cùng

**Bước 5 — In bản dự phòng**
- Xuất PDF từ Gamma
- In 2 bản (1 cho presenter, 1 dự phòng technical issue)

---

## 6. Rủi ro & giảm thiểu

| Rủi ro | Tác động | Giảm thiểu |
|---|---|---|
| Gamma sinh slide không bám đúng outline (cắt nhầm heading, gộp bullets) | Phải chỉnh tay nhiều | Outline ở Bước 2 viết theo format Gamma-friendly đã được test; nếu vẫn lệch, có thể chia thành 3 lần generate (Phần I–III, IV–V, VI–VII) |
| 3 plot mới không kịp render | Slide 5–7 + 18 mất visual chính | Plot fallback: dùng bảng + caption text mô tả; ưu tiên render `topology_map_2d.png` trước vì dùng ở cả Slide 7 và 18 |
| Hội đồng hỏi sâu về 1 dataset cụ thể không có trong Slide 17 | Bị động | Appendix A2 đã chuẩn bị sẵn 4 learning curve còn lại |
| Hội đồng chất vấn deviation LastFM | Trục diễn giải bị lung lay | Slide 19 đã đặt thẳng giả thuyết; Appendix A3 có bảng deviation chi tiết |
| Thời lượng vượt 20 phút | Q&A bị cắt | Tập dượt với timer trước; có thể bỏ slide 17 (learning curves) hoặc slide 6 (edge growth) làm buffer |

---

## 7. Tiêu chí thành công

- [ ] 22 slide chính + 6 appendix sinh được trong Gamma, tổng <20 phút khi nói
- [ ] 3 plot mới (`dataset_snapshots_grid`, `edge_growth_density`, `topology_map_2d`) render đẹp, đọc được ở 16:9
- [ ] Mỗi slide có thông điệp duy nhất (one-message-per-slide)
- [ ] Trục "biểu đồ mạng tiến hóa" xuất hiện ở Slide 1 (visual), 2 (định nghĩa), 5 (snapshots), 7 (bản đồ), 18 (diagnosis), 22 (visual đóng) — tổng 6 slide có yếu tố này
- [ ] Diagnosis 3 mệnh đề ở Slide 18 đối chiếu được trực tiếp với 3 winner ở Slide 16 (heatmap)

---

## 8. Liên kết tới các spec trước

- `2026-05-16-gcn-ma-link-prediction-design.md` — spec gốc của GCN_MA reproduction
- `2026-05-16-evolvegcn-o-integration-design.md`, `2026-05-17-htgn-integration-design.md`, `2026-05-17-dygnn-integration-design.md`, `2026-05-18-dgcn-integration-design.md` — spec tích hợp 4 baseline
- `2026-05-18-final-aggregation-thesis-design.md` — spec viết chapter thesis
- **Spec này** là spec cuối của chuỗi: từ chapter PDF → slide bảo vệ
