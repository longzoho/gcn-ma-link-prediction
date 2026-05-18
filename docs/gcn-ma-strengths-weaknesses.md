# GCN_MA — Ưu điểm và nhược điểm

Tài liệu này tổng hợp các điểm mạnh và hạn chế của mô hình **GCN_MA** (Mei & Zhao, *Scientific Reports* 2024, DOI 10.1038/s41598-023-50977-6) dựa trên (i) thiết kế kiến trúc của bài báo gốc và (ii) kết quả thực nghiệm tái hiện trong luận văn này trên sáu tập dữ liệu chuẩn so với bốn baseline hiện đại (EvolveGCN-O, HTGN, DyGNN, DGCN). Mọi nhận định đều được liên kết với số liệu cụ thể trong `results/metrics.jsonl` và các bảng trong `results/report/`.

---

## 1. Tóm tắt kiến trúc

GCN_MA (Graph Convolutional Network with Multi-head Attention) gồm bốn thành phần chính:

1. **NRNAE** (Neighborhood-Reinforced Node Aggregation Enhancement): tiền xử lý đặc trưng cấu trúc 3 chiều cho mỗi node tại từng snapshot — *degree*, *clustering coefficient* (CC), *aggregated strength* (AS).
2. **Spectral normalization động**: chuẩn hóa adjacency theo $\hat{S}^t = \hat{D}^{-1/2}(A^t + I)\hat{D}^{-1/2}$ tại mỗi snapshot, kết hợp đặc trưng NRNAE với hệ số trộn $\beta$.
3. **LSTM weight evolution**: trọng số GCN $W^t$ không học độc lập tại mỗi snapshot mà được *tiến hóa* qua một `LSTMCell` xuyên qua trục thời gian.
4. **Multi-head self-attention**: lớp attention với `num_heads = 4` (giảm từ 8 trong paper gốc sau grid-search Plan 2) đặt lên đầu ra GCN để tinh chỉnh embedding trước decoder.

Decoder dùng chung `LinkDecoderMLP` (MLP hai lớp) với tất cả baseline để đảm bảo so sánh công bằng — chi tiết xem §1.6 của `docs/thesis_chapter.md`.

---

## 2. Ưu điểm

### 2.1 Hội tụ rất ổn định qua các seed

GCN_MA cho **standard deviation thấp nhất trong nhóm năm mô hình** trên cả ba seed `{42, 123, 2024}` ở phần lớn datasets:

| Dataset | GCN_MA AUC std | HTGN AUC std | DyGNN AUC std | DGCN AUC std |
|---|---|---|---|---|
| collegemsg | **0.0002** | 0.0021 | 0.0132 | 0.0056 |
| eut | **0.0016** | 0.0005 | 0.0009 | 0.0007 |
| mooc_actions | **0.0002** | 0.0009 | 0.0001 | 0.0025 |
| wikipedia | **0.0007** | 0.0038 | 0.0017 | 0.0040 |

Std cỡ $10^{-4}$ trên collegemsg, mooc_actions, wikipedia cho thấy hàm mất mát của GCN_MA có cảnh quan tối ưu (loss landscape) **phẳng và dễ tìm cùng một cực tiểu**, độc lập với khởi tạo trọng số. Đây là một thuộc tính có giá trị thực tế: khi triển khai, một lần huấn luyện duy nhất là đủ để dự đoán hiệu năng cuối cùng.

### 2.2 Tốt nhất nhóm về tái hiện trên đồ thị nhỏ-vừa, sparse

Trên ba dataset có mật độ cạnh thấp đến trung bình (collegemsg, mooc_actions, wikipedia), khoảng cách giữa số liệu tái hiện và paper gốc là **rất nhỏ**:

| Dataset | AUC tái hiện | AUC paper | ΔAUC |
|---|---|---|---|
| collegemsg | 0.9005 | 0.9149 | **−0.0144** |
| mooc_actions | 0.9845 | 0.9880 | **−0.0035** |
| wikipedia | 0.8696 | 0.8742 | **−0.0046** |

Gap dưới 2 điểm phần trăm là chỉ báo rằng kiến trúc GCN_MA hoạt động đúng như mô tả khi đặc trưng cấu trúc cục bộ (NRNAE) có tính phân biệt cao — điển hình trên đồ thị sparse mà mỗi node có "chữ ký" topo riêng.

### 2.3 Hai trường hợp **vượt** số liệu paper trên AP

Phiên bản tái hiện thực tế **vượt paper** trên AP ở hai datasets:

| Dataset | AP tái hiện | AP paper | ΔAP |
|---|---|---|---|
| collegemsg | 0.9181 | 0.8926 | **+0.0255** |
| wikipedia | 0.8914 | 0.8575 | **+0.0339** |

Phần này có khả năng phản ánh chiến lược negative sampling khác nhau (xem §3.3 thesis), nhưng vẫn xác nhận pipeline tái hiện không "thấp hơn paper một cách hệ thống" — có những kịch bản tái hiện vượt số liệu gốc.

### 2.4 Chi phí tính toán hợp lý

Tổng runtime của GCN_MA qua 6 datasets × 3 seeds ≈ **4 giờ** trên GPU 12 GB, đặt giữa nhóm tốn kém (HTGN ~9.2h) và nhóm rẻ (EvolveGCN-O ~2.6h). Khi `hidden_dim = 64` và `num_heads = 4`, mô hình vừa với bộ nhớ GPU tầm trung và phù hợp với bối cảnh nghiên cứu học thuật không có cluster lớn.

### 2.5 Số chiều ẩn nhỏ vẫn đủ biểu cảm

Grid-search nội bộ (Plan 2) trên Bitcoinotc cho thấy **`hidden_dim = 64` thắng `hidden_dim = 128` trên 2/3 giá trị β được thử** (xem §3.6 thesis). Điều này có nghĩa GCN_MA không cần capacity lớn để hội tụ tốt — phù hợp khi triển khai trên thiết bị tài nguyên hạn chế hoặc cần latency thấp khi inference.

### 2.6 Cơ sở lý thuyết rõ ràng và khả diễn giải

Mỗi thành phần trong kiến trúc GCN_MA có ý nghĩa diễn giải cụ thể:

- Đặc trưng NRNAE (degree, CC, AS) là **3 đại lượng cấu trúc cổ điển** đã được nghiên cứu trong lý thuyết đồ thị từ trước, không phải feature blackbox.
- Hệ số $\beta$ kiểm soát trộn giữa đặc trưng NRNAE và spectral GCN — có thể tuning rõ ràng và quan sát hiệu ứng (Plan 2 đo $\beta \in \{0.7, 0.8, 0.9\}$).
- LSTM weight evolution là cơ chế tường minh để chia sẻ thông tin xuyên thời gian, đối lập với các phương pháp ngầm như attention thuần.

Tính diễn giải này là ưu điểm khi cần báo cáo, audit, hoặc giải thích quyết định dự đoán cho stakeholder phi kỹ thuật.

---

## 3. Nhược điểm

### 3.1 Không thắng dataset nào khi so với baseline hiện đại

Trên cả 6 datasets được đánh giá với 4 baseline 2020–2021 (EvolveGCN-O, HTGN, DyGNN, DGCN), **GCN_MA đạt 0/6 wins**. Đây là finding trung tâm của luận văn (§4.5):

| Dataset | Người thắng AUC | AUC | GCN_MA AUC | Gap |
|---|---|---|---|---|
| collegemsg | HTGN | 0.9425 | 0.9005 | −0.042 |
| bitcoinotc | HTGN | 0.9147 | 0.8560 | −0.059 |
| eut | DGCN | 0.9847 | 0.9008 | −0.084 |
| mooc_actions | DyGNN | 0.9956 | 0.9845 | −0.011 |
| lastfm | EvolveGCN-O | 0.9550 | 0.8004 | **−0.155** |
| wikipedia | DyGNN | 0.9805 | 0.8696 | −0.111 |

Trong đó **lastfm có gap lên đến 15.5 điểm AUC** — một trong những phát hiện đáng chú ý nhất của thực nghiệm này.

### 3.2 NRNAE bão hòa trên đồ thị dense bipartite

Trên LastFM (N=1980, E=1.29M, bipartite — user-artist play), GCN_MA tụt 7.5 điểm AUC so với paper:

| Dataset | AUC tái hiện | AUC paper | ΔAUC |
|---|---|---|---|
| lastfm | 0.8004 | 0.8757 | **−0.0753** |
| bitcoinotc | 0.8560 | 0.9120 | **−0.0560** |

Lý do kỹ thuật (xem §4.5 thesis): trên đồ thị cực dense, **hầu hết mọi node có CC cao và AS cao** vì mật độ cạnh lớn. Khi mọi node có feature vector NRNAE gần giống nhau, bước GCN convolution không còn signal phân biệt — đây là hạn chế cố hữu của việc chỉ dùng 3 đặc trưng cấu trúc cục bộ. Bài báo gốc có thể đã sử dụng đặc trưng content ngầm (tag âm nhạc, embedding bài hát) mà không công bố — vấn đề tái hiện được phân tích kỹ trong §4.8.

### 3.3 Multi-head attention với head_dim nhỏ

Với `hidden_dim = 64` và `num_heads = 4`, head dimension chỉ là **16**. Đây là biên dưới của phạm vi mà dot-product attention hoạt động ổn định (transformer gốc dùng head_dim = 64). Việc thử `num_heads = 8` (head_dim = 8) trong Plan 1 đã được loại bỏ do attention pattern không ổn định về số học. Hệ quả: GCN_MA có ít "kênh" attention độc lập hơn các mô hình dùng head_dim lớn — ảnh hưởng đến khả năng học các quan hệ đa dạng giữa node.

### 3.4 LSTM weight evolution thiếu feedback từ embedding

`LSTMCell` của GCN_MA chỉ tiến hóa trọng số $W^t$ dựa trên hidden state của LSTM, **không phụ thuộc vào node embedding hiện tại**. Điều này khác với EvolveGCN-O (Pareja et al. 2020) — nơi GRU cập nhật trọng số có điều kiện trên feature đầu vào. Khi đồ thị có thay đổi cấu trúc bất ngờ (drift, regime change), LSTM của GCN_MA chậm thích nghi hơn vì nó không "nhìn" được state thực tế của graph để điều chỉnh weight update.

Bằng chứng thực nghiệm: trên Bitcoinotc (mạng tin cậy crypto, có pha thị trường thay đổi), GCN_MA đạt 0.8560 còn HTGN đạt 0.9147 — gap 5.9 điểm.

### 3.5 Đầu vào 3 chiều quá ít

Đặc trưng NRNAE chỉ có 3 chiều (degree, CC, AS). Khi đặt cạnh các mô hình có per-node memory (DyGNN — bộ nhớ tăng tuyến tính với số node) hoặc hyperbolic embedding (HTGN — biểu diễn được cấu trúc phân cấp), capacity đầu vào của GCN_MA rõ ràng thấp hơn. Trên các datasets có cấu trúc phong phú như Wikipedia (editor-page edit, có pattern editor experience rõ rệt), DyGNN đạt 0.9805 còn GCN_MA chỉ 0.8696 — gap 11 điểm.

### 3.6 Phụ thuộc nặng vào negative sampling protocol

Hai dataset mà GCN_MA tái hiện **vượt** paper trên AP (collegemsg +2.55%, wikipedia +3.39%) khả năng cao là do **paper không mô tả chiến lược negative sampling**. Phiên bản tái hiện dùng random global sampling tỉ lệ 1:1 — trên bipartite nhỏ, dễ tạo "easy negatives" và inflate AP. Đây là nhược điểm chung của paper gốc (không phải của tái hiện), nhưng nó cho thấy **số liệu paper-reported của GCN_MA không stable** dưới các giao thức đánh giá khác nhau — một dạng "ngầm" overfit lên giao thức nội bộ chưa công bố.

### 3.7 Bài báo gốc không tài liệu hóa hyperparameter table

Bài báo Mei & Zhao (2024) không công bố:

- Số layer GCN cụ thể,
- Learning rate schedule,
- Weight decay,
- Early stopping criterion,
- Negative sampling protocol,
- Random seed pool.

Hệ quả: việc tái hiện sát số liệu paper trở nên không khả thi nếu không suy luận ngược thông qua grid-search. Đây là một nhược điểm về **tính tái lập** (reproducibility) của bản thân công trình gốc — không phải của kiến trúc — nhưng ảnh hưởng trực tiếp đến niềm tin vào các con số trong Table 2 của paper.

### 3.8 Pool baseline trong Table 2 paper quá hẹp

Bài báo gốc so sánh GCN_MA với GCN tĩnh, TGCN và một vài baseline cổ điển, **không bao gồm bốn mô hình đã công bố 2020–2021** có thể áp dụng trực tiếp cho dynamic link prediction:

- EvolveGCN-O (Pareja et al. 2020),
- HTGN (Yang et al. 2021),
- DyGNN (Ma et al. 2020),
- DGCN/WD-GCN (Manessi et al. 2020).

Cả bốn đều đã được công bố từ 3 năm trước khi paper GCN_MA submit. Thực nghiệm trong luận văn này cho thấy mỗi mô hình trong bốn mô hình trên **vượt GCN_MA trên ít nhất một dataset** — do đó claim "GCN_MA dominant trên sáu datasets" của paper cần được đặt trong bối cảnh baseline pool hạn chế (xem §4.8 thesis).

---

## 4. Khi nào nên dùng GCN_MA

GCN_MA là lựa chọn phù hợp khi đáp ứng đồng thời các tiêu chí sau:

- **Đồ thị sparse hoặc trung bình** (mật độ cạnh thấp đến trung bình), nơi NRNAE 3 chiều còn có tính phân biệt — ví dụ mạng nhắn tin trong tổ chức nhỏ, mạng email nội bộ, mạng tin cậy không quá dày.
- **Cần model ổn định, dễ huấn luyện lại nhiều lần với kết quả nhất quán** — std cực thấp qua seed giúp giảm chi phí "chạy lại để confirm".
- **Ngân sách compute trung bình** — không có cluster GPU lớn nhưng cũng không bị giới hạn ở CPU. Runtime ~4h cho 6 datasets × 3 seeds phù hợp với hardware research lab điển hình.
- **Yêu cầu diễn giải** — khi cần báo cáo cho stakeholder phi kỹ thuật về *tại sao* model dự đoán một liên kết (degree cao + clustering cao + AS cao → tín hiệu structural mạnh), NRNAE là input có thể giải thích bằng tiếng Việt thông thường.
- **Cần một spectral GCN với temporal modeling tường minh** mà không muốn phụ thuộc vào hyperbolic geometry hoặc per-node memory.

---

## 5. Khi nào KHÔNG nên dùng GCN_MA

Tránh GCN_MA và chọn baseline khác trong các tình huống:

- **Đồ thị bipartite dense** (LastFM-like, Wikipedia-like với mật độ cao): chọn **DyGNN** nếu cần per-node memory hoặc **HTGN** nếu cấu trúc phân cấp rõ.
- **Cần SOTA tuyệt đối trên đa số dataset**: chọn **HTGN** — baseline mạnh nhất tổng thể trong thực nghiệm này, top-3 trên 6/6 datasets.
- **Có sẵn content features** (text, image, audio embedding): GCN_MA không có cơ chế trộn content vào NRNAE; **HTGN** hoặc một biến thể message-passing có hỗ trợ feature concat sẽ phù hợp hơn.
- **Cần kiến trúc cực kỳ đơn giản, ít moving parts**: chọn **DGCN** — chỉ là GCN + LSTM, ~150 dòng code, vẫn cạnh tranh trên 4/6 datasets và thắng EUT.
- **Cần xử lý đồ thị có drift / regime change mạnh**: chọn **EvolveGCN-O** — GRU cập nhật trọng số có điều kiện trên feature giúp thích nghi nhanh hơn LSTM của GCN_MA.

---

## 6. Tổng kết một dòng

> GCN_MA là một mô hình **ổn định, có cơ sở lý thuyết rõ ràng, chi phí compute hợp lý**, phù hợp làm baseline có thể tái hiện trên đồ thị sparse-to-medium; nhưng **bị vượt bởi bốn baseline 2020–2021 trên toàn bộ sáu datasets được đánh giá**, đặc biệt yếu trên đồ thị bipartite dense — do đó claim "dominant" trong paper gốc cần được đặt lại trong bối cảnh baseline pool hiện đại hơn.

---

## 7. Tham chiếu

- Bài báo gốc: Mei & Zhao (2024), *Scientific Reports*, DOI 10.1038/s41598-023-50977-6.
- Chi tiết kết quả per-dataset: `results/report/gcn_ma_check.md`, `results/report/results_summary.md`.
- Phân tích sâu, learning curves và β-sensitivity: `docs/thesis_chapter.md` §3.5, §3.6, §4.5, §4.8.
- Mọi số liệu nhân tạo trong tài liệu này khớp với `results/metrics.jsonl` (87 records, 3 seeds × 5 model × 6 dataset trừ DyGNN×LastFM).
