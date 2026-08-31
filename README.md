# harness-dsc — Bộ đánh giá nội bộ cho DSC 2026 (LegalIR / LegalQA)

Harness chấm điểm sao chép **chính xác** `scoring.py` của ban tổ chức, kể cả
những chỗ trông như lỗi.

Mục đích không phải "chấm cho ra số", mà là trả lời **một câu hỏi duy nhất:
cấu hình A hay B tốt hơn, và chênh lệch đó có thật không?** — nhanh, lặp lại
được, không cần chờ leaderboard.

---

## Cài đặt

```bash
git clone https://github.com/ZaKhimNe/harness-dsc
cd harness-dsc
pip install nltk numpy six
```

Thư mục `rouge_score/` sao nguyên từ gói BTC dùng (Google Research, Apache-2.0).
**Đừng thay bằng `pip install rouge-score`** — phiên bản có thể khác, mà ta cần
khớp đúng bản BTC chạy.

Lần chạy đầu `nltk` tự tải `wordnet` và `omw-1.4` (cần mạng, một lần duy nhất).

**Không có mạng?** Harness vẫn chạy bằng bộ chấm dự phòng thuần Python
(`meteor_ref.py`), nhưng phải chạy `selftest.py` trên máy có `nltk` để đo sai
lệch **trước khi** dùng số ra quyết định. Mọi script sẽ in cảnh báo khi đang ở
chế độ dự phòng.

---

## Dữ liệu — không nằm trong repo

Dữ liệu BTC (`train.json`, `public-official.json`, `selected-contexts.zip`)
**cố ý không được commit**. Tự tải từ kênh chính thức của cuộc thi rồi đặt cạnh
repo. Chỉ `data/dev.lock` được commit — nó chứa vân tay SHA-256 để mọi người
xác minh đang dùng **cùng một tập dev**.

```
../
├── harness-dsc/                      <- repo này
└── drive-download-.../
    ├── train.json
    ├── public-official.json
    └── selected-contexts.zip
```

---

## Năm bước bắt đầu

```bash
# 0. Harness có đáng tin không?
python selftest.py

# 1. Đóng băng tập dev. CHẠY MỘT LẦN DUY NHẤT.
python make_dev.py --input ../drive-download-.../train.json

# 2. Sinh baseline trần/sàn
python baselines.py --gold data/dev_main.json --outdir preds

# 3. Chấm một cấu hình
python evaluate.py --pred preds/oracle_copy.json

# 4. So nhiều cấu hình (vòng bake-off)
python compare.py --preds 'preds/*.json'
```

Sau bước 1, **kiểm `data/dev.lock` khớp với vân tay trong repo rồi không sửa
nữa.** `make_dev.py` sẽ từ chối chạy lại nếu đã có `dev.lock`.

---

## Kiến trúc: mọi thứ xoay quanh `preds/*.json`

```json
{ "10001": {"answer": "Căn cứ Điều 32 Thông tư 08/2022/TT-BYT ..."},
  "10002": {"answer": "..."} }
```

Mọi script chỉ làm một trong hai việc — **sinh ra** tệp này, hoặc **đọc** nó:

```
   HỆ THỐNG CỦA BẠN                     │      HARNESS
   ─────────────────                    │      ───────
   câu hỏi                              │
     ↓                                  │
   ① truy hồi  → context_237045         │
     ↓                                  │
   ② định vị Điều                       │
     ↓                                  │
   ③ sinh câu trả lời (template / LLM)  │
     ↓                                  │
   preds/v1.json  ──────────────────────┼──→  evaluate.py  điểm + chẩn đoán
                                        │     compare.py   A vs B, có KTC
                                        │     guard.py     hợp lệ chưa
                                        │     make_submission.py  → .zip
```

Nhờ ranh giới này, thêm một hệ thống mới chỉ cần viết phần **sinh** — bốn script
tiêu thụ dùng lại nguyên vẹn. Đây cũng là điều kiện để bake-off hoạt động: ba
thành viên nộp ba tệp cùng định dạng, `compare.py` đọc cả ba mà không cần biết
chúng được sinh ra thế nào.

---

## Từng script làm gì

| Tệp | Vai | Chức năng |
|---|---|---|
| `selftest.py` | hạ tầng | **Chạy đầu tiên.** Harness có chấm đúng không |
| `btc_metrics.py` | hạ tầng | Bản sao cách chấm của BTC. Mọi script gọi nó |
| `meteor_ref.py` | hạ tầng | METEOR thuần Python, dự phòng khi thiếu `nltk` |
| `make_dev.py` | hạ tầng | Đóng băng tập dev hai tầng. **Chạy một lần** |
| `make_silver_labels.py` | hạ tầng | Dựng nhãn văn bản nguồn cho Task 2 (xem dưới) |
| `baselines.py` | sinh | 7 baseline trần/sàn |
| `baseline_template.py` | sinh | Baseline thật: truy hồi + định vị Điều + template |
| `evaluate.py` | đọc | 1 tệp → điểm + chẩn đoán độ dài + phân rã + 5 câu tệ nhất |
| `compare.py` | đọc | N tệp → xếp hạng + khoảng tin cậy + kiểm bất biến |
| `guard.py` | đọc | Kiểm định dạng, cả Task 1 và Task 2, có mã thoát |
| `make_submission.py` | đọc | Dựng `submission.zip`, tự điền câu thiếu |

### `evaluate.py` và `compare.py` không thay thế nhau

| | Trả lời câu gì | Dùng khi |
|---|---|---|
| `evaluate.py` | *"Hệ thống này hỏng ở đâu?"* | Debug **một** cấu hình |
| `compare.py` | *"A hay B tốt hơn, chênh lệch có thật không?"* | **Chọn** giữa nhiều cấu hình |

`compare.py` dùng **bootstrap so cặp** trên cùng tập câu hỏi, nên nhiễu
"câu này khó, câu kia dễ" bị triệt tiêu. Quan trọng vì nhiều quyết định nằm ở
khoảng chênh 0,01–0,03.

---

## Tập dev hai tầng — và vì sao

| Tệp | Cỡ | Dùng khi nào |
|---|---|---|
| `data/dev_fast.json` | **300** | Thí nghiệm hàng ngày: chỉnh nút, sửa prompt |
| `data/dev_main.json` | **1000** | Điểm quyết định: bake-off, chốt kiến trúc |
| `data/sft.json` | 6000 | Huấn luyện. **Không bao giờ** đem chấm |

`dev_fast` là **tập con lồng trong** `dev_main` — hai số luôn so được với nhau,
và tổng số câu lấy khỏi `train.json` chỉ là **1000**, không phải 1300.

Cỡ dev ra từ một công thức: `MDE ≈ 1,96 × sd / √n`.

| So cái gì | sd chênh/câu | n cần | Chênh nhỏ nhất phát hiện được |
|---|---|---|---|
| Hai **kiến trúc** khác nhau | 0,50 | 1000 | 0,031 |
| Cùng hệ thống, **đổi một nút** | 0,037 | 300 | 0,004 |

Chi phí thật của dev lớn không nằm ở lúc chấm (1000 câu < 1 giây) mà ở lúc
**sinh** câu trả lời. Đó là lý do có tầng nhanh.

---

## Bốn đặc điểm của cách chấm BTC — đã sao chép nguyên

**1. Không tách từ.** `build_in_tokenizer` trả nguyên chuỗi, `pyvi` bị comment
→ METEOR chạy trên `.split()`, tức tách theo **âm tiết**.

**2. Recall nặng gấp 9 lần Precision** (`nltk` mặc định α = 0,9). Đo trên dev:

| | Lệch độ dài | Mất điểm |
|---|---|---|
| Cắt còn 80% | thiếu 20% | **−0,228** |
| Đệm gấp rưỡi | thừa 50% | **−0,081** |

Thiếu 20% ≈ phình 2,5 lần. **Không bao giờ viết ngắn.**

**3. `str()` áp lên `y_true` mà không bóc `answer`.** Nếu tệp tham chiếu là dict,
`str()` kèm luôn dấu ngoặc nhọn và tên khoá vào chuỗi tham chiếu. Chưa biết
cấu trúc thật, nên harness mô phỏng **cả ba kịch bản**: `plain`, `answer_only`,
`full`. Một quyết định chỉ đáng tin khi thứ hạng **không đổi** qua cả ba —
`compare.py` kiểm tự động.

**4. ROUGE-L hỏng với tiếng Việt.** Tokenizer xoá mọi ký tự ngoài `[a-z0-9]`:

```
"Căn cứ Điều 76 Bộ luật Lao động"
→ ['c','n','c','i','u','76','b','lu','t','lao','ng']
```

Harness **mặc định không tính ROUGE-L**. Bật bằng `--rouge` khi cần số cho bài báo.

---

## Guard — chạy trước MỌI lần nộp

```bash
python guard.py --task 2 --pred preds/final.json \
    --qids ../drive-download-.../public-official.json \
    --zip submission.zip
```

Vòng lặp chấm của BTC chạy theo khoá **bài nộp**, không theo khoá tham chiếu:

```python
for k in ids_preds:  ... y_true[k] ...
```

Ba hệ quả:

1. `question_id` **thừa** → `KeyError` → crash → mất trắng.
2. `question_id` **thiếu** → số câu không khớp → `Exception` → mất trắng.
3. BTC chỉ so **số lượng**, không so tập. Thừa một câu và thiếu một câu khác
   lọt qua phép kiểm đó rồi chết ở dòng dưới.

→ Guard so **TẬP** `question_id`, không so số lượng.

Riêng Task 1: `len()` đếm trên danh sách **thô**, `set()` chỉ dùng lúc giao.
Nộp 6 id trong đó 1 trùng vẫn tính là 6 → câu đó 0 điểm dù chỉ có 5 id phân
biệt. `--autofix` **khử trùng lặp trước, rồi mới cắt 5** — đúng thứ tự này.

Guard trả mã thoát khác 0 khi có lỗi, nên nối được vào script tự động.

---

## `baseline_template.py` — sàn để so mọi thứ khác

Truy hồi + định vị điều luật + template cứng. Không mô hình sinh, không GPU,
không tham số học được.

```bash
python baseline_template.py --retrieved retrieved_dev.json \
       --gold data/dev_fast.json --locator article --out preds/tpl_article.json
```

Định dạng `--retrieved` (tự nhận diện cả ba):

```json
{"10001": ["280282", "56081"]}
{"10001": {"answer": ["280282"]}}
{"10001": [{"id": "280282", "score": 12.3}]}
```

### Baseline có BA tầng, không phải hai

```
① TRUY HỒI VĂN BẢN   8.532 tệp → 1–3 văn bản
② ĐỊNH VỊ ĐIỀU LUẬT  ← tầng dễ bị bỏ sót
③ GHÉP TEMPLATE
```

| | Âm tiết |
|---|---|
| 1 văn bản (trung vị) | **4.945**, 13 Điều |
| Gold answer | **347** |
| Tỉ lệ | **14×** |

Corpus là **văn bản luật đầy đủ**, không phải từng điều. Tìm đúng văn bản mới
là nửa việc — trong 13 Điều đó phải biết chép Điều nào.

### Kết quả trên `dev_fast`, truy hồi oracle

| Cấu hình | plain | ans_only | full | Độ dài |
|---|---|---|---|---|
| `echo_question` — **sàn** | 0,0658 | 0,0670 | 0,0667 | 19 |
| `template_only` | 0,0846 | 0,0763 | 0,0727 | 67 |
| `tpl_whole` — dán cả văn bản | 0,1411 | 0,1343 | 0,1403 | 17.361 |
| `oracle_half` | 0,4751 | 0,4549 | 0,4225 | 154 |
| `tpl_window` — cửa sổ trượt | 0,4926 | 0,4759 | 0,4851 | 577 |
| **`tpl_article`** ← baseline | **0,5144** | **0,4965** | **0,5057** | 620 |
| `oracle_copy` — **trần** | 1,0000 | 0,9551 | 0,8947 | 308 |

Thứ hạng **bất biến qua cả ba kịch bản** tham chiếu.

**Tầng ② đáng +0,373** (0,141 → 0,514) — gấp 3,6 lần giá trị của bản thân
template (+0,104). Nếu chỉ sửa được một chỗ, sửa tầng định vị.

**Tách theo `Điều` thắng cửa sổ trượt +0,022** ở mọi mức độ dài. Cấu trúc văn
bản luật là tín hiệu thật.

### Vì sao template có dạng này

| | plain |
|---|---|
| Chỉ nguyên văn điều luật, không khung | 0,7344 |
| + khung **CHUNG** (không cần cite thật) | 0,7970 ← **+0,063 miễn phí** |
| + khung có **cite đúng** | 0,8215 ← cite chỉ thêm +0,025 |
| + đoạn kết | 0,8378 |

Khung không phải trang trí. Gold answer luôn có nó: **88,0%** mở bằng
`Căn cứ`/`Theo`/`Tại`, **85,8%** chứa `như sau:`, **62,0%** có cụm kết
`Theo đó`/`Như vậy`.

### Nghịch lý độ dài — đọc kỹ chỗ này

`LEN_TARGET` dò ra trên dev, không phải đoán:

| target | 250 | 300 | 350 | 400 | **480** | 600 |
|---|---|---|---|---|---|---|
| `article` | .4712 | .4965 | .5050 | .5110 | **.5144** | .5122 |

Độ dài tối ưu (~620 âm tiết) **gấp đôi trung vị gold** (305). Đường cong đệm ở
trên nói 2× mất 0,16 — nhưng nó giả định nội dung **đúng tuyệt đối**. Khi nội
dung là văn bản truy hồi chưa chắc đúng, recall mới là ràng buộc: viết thêm rẻ
hơn viết thiếu.

> **Hệ quả:** `LEN_TARGET` là tham số **phụ thuộc chất lượng tầng dưới**, không
> phải hằng số. Khi thay bằng LLM sinh câu chính xác hơn, **phải dò lại** — nó
> sẽ giảm.

---

## `make_silver_labels.py` — nhãn văn bản nguồn cho Task 2

`train.json` Task 2 chỉ có `question` + `answer`, **không có nhãn văn bản
nguồn**. Nhưng gold answer trích **nguyên văn** điều luật → khớp chuỗi ngược về
corpus lấy lại được **97,0%** (đo trên `dev_fast`).

```bash
python make_silver_labels.py --gold data/dev_fast.json --out retrieved_oracle.json
```

Dùng để:

1. Sinh tệp `--retrieved` oracle, đo **trần** của kiến trúc template khi tầng ①
   hoàn hảo — tách bạch lỗi truy hồi khỏi lỗi định vị.
2. **7.000 cặp (câu hỏi → văn bản)** huấn luyện retriever cho Task 2.
3. Ước lượng trần recall cho nhóm IR trên chính phân bố câu hỏi Task 2.

**Hai cảnh báo.** Nhãn bạc **có nhiễu**: 97% khớp không có nghĩa 97% đúng — khớp
chuỗi có thể trúng nhầm văn bản chứa đoạn giống nhau (bản gốc vs bản sửa đổi).
Nên đo tỉ lệ khớp đa nghĩa trước khi dùng fine-tune. Và đây là **suy ra nhãn từ
dữ liệu BTC cấp**, không phải tạo dữ liệu mới — nhiều khả năng hợp lệ, nhưng
nên xác nhận với BTC cho chắc.

---

## Nhật ký thí nghiệm

Mỗi lần submit public, ghi lại **bộ ba số**:

| Ngày | Cấu hình | dev_fast | dev_main | public LB | Lệch (LB − dev_main) |
|---|---|---|---|---|---|
| | | | | | |

Theo dõi cột **Lệch**. Ổn định → harness đáng tin, cứ dùng dev chạy nhanh.
Ngày càng lớn → đang overfit public leaderboard, quay về tin dev nội bộ.

---

## Bảy nguyên tắc dựng harness

| # | Nguyên tắc |
|---|---|
| 1 | Xác định harness dùng để **ra quyết định gì** trước khi viết code. Ở đây: cần đúng **thứ hạng**, không cần đúng **giá trị tuyệt đối** |
| 2 | **Chạy thử** code chấm, đừng chỉ đọc. Mỗi giả thuyết phải có phép kiểm phân biệt được hai khả năng |
| 3 | Sao chép **kể cả lỗi**. Harness phải sai giống hệt hệ thống chấm sai |
| 4 | Với điều chưa biết: **tham số hoá và kiểm tra tính bất biến**, đừng đoán |
| 5 | **Đóng băng** thước đo trước khi đo |
| 6 | Dựng **trần và sàn** trước khi dựng hệ thống |
| 7 | In cái **chẩn đoán được**, không chỉ in điểm. Và in **khoảng tin cậy** — một con số không kèm sai số thì không chốt được |

---

## Giấy phép

Mã trong repo theo `LICENSE`. Thư mục `rouge_score/` là của Google Research
(Apache-2.0), vendor nguyên bản để khớp đúng phiên bản BTC dùng — bản quyền
thuộc tác giả gốc.

Dữ liệu cuộc thi **không** thuộc repo này và không được phân phối lại ở đây.
