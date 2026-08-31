"""
ir_metrics.py — Sao chép CHÍNH XÁC cách chấm Task 1 của BTC.

Nguồn đối chiếu: `Scoring-Program-Task-LegalIR/scoring.py`, hàm `eval_retrieval`.
Nguyên tắc giống `btc_metrics.py`: không "sửa cho đẹp". Harness phải sai giống
hệt hệ thống chấm sai.

BỐN ĐẶC ĐIỂM ĐÃ SAO CHÉP
------------------------
1. Điều kiện hợp lệ đếm trên DANH SÁCH THÔ, không phải tập:

       ... if len(y_pred.get(k)) > 0 and len(y_pred.get(k)) <= 5 else 0

   Nộp 6 id trong đó 1 trùng -> len(danh sách) = 6 -> **câu đó 0 điểm**, dù chỉ
   có 5 id phân biệt. `guard.py --autofix` phải KHỬ TRÙNG LẶP TRƯỚC, RỒI cắt 5.

2. `else 0` nằm trong biểu thức từng câu -> vượt trần 5 chỉ giết **câu đó**,
   KHÔNG giết cả bài nộp. Đây là đính chính so với hiểu lầm ban đầu.

3. Thiếu `question_id`: `y_pred.get(k)` trả None -> `len(None)` -> TypeError
   -> crash -> **mất trắng**. Nguy hiểm hơn vượt trần rất nhiều.

4. Precision chia cho `len(y_pred[k])` THÔ trong khi tử số dùng `set()`.
   -> Trùng lặp bị phạt HAI LẦN: vừa làm phình mẫu số, vừa có thể đẩy qua trần 5.

ẨN SỐ 1 — DẠNG TỆP THAM CHIẾU (giống hệt bug str() bên Task 2)
--------------------------------------------------------------
BTC bóc `answer` cho `y_pred` nhưng KHÔNG bóc cho `y_true`:

    y_pred = {k: v['answer'] for k, v in y_pred.items()}   # CÓ bóc
    y_true = {k: v           for k, v in y_true.items()}   # KHÔNG bóc

Nếu tệp tham chiếu có dạng như `train.json` (`{qid: {question, answer}}`) thì
`set(y_true[k])` = `{"question", "answer"}` — TÊN KHOÁ, không phải id văn bản —
và `len(y_true[k])` = 2. Recall về 0 cho MỌI bài nộp.

    -> Hai trong ba dạng là SUY BIẾN (ai cũng 0 điểm). Nên tệp tham chiếu gần
       như chắc chắn là `{qid: [ids]}`. Kiểm bằng `--ref-shape` nếu nghi ngờ,
       nhưng đừng tốn thời gian: ẩn số THẬT của Task 1 nằm ở dưới.

ẨN SỐ 2 — LEADERBOARD CHẤM PRECISION HAY RECALL? (quan trọng nhất)
-------------------------------------------------------------------
`eval_retrieval` trả về CẢ HAI. `metadata.yaml` chỉ ghi `command: python3
scoring.py`, không nói lấy cột nào. Đây là quyết định lớn nhất của Task 1:

    LB dùng      | Nên nộp   | Vì
    -------------|-----------|------------------------------------------
    chỉ recall   | LUÔN 5 id | thêm id không mất gì, chỉ có thể được thêm
    chỉ precision| 1 id      | gold trung bình chỉ 1,09 văn bản/câu
    trung bình   | tuỳ tin   | nộp 5 khi gold=1 -> precision 0,2, recall 1,0

Chênh lệch giữa hai chiến thuật này LỚN HƠN phần lớn cải tiến mô hình.
Public test submit không giới hạn -> đo thẳng: nộp cùng hệ thống hai lần,
k=1 và k=5, xem cột nào nhúc nhích.

Vì vậy module tham số hoá **chỉ số chính**, không phải dạng tham chiếu:

    REF_MODES = {"recall", "precision", "mean"}

`compare.py` sẽ kiểm THỨ HẠNG CÓ BẤT BIẾN qua cả ba hay không — đúng cùng cơ
chế đã dùng cho `plain/answer_only/full` bên Task 2. Một quyết định chỉ đáng
tin khi nó không đổi dù BTC lấy cột nào.
"""

BACKEND = "exact"   # Task 1 không cần nltk -> luôn khớp BTC, không có bản dự phòng
PRIMARY = "score"   # tên khoá điểm chính, để evaluate/compare dùng chung

# Ba cách BTC có thể tổng hợp. Xem ẩn số 2.
REF_MODES = {
    "recall":    lambda p, r: r,
    "precision": lambda p, r: p,
    "mean":      lambda p, r: (p + r) / 2,
}

DEFAULT_MODE = "recall"   # giả định thận trọng: recall là cột hay được báo nhất

MAX_IDS = 5               # trần cứng của BTC


def _as_list(v):
    """Bóc `answer` giống BTC làm với y_pred, và chịu được vài dạng tệp khác."""
    if isinstance(v, dict):
        v = v.get("answer", v.get("ids", []))
    if not isinstance(v, list):
        v = [v]
    return [str(x) for x in v]


def _as_truth(v, ref_shape="list"):
    """Dựng lại `y_true[k]` đúng như BTC NHÌN THẤY nó.

    ref_shape='list'   -> {qid: [ids]}                  (dạng khả dĩ duy nhất)
    ref_shape='dict'   -> {qid: {question, answer}}     -> set() ra TÊN KHOÁ
    """
    if ref_shape == "dict" and isinstance(v, dict):
        return list(v)                      # set(dict) = tên khoá. SUY BIẾN.
    return _as_list(v)


def score_pair(true_ids, pred_ids):
    """Chấm một câu. Ba dòng dưới là bản sao nguyên văn logic của BTC."""
    k_raw = len(pred_ids)                   # THÔ — trùng lặp vẫn tính
    hit = len(set(true_ids) & set(pred_ids))
    valid = 0 < k_raw <= MAX_IDS            # `else 0` của BTC
    recall = hit / len(true_ids) if valid and true_ids else 0.0
    precision = hit / k_raw if valid else 0.0
    return {
        "recall": recall,
        "precision": precision,
        "hit": hit,
        "k": k_raw,
        "k_distinct": len(set(pred_ids)),
        "n_true": len(true_ids),
        "valid": valid,
    }


def score_all(gold, preds, ref_mode=DEFAULT_MODE, ref_shape="list", **_):
    """Chấm toàn bộ tập. Cùng chữ ký với `btc_metrics.score_all`.

    gold  : {qid: [ids]} hoặc {qid: {"question":..., "answer": [ids]}}
    preds : {qid: [ids]} hoặc {qid: {"answer": [ids]}}
    """
    if ref_mode not in REF_MODES:
        raise ValueError(f"ref_mode phải thuộc {list(REF_MODES)}, nhận '{ref_mode}'")

    per_sample = {}
    for qid, item in gold.items():
        if qid not in preds:
            raise KeyError(
                f"Thiếu câu {qid} trong bài nộp. BTC sẽ TypeError -> crash -> mất trắng."
            )
        row = score_pair(_as_truth(item, ref_shape), _as_list(preds[qid]))
        row[PRIMARY] = REF_MODES[ref_mode](row["precision"], row["recall"])
        per_sample[qid] = row

    n = len(per_sample)
    vals = per_sample.values()
    agg = {
        PRIMARY: sum(v[PRIMARY] for v in vals) / n,
        "recall": sum(v["recall"] for v in vals) / n,
        "precision": sum(v["precision"] for v in vals) / n,
        "n": n,
        "n_invalid": sum(1 for v in vals if not v["valid"]),
        "n_dup": sum(1 for v in vals if v["k"] != v["k_distinct"]),
        "k_mean": sum(v["k"] for v in vals) / n,
    }
    return agg, per_sample


def score_all_modes(gold, preds, ref_shape="list"):
    """Chấm trên CẢ BA cách tổng hợp, để kiểm tính bất biến thứ hạng."""
    return {m: score_all(gold, preds, m, ref_shape)[0] for m in REF_MODES}
