"""
btc_metrics.py — Sao chép CHÍNH XÁC cách chấm điểm của BTC.

Nguồn đối chiếu: `Scoring-Program-Task-LegalQA/scoring.py`, hàm `eval_qa`.
Nguyên tắc: không "sửa cho đẹp". Mọi đặc điểm của code BTC đều giữ nguyên,
kể cả những chỗ trông như lỗi — vì đó mới là thứ quyết định điểm thật.

BỐN ĐẶC ĐIỂM ĐÃ SAO CHÉP
------------------------
1. KHÔNG tách từ. BTC comment mất `pyvi`, `build_in_tokenizer` trả nguyên
   chuỗi -> METEOR chạy trên `.split()` = tách theo ÂM TIẾT.

2. `str()` áp lên `y_true` mà không bóc `answer`. BTC làm:
       y_pred = {k: v['answer'] for k, v in y_pred.items()}     # CÓ bóc
       ...  meteor_score([str(y_true[k]).split()], str(y_pred[k]).split())
   `y_true` không có dòng bóc tương ứng. Nếu tệp tham chiếu là dict thì
   `str()` kèm luôn dấu ngoặc nhọn và tên khoá vào chuỗi tham chiếu.
   -> Trần điểm có thể KHÔNG phải 1.0. Xem REF_MODES.

3. Vòng lặp chấm chạy theo `ids_preds`, KHÔNG theo `ids_truth`:
       for k in ids_preds   ->   y_true[k]
   Nghĩa là một `question_id` thừa trong bài nộp gây `KeyError` -> crash.
   Phép kiểm `len(ids_preds) != len(ids_truth)` chỉ so SỐ LƯỢNG, không so tập.
   -> Nộp thừa 1 câu và thiếu 1 câu khác vẫn qua được phép kiểm đó rồi chết ở
      dòng dưới. `guard.py` phải so TẬP question_id, không so số lượng.

4. ROUGE-L dùng tokenizer mặc định của `rouge_score`, vốn xoá mọi ký tự ngoài
   [a-z0-9] -> băm nát tiếng Việt. Giữ nguyên để biết con số thật, nhưng
   KHÔNG dùng nó để ra quyết định.

BA KỊCH BẢN THAM CHIẾU
----------------------
Vì chưa biết cấu trúc tệp tham chiếu của BTC, module mô phỏng cả ba qua
tham số `ref_mode`. Dùng `compare.py` để kiểm tra thứ hạng có bất biến không.
"""

import warnings

# ---------------------------------------------------------------------------
# Ưu tiên nltk + rouge_score của BTC. Thiếu thì dùng bản dự phòng thuần Python.
# ---------------------------------------------------------------------------
BACKEND = "nltk"

try:
    import nltk
    from nltk.translate.meteor_score import meteor_score as _meteor

    for _pkg in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{_pkg}.zip")
        except LookupError:
            nltk.download(_pkg, quiet=True)
    # Xác nhận wordnet nạp được thật, không chỉ tải về
    from nltk.corpus import wordnet as _wn
    _wn.ensure_loaded()
except Exception:  # noqa: BLE001
    from meteor_ref import meteor_score as _meteor
    BACKEND = "fallback"

try:
    from rouge_score import rouge_scorer
    _rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    def _rouge_f(ref_str, pred_str):
        return _rouge.score(ref_str, pred_str)["rougeL"].fmeasure
except Exception:  # noqa: BLE001
    from meteor_ref import rouge_l_fmeasure as _rouge_f
    BACKEND = "fallback"

if BACKEND == "fallback":
    warnings.warn(
        "Đang dùng bộ chấm DỰ PHÒNG (thiếu nltk hoặc rouge_score của BTC).\n"
        "  Chạy `python selftest.py` trên máy có nltk để đo sai lệch trước khi\n"
        "  dùng số liệu này ra quyết định. Xem đầu tệp meteor_ref.py.",
        stacklevel=2,
    )

# ---------------------------------------------------------------------------
# Ba kịch bản cấu trúc tệp tham chiếu
# ---------------------------------------------------------------------------
REF_MODES = {
    # truth[k] = "Căn cứ Điều 76..."               -> str() trả đúng chuỗi đó
    "plain":       lambda q, a: a,
    # truth[k] = {"answer": "Căn cứ..."}           -> str() thêm {'answer': '...'}
    "answer_only": lambda q, a: {"answer": a},
    # truth[k] = {"question": ..., "answer": ...}  -> str() kèm CẢ CÂU HỎI
    "full":        lambda q, a: {"question": q, "answer": a},
}

DEFAULT_MODE = "answer_only"  # khớp định dạng submission BTC công bố


def build_reference(question, answer, ref_mode=DEFAULT_MODE):
    """Dựng lại đối tượng tham chiếu theo kịch bản, giống `truth[k]` của BTC."""
    if ref_mode not in REF_MODES:
        raise ValueError(f"ref_mode phải thuộc {list(REF_MODES)}, nhận '{ref_mode}'")
    return REF_MODES[ref_mode](question, answer)


def score_pair(ref_obj, pred_answer, with_rouge=False):
    """Chấm một cặp. Hai dòng dưới là bản sao nguyên văn logic của BTC.

    `with_rouge` mặc định TẮT — có chủ ý. ROUGE-L đã hỏng với tiếng Việt
    (tokenizer xoá mọi ký tự ngoài [a-z0-9], hai câu lạc đề nhau vẫn cùng
    điểm — xem README). Nó là metric phụ và không được dùng để ra quyết định,
    trong khi LCS lại là phần tốn thời gian nhất. Bật bằng --rouge khi thật
    sự cần con số để đưa vào bài báo.
    """
    ref_str = str(ref_obj)
    pred_str = str(pred_answer)
    out = {
        "meteor": float(_meteor([ref_str.split()], pred_str.split())),
        "len_ref": len(ref_str.split()),
        "len_pred": len(pred_str.split()),
    }
    if with_rouge:
        out["rougeL"] = float(_rouge_f(ref_str, pred_str))
    return out


def score_all(gold, preds, ref_mode=DEFAULT_MODE, with_rouge=False):
    """Chấm toàn bộ tập.

    gold  : {qid: {"question": ..., "answer": ...}}
    preds : {qid: "câu trả lời"} hoặc {qid: {"answer": "..."}}
    Trả về (điểm trung bình, điểm từng câu).
    """
    per_sample = {}
    for qid, item in gold.items():
        if qid not in preds:
            raise KeyError(
                f"Thiếu câu {qid} trong bài nộp. BTC sẽ raise Exception."
            )
        p = preds[qid]
        pred_answer = p["answer"] if isinstance(p, dict) else p
        per_sample[qid] = score_pair(
            build_reference(item["question"], item["answer"], ref_mode),
            pred_answer,
            with_rouge=with_rouge,
        )

    n = len(per_sample)
    agg = {"meteor": sum(v["meteor"] for v in per_sample.values()) / n, "n": n}
    if with_rouge:
        agg["rougeL"] = sum(v["rougeL"] for v in per_sample.values()) / n
    return agg, per_sample


def score_all_modes(gold, preds):
    """Chấm trên CẢ BA kịch bản tham chiếu, để kiểm tra tính bất biến."""
    return {mode: score_all(gold, preds, mode)[0] for mode in REF_MODES}
