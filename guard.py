"""
guard.py — Kiểm tra bài nộp TRƯỚC KHI tải lên. Chạy mọi lần, không có ngoại lệ.

BẢN NÀY ĐÃ SỬA THEO MÃ CHẤM THẬT của BTC (09/08). Bản cũ canh sai chỗ.

Task 2 — `Scoring-Program-Task-LegalQA/scoring.py`
--------------------------------------------------
Vòng lặp chấm chạy theo khoá của BÀI NỘP, không theo khoá tham chiếu:

    for k in ids_preds:  ... y_true[k] ...

Kéo theo ba hệ quả, xếp theo mức nguy hiểm:

  1. `question_id` THỪA trong bài nộp -> `y_true[k]` -> KeyError -> crash
     -> KHÔNG CÓ ĐIỂM cho toàn bài. Đây là lỗi chết người số một.
  2. `question_id` THIẾU -> `len(ids_preds) != len(ids_truth)` -> Exception
     -> cũng không có điểm.
  3. Phép kiểm của BTC chỉ so SỐ LƯỢNG. Thừa một câu và thiếu một câu khác
     sẽ LỌT qua phép kiểm đó rồi chết ở dòng dưới.
     -> Guard phải so TẬP question_id, không so số lượng.

Thiếu khoá `answer` -> KeyError ngay ở dòng bóc `v['answer']` -> crash.

Task 1 — `Scoring-Program-Task-LegalIR/scoring.py`  (để đối chiếu)
------------------------------------------------------------------
ĐÍNH CHÍNH so với hồ sơ 02/08: vượt 5 answer KHÔNG làm mất trắng cả bài.
Công thức có `else 0` nên chỉ CÂU ĐÓ bị 0 điểm:

    ... if len(y_pred.get(k)) > 0 and len(y_pred.get(k)) <= 5 else 0

Nhưng `len()` chạy trên danh sách THÔ, còn `set()` chỉ dùng lúc giao. Nộp 6 id
trong đó có 1 trùng vẫn tính là 6 -> câu đó 0 điểm dù chỉ có 5 id phân biệt.
-> Phải KHỬ TRÙNG LẶP TRƯỚC rồi mới cắt về 5. Đúng thứ tự này.
Thiếu `question_id` -> `len(None)` -> TypeError -> mất trắng cả bài.

Cách dùng:
    python guard.py --task 2 --pred preds/final.json \
        --qids ../../drive-download-20260806T033107Z-1-001/public-official.json
    python guard.py --task 2 --pred preds/final.json --qids ... --zip submission.zip
"""

import argparse
import json
import zipfile
from pathlib import Path

MAX_ANSWERS_TASK1 = 5
MEDIAN_REF_LEN = 312  # trung vị độ dài câu trả lời, đo trên 7000 câu train


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def check_qid_set(pred, expected):
    """So TẬP question_id. Đây là phép kiểm quan trọng nhất."""
    errors = []
    exp, got = set(expected), set(pred)

    extra = got - exp
    if extra:
        errors.append(
            f"THỪA {len(extra)} question_id không có trong đề "
            f"-> KeyError khi BTC chấm -> MẤT TRẮNG. Ví dụ: {sorted(extra)[:5]}"
        )
    missing = exp - got
    if missing:
        errors.append(
            f"THIẾU {len(missing)} question_id "
            f"-> số câu không khớp -> Exception -> MẤT TRẮNG. "
            f"Ví dụ: {sorted(missing)[:5]}"
        )
    return errors


def check_task2(pred, expected, autofix=False):
    errors = check_qid_set(pred, expected)
    warnings, fixed = [], {}

    n_short = 0
    for qid in expected:
        if qid not in pred:
            continue
        item = pred[qid]

        if not isinstance(item, dict) or "answer" not in item:
            errors.append(f"[{qid}] không có khoá 'answer' -> KeyError -> MẤT TRẮNG")
            continue

        ans = item["answer"]
        if ans is None:
            errors.append(f"[{qid}] 'answer' là null — đề bài chưa được điền")
            continue
        if not isinstance(ans, str):
            errors.append(f"[{qid}] 'answer' phải là chuỗi, đang là {type(ans).__name__}")
            continue
        if not ans.strip():
            errors.append(f"[{qid}] 'answer' rỗng -> câu này 0 điểm")
            continue
        if "\x00" in ans:
            errors.append(f"[{qid}] chứa ký tự NULL")
            continue

        n_tok = len(ans.split())
        if n_tok < 60:
            n_short += 1
            if n_short <= 5:
                warnings.append(
                    f"[{qid}] chỉ {n_tok} âm tiết. Trung vị tham chiếu ~{MEDIAN_REF_LEN}. "
                    f"METEOR alpha=0.9 phạt viết ngắn rất nặng"
                )
        fixed[qid] = {"answer": ans}

    if n_short > 5:
        warnings.append(f"... tổng cộng {n_short} câu dưới 60 âm tiết")

    return errors, warnings, (fixed if autofix else None)


def check_task1(pred, expected, autofix=False):
    errors = check_qid_set(pred, expected)
    warnings, fixed = [], {}
    n_over, n_dup, n_under = 0, 0, 0

    for qid in expected:
        if qid not in pred:
            continue
        item = pred[qid]

        if not isinstance(item, dict) or "answer" not in item:
            errors.append(f"[{qid}] không có khoá 'answer'")
            continue
        ans = item["answer"]
        if not isinstance(ans, list):
            errors.append(f"[{qid}] 'answer' phải là mảng, đang là {type(ans).__name__}")
            continue

        # Khử trùng lặp TRƯỚC, giữ nguyên thứ tự, rồi mới cắt 5
        seen, dedup = set(), []
        for d in ans:
            d = str(d)
            if d not in seen:
                seen.add(d)
                dedup.append(d)

        if len(dedup) < len(ans):
            n_dup += 1
        if len(ans) > MAX_ANSWERS_TASK1:
            n_over += 1
            if not autofix:
                errors.append(
                    f"[{qid}] nộp {len(ans)} answer (len tính trên danh sách THÔ, "
                    f"kể cả trùng lặp) -> CÂU NÀY 0 ĐIỂM. Dùng --autofix để cắt"
                )
        elif len(dedup) < MAX_ANSWERS_TASK1:
            n_under += 1

        fixed[qid] = {"answer": dedup[:MAX_ANSWERS_TASK1]}

    if n_dup:
        warnings.append(f"{n_dup} câu có document_id trùng lặp — trùng vẫn bị TÍNH vào trần 5")
    if n_under:
        warnings.append(
            f"{n_under} câu nộp dưới 5 id. Recall là metric chính, Precision chỉ "
            f"phá hoà -> luôn nộp đủ 5"
        )
    if n_over and autofix:
        warnings.append(f"{n_over} câu vượt trần 5, đã cắt bằng --autofix")
    return errors, warnings, (fixed if autofix else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, choices=[1, 2], default=2)
    ap.add_argument("--pred", required=True, help="tệp submission.json cần kiểm")
    ap.add_argument("--qids", required=True,
                    help="public-official.json của task tương ứng "
                         "(hoặc tệp JSON chứa danh sách question_id)")
    ap.add_argument("--autofix", action="store_true",
                    help="Task 1: khử trùng lặp rồi cắt về đúng trần 5")
    ap.add_argument("--zip", default=None, help="đóng gói submission.zip nếu sạch lỗi")
    args = ap.parse_args()

    pred = {str(k): v for k, v in load_json(args.pred).items()}
    raw = load_json(args.qids)
    expected = [str(q) for q in (raw if isinstance(raw, (list, dict)) else [])]

    check = check_task1 if args.task == 1 else check_task2
    errors, warnings, fixed = check(pred, expected, args.autofix)

    print("=" * 68)
    print(f"KIỂM TRA BÀI NỘP — Task {args.task}")
    print("=" * 68)
    print(f"Số câu trong đề    : {len(expected)}")
    print(f"Số câu trong bài   : {len(pred)}")
    print(f"Tập question_id    : {'KHỚP' if set(pred) == set(expected) else 'KHÔNG KHỚP'}")

    if warnings:
        print(f"\nCẢNH BÁO ({len(warnings)}):")
        for w in warnings[:15]:
            print(f"  ! {w}")
        if len(warnings) > 15:
            print(f"  ... và {len(warnings) - 15} cảnh báo khác")

    if errors:
        print(f"\nLỖI ({len(errors)}) — KHÔNG ĐƯỢC NỘP:")
        for e in errors[:20]:
            print(f"  X {e}")
        if len(errors) > 20:
            print(f"  ... và {len(errors) - 20} lỗi khác")
        raise SystemExit(1)

    print("\n✓ Không có lỗi chặn. Bài nộp hợp lệ.")

    out_path = Path(args.pred)
    if args.autofix and fixed:
        out_path = out_path.with_name(out_path.stem + "_fixed.json")
        out_path.write_text(json.dumps(fixed, ensure_ascii=False), encoding="utf-8")
        print(f"✓ Đã ghi bản đã sửa: {out_path}")

    if args.zip:
        with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as z:
            # Bên trong zip PHẢI là đúng một tệp tên submission.json
            z.write(out_path, arcname="submission.json")
        print(f"✓ Đã đóng gói: {args.zip}  (chứa duy nhất submission.json)")


if __name__ == "__main__":
    main()
