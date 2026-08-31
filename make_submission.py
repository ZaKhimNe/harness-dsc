"""
make_submission.py — Dựng `submission.zip` cho đề public 1000 câu.

Việc này nghe tầm thường nhưng là chỗ mất điểm phổ biến nhất: `scoring.py` của
BTC không có fallback, sai định dạng là crash và KHÔNG CÓ ĐIỂM. Script gom mọi
bước vào một lệnh, và không cho phép bỏ qua bước kiểm tra.

Bốn việc:
  1. Đọc đề `public-official.json`, lấy đúng tập 1000 question_id.
  2. Ghép dự đoán của bạn vào. Câu nào thiếu thì điền câu mặc định
     (`--fill`) — có điểm thấp còn hơn crash mất trắng cả bài.
  3. Chạy toàn bộ phép kiểm của `guard.py`.
  4. Đóng gói `submission.zip` chứa DUY NHẤT `submission.json`.

Chốt chặn 10/08 — nộp một bản hợp lệ trước khi có mô hình:

    python make_submission.py --pred '' --out submission_smoke.zip
        (không có dự đoán nào -> mọi câu dùng câu mặc định.
         Mục đích là xác nhận đường ống nộp bài chạy được, không phải ăn điểm.
         Public test submit không giới hạn nên phép thử này miễn phí.)

Dùng bình thường:

    python make_submission.py --pred preds/he_thong_v3.json --out submission.zip

Thí nghiệm lặp-câu-hỏi (xem README):

    python make_submission.py --pred preds/v3.json --out sub_A.zip
    python make_submission.py --pred preds/v3.json --out sub_B.zip --prepend-question
"""

import argparse
import json
import zipfile
from pathlib import Path

import guard

DEFAULT_PUBLIC = "../../drive-download-20260806T033107Z-1-001/public-official.json"

# Câu mặc định cho ô trống. Dài vừa phải và đúng văn phong tham chiếu, để câu
# thiếu vẫn nhặt được vài phần trăm thay vì 0. KHÔNG phải chiến lược, chỉ là lưới an toàn.
FILL = (
    "Căn cứ quy định của pháp luật hiện hành về vấn đề này, nội dung được quy "
    "định cụ thể như sau: Các cơ quan, tổ chức, cá nhân có liên quan có trách "
    "nhiệm thực hiện đầy đủ các quy định của pháp luật, bảo đảm đúng trình tự, "
    "thủ tục và thời hạn theo quy định. Trường hợp vi phạm thì tuỳ theo tính "
    "chất, mức độ mà bị xử lý theo quy định của pháp luật. Theo đó, việc áp "
    "dụng được thực hiện theo các văn bản quy phạm pháp luật có liên quan đang "
    "còn hiệu lực thi hành."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", default=DEFAULT_PUBLIC, help="đề public-official.json")
    ap.add_argument("--pred", default="", help="tệp dự đoán; để trống = dùng toàn câu mặc định")
    ap.add_argument("--out", default="submission.zip")
    ap.add_argument("--fill", default=None, help="câu mặc định cho ô trống")
    ap.add_argument("--prepend-question", action="store_true",
                    help="lặp câu hỏi ở đầu mỗi câu trả lời (nhánh B của thí nghiệm)")
    ap.add_argument("--task", type=int, choices=[1, 2], default=2)
    args = ap.parse_args()

    public = json.loads(Path(args.public).read_text(encoding="utf-8"))
    qids = [str(k) for k in public]
    fill = args.fill or FILL

    raw = {}
    if args.pred:
        raw = {str(k): v for k, v in
               json.loads(Path(args.pred).read_text(encoding="utf-8")).items()}

    submission, n_filled = {}, 0
    for qid in qids:
        v = raw.get(qid)
        ans = v.get("answer") if isinstance(v, dict) else v
        if not isinstance(ans, str) or not ans.strip():
            ans, n_filled = fill, n_filled + 1
        if args.prepend_question:
            ans = f"{public[qid]['question']}\n{ans}"
        submission[qid] = {"answer": ans}

    tmp = Path(args.out).with_suffix(".json")
    tmp.write_text(json.dumps(submission, ensure_ascii=False), encoding="utf-8")

    print("=" * 68)
    print("DỰNG BÀI NỘP")
    print("=" * 68)
    print(f"Đề              : {args.public}  ({len(qids)} câu)")
    print(f"Dự đoán         : {args.pred or '(không có — dùng toàn câu mặc định)'}")
    print(f"Câu phải điền   : {n_filled}/{len(qids)}"
          + ("   <- toàn bộ là câu mặc định, đây chỉ là phép thử đường ống"
             if n_filled == len(qids) else ""))
    print(f"Lặp câu hỏi     : {'CÓ (nhánh B)' if args.prepend_question else 'không (nhánh A)'}")
    print()

    errors, warnings, _ = (guard.check_task2 if args.task == 2 else guard.check_task1)(
        submission, qids, autofix=False)

    if warnings:
        print(f"CẢNH BÁO ({len(warnings)}):")
        for w in warnings[:8]:
            print(f"  ! {w}")
        print()
    if errors:
        print(f"LỖI ({len(errors)}) — KHÔNG ĐÓNG GÓI:")
        for e in errors[:10]:
            print(f"  X {e}")
        raise SystemExit(1)

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(tmp, arcname="submission.json")
    tmp.unlink()

    size = Path(args.out).stat().st_size / 1024
    print(f"✓ Bài nộp hợp lệ. Đã đóng gói {args.out} ({size:.0f} KB, chứa submission.json)")


if __name__ == "__main__":
    main()
