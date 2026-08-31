"""
selftest.py — Kiểm tra harness có đáng tin không. Chạy TRƯỚC khi tin bất kỳ số nào.

Ba phép kiểm:

  1. NỀN CHẤM ĐANG DÙNG LÀ GÌ — nltk (khớp BTC) hay bản dự phòng?

  2. ĐỐI CHIẾU DỰ PHÒNG VỚI NLTK trên dữ liệu thật. Chỉ chạy được khi có nltk.
     In sai lệch tối đa và trung bình. Nếu sai lệch tối đa < 0.001 thì bản dự
     phòng dùng thay được; lớn hơn thì bắt buộc phải có nltk mới ra quyết định.

  3. TÁI HIỆN CÁC MỐC ĐÃ BIẾT từ hồ sơ phân tích 02/08 — nếu ba con số này
     lệch thì harness đã bị sửa hỏng ở đâu đó.

Cách dùng:
    python selftest.py
    python selftest.py --gold data/dev_main.json --n 300
"""

import argparse
import json
from pathlib import Path

import btc_metrics as M

# Ba mốc từ mục 4.1 hồ sơ bàn giao. Trần phạt phân mảnh đúng 0.5.
KNOWN = [
    ("chép y hệt",            "công ty phải trả lương đúng hạn",
                              "công ty phải trả lương đúng hạn", 0.999),
    ("đảo hai khối",          "công ty phải trả lương đúng hạn",
                              "đúng hạn công ty phải trả lương", 0.988),
    ("xáo tung hoàn toàn",    "công ty phải trả lương đúng hạn",
                              "hạn trả công đúng phải lương ty", 0.500),
]


def check_known():
    print("-" * 66)
    print("3. TÁI HIỆN MỐC ĐÃ BIẾT (hồ sơ 02/08, mục 4.1)")
    print("-" * 66)
    ok = True
    for name, ref, hyp, expect in KNOWN:
        got = M.score_pair(ref, hyp)["meteor"]
        good = abs(got - expect) < 0.002
        ok &= good
        print(f"  {'✓' if good else '✗'} {name:<22} kỳ vọng {expect:.3f}  đo được {got:.3f}")
    return ok


def check_backend_agreement(gold_path, n):
    print("-" * 66)
    print("2. ĐỐI CHIẾU BẢN DỰ PHÒNG VỚI NLTK")
    print("-" * 66)

    if M.BACKEND != "nltk":
        print("  ⚠  Máy này KHÔNG có nltk nên không đối chiếu được.")
        print("     Số liệu đang sinh ra là của bản dự phòng. Chạy lại lệnh này")
        print("     trên máy có nltk trước khi dùng số để chốt cấu hình.")
        return None

    try:
        import meteor_ref
    except ImportError:
        print("  ✗ Không tìm thấy meteor_ref.py")
        return False

    p = Path(gold_path)
    if not p.exists():
        print(f"  ⚠  Chưa có {gold_path}. Chạy make_dev.py trước.")
        return None

    gold = json.loads(p.read_text(encoding="utf-8"))
    items = list(gold.items())[:n]

    diffs = []
    for _, it in items:
        # So trên chính chuỗi tham chiếu thật, và trên một bản bị cắt ngắn
        ref = str({"answer": it["answer"]})
        for hyp in (it["answer"], " ".join(it["answer"].split()[: len(it["answer"].split()) // 2])):
            a = M.score_pair({"answer": it["answer"]}, hyp)["meteor"]
            b = meteor_ref.meteor_score([ref.split()], str(hyp).split())
            diffs.append(abs(a - b))

    mx, mean = max(diffs), sum(diffs) / len(diffs)
    print(f"  Số cặp đối chiếu     : {len(diffs)}")
    print(f"  Sai lệch trung bình  : {mean:.6f}")
    print(f"  Sai lệch TỐI ĐA      : {mx:.6f}")
    if mx < 0.001:
        print("  ✓ Bản dự phòng dùng thay được (chênh dưới 0.001).")
        return True
    print("  ✗ Chênh quá lớn. Chỉ được ra quyết định trên máy có nltk.")
    print("    Nguyên nhân nhiều khả năng: tầng ghép gốc từ Porter của nltk.")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/dev_main.json")
    ap.add_argument("--n", type=int, default=200, help="số câu đem đối chiếu")
    args = ap.parse_args()

    print("=" * 66)
    print("SELFTEST HARNESS — LegalQA")
    print("=" * 66)
    print("-" * 66)
    print("1. NỀN CHẤM ĐANG DÙNG")
    print("-" * 66)
    if M.BACKEND == "nltk":
        print("  ✓ nltk + rouge_score của BTC — khớp tuyệt đối với hệ thống chấm.")
    else:
        print("  ⚠  BẢN DỰ PHÒNG (thiếu nltk hoặc rouge_score).")
        print("     Cài đặt để khớp BTC:  pip install nltk numpy six")

    agree = check_backend_agreement(args.gold, args.n)
    known_ok = check_known()

    print("=" * 66)
    if known_ok and agree is not False:
        print("KẾT LUẬN: harness dùng được.")
        if M.BACKEND != "nltk":
            print("          Nhưng hãy chạy lại trên máy có nltk trước khi chốt.")
    else:
        print("KẾT LUẬN: CÓ VẤN ĐỀ. Đừng tin số liệu cho tới khi sửa xong.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
