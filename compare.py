"""
compare.py — So nhiều cấu hình. Công cụ chính của vòng bake-off.

Bản cũ chỉ in bảng điểm rồi xếp hạng. Vấn đề: nó không trả lời được câu hỏi
thật sự quan trọng — **chênh lệch này là thật hay là nhiễu?**

Đo trên 2000 câu train thật, hai kiến trúc khác nhau trượt ở những câu khác
nhau nên độ lệch chuẩn của chênh lệch từng câu lên tới ~0.50. Với dev 300 câu,
chênh lệch quan sát được phải vượt ~0.056 mới đáng tin ở mức 95%. Xếp hạng hai
cấu hình chênh nhau 0.02 trên 300 câu chỉ đúng khoảng 76% số lần.

Nên script làm ba việc:

  1. BẢNG ĐIỂM qua cả ba kịch bản cấu trúc tham chiếu.
  2. SO CẶP CÓ KHOẢNG TIN CẬY — bootstrap theo cặp trên cùng tập câu hỏi.
     So cặp mạnh hơn so hai điểm trung bình rời rạc rất nhiều, vì nhiễu do
     "câu khó / câu dễ" bị triệt tiêu.
  3. KIỂM TRA BẤT BIẾN THỨ HẠNG. Ta chưa biết BTC dùng cấu trúc tham chiếu
     nào, nên một quyết định chỉ đáng tin khi nó không đổi dù kịch bản nào đúng.

Cách dùng:
    python compare.py --preds 'preds/*.json'
    python compare.py --preds 'preds/a.json' 'preds/b.json' --gold data/dev_main.json
"""

import argparse
import glob
import json
import random
from pathlib import Path

from btc_metrics import BACKEND, DEFAULT_MODE, REF_MODES, score_all


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def normalize_preds(raw):
    return {k: (v["answer"] if isinstance(v, dict) else v) for k, v in raw.items()}


def bootstrap_diff(a_scores, b_scores, n_boot=2000, seed=0):
    """Bootstrap THEO CẶP trên cùng tập question_id.

    Trả về (chênh lệch trung bình b−a, cận dưới 95%, cận trên 95%,
            tỉ lệ lần lặp mà b thắng a).
    """
    diffs = [b - a for a, b in zip(a_scores, b_scores)]
    n = len(diffs)
    mean = sum(diffs) / n

    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    win = sum(1 for m in means if m > 0) / n_boot
    return mean, lo, hi, win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True,
                    help="danh sách tệp dự đoán (chấp nhận ký tự đại diện)")
    ap.add_argument("--gold", default="data/dev_main.json")
    ap.add_argument("--boot", type=int, default=2000, help="số lần lặp bootstrap")
    ap.add_argument("--no-pairwise", action="store_true",
                    help="bỏ phần so cặp, chỉ in bảng điểm")
    args = ap.parse_args()

    files = sorted({f for pat in args.preds for f in glob.glob(pat)})
    if not files:
        raise SystemExit("Không tìm thấy tệp dự đoán nào.")

    gold = load_json(args.gold)
    modes = list(REF_MODES)

    table, per_q = {}, {}
    for f in files:
        name = Path(f).stem
        preds = normalize_preds(load_json(f))
        row, per = {}, None
        for m in modes:
            agg, p = score_all(gold, preds, m)
            row[m] = agg["meteor"]
            if m == DEFAULT_MODE:
                per = p
        table[name] = row
        per_q[name] = [per[q]["meteor"] for q in sorted(gold)]

    w = max(max(len(n) for n in table), 14) + 2
    width = w + 12 * len(modes)

    print("=" * width)
    print(f"BẢNG SO SÁNH — METEOR   ({len(gold)} câu, {Path(args.gold).name}"
          + (f", nền chấm: {BACKEND}" if BACKEND != "nltk" else "") + ")")
    print("=" * width)
    print(f"{'Cấu hình':<{w}}" + "".join(f"{m:>12}" for m in modes))
    print("-" * width)
    order = [n for n, _ in sorted(table.items(), key=lambda kv: -kv[1][DEFAULT_MODE])]
    for name in order:
        print(f"{name:<{w}}" + "".join(f"{table[name][m]:>12.4f}" for m in modes))

    # ---------------- So cặp có khoảng tin cậy ----------------
    if not args.no_pairwise and len(order) > 1:
        print("\n" + "=" * width)
        print(f"SO CẶP VỚI BẢN DẪN ĐẦU — '{order[0]}'   (bootstrap {args.boot} lần, "
              f"kịch bản {DEFAULT_MODE})")
        print("=" * width)
        print(f"{'Cấu hình':<{w}}{'chênh':>10}{'KTC 95%':>20}{'':>4}kết luận")
        print("-" * width)
        base = per_q[order[0]]
        for name in order[1:]:
            d, lo, hi, _ = bootstrap_diff(base, per_q[name], args.boot)
            if hi < 0:
                verdict = "THUA rõ rệt"
            elif lo > 0:
                verdict = "THẮNG rõ rệt (xem lại thứ tự xếp hạng)"
            else:
                verdict = "CHƯA phân biệt được — nhiễu"
            print(f"{name:<{w}}{d:>+10.4f}   [{lo:+.4f}, {hi:+.4f}]    {verdict}")

        print("\n  Cách đọc: khoảng tin cậy chứa số 0 nghĩa là tập dev này CHƯA đủ")
        print("  để phân biệt hai cấu hình. Đừng chốt dựa trên chênh lệch đó —")
        print("  hoặc chuyển sang data/dev_main.json nếu đang chạy dev_fast.")

    # ---------------- Bất biến thứ hạng ----------------
    rankings = {m: [n for n, _ in sorted(table.items(), key=lambda kv: -kv[1][m])]
                for m in modes}
    invariant = all(rankings[m] == rankings[modes[0]] for m in modes)

    print("\n" + "=" * width)
    print("KIỂM TRA BẤT BIẾN THỨ HẠNG QUA BA KỊCH BẢN THAM CHIẾU")
    print("=" * width)
    for m in modes:
        print(f"  {m:<14} : {' > '.join(rankings[m])}")

    if invariant:
        print(f"\n  ✓ Thứ hạng GIỐNG NHAU ở cả ba. Chọn '{rankings[modes[0]][0]}' là an toàn,")
        print("    không cần biết BTC dùng cấu trúc tham chiếu nào.")
    else:
        print("\n  ✗ Thứ hạng ĐẢO giữa các kịch bản — CHƯA ĐƯỢC CHỐT.")
        print("    Phải chạy thí nghiệm lặp-câu-hỏi trên public LB trước.")
        print("    Xem README, mục 'Thí nghiệm quyết định'.")

    print("\n  Độ nhạy với việc câu hỏi có nằm trong tham chiếu hay không (full − answer_only):")
    for name in sorted(table, key=lambda n: -abs(table[n]["full"] - table[n]["answer_only"])):
        d = table[name]["full"] - table[name]["answer_only"]
        print(f"    {name:<{w}}{d:+.4f}" + ("   <- rất nhạy" if abs(d) > 0.05 else ""))


if __name__ == "__main__":
    main()
