"""
evaluate.py — Chấm một tệp dự đoán trên tập dev đã đóng băng.

Điểm số chỉ là một dòng. Phần có giá trị hơn là CHẨN ĐOÁN bên dưới:

  - Tỉ lệ độ dài t/r. METEOR có alpha = 0.9 nên Recall nặng gấp 9 lần
    Precision; viết ngắn hơn tham chiếu bị phạt nặng hơn viết dài 3–4 lần.
    Đo trên dữ liệu thật: giữ 80% độ dài mất ~0.19 điểm, còn viết dài gấp rưỡi
    chỉ mất ~0.08. Nếu phần lớn câu đang ngắn hơn tham chiếu thì đó là chỗ
    mất điểm rẻ nhất, sửa trước khi đụng vào mô hình.

  - Khoảng tin cậy của chính điểm số, để biết con số này ổn định tới đâu.

  - Phân rã điểm theo nhóm độ dài tham chiếu, để thấy hệ thống hỏng ở đâu:
    câu trả lời ngắn hay câu trả lời dài.

  - 5 câu tệ nhất, làm nguyên liệu phân tích lỗi cho bài báo.

Cách dùng:
    python evaluate.py --pred preds/oracle_copy.json
    python evaluate.py --pred preds/he_thong_v3.json --gold data/dev_fast.json
    python evaluate.py --pred preds/he_thong_v3.json --modes all --save scores_v3.json
"""

import argparse
import json
import random
import statistics as st
from pathlib import Path

from btc_metrics import BACKEND, DEFAULT_MODE, REF_MODES, score_all


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def normalize_preds(raw):
    """Chấp nhận cả {qid: "text"} lẫn {qid: {"answer": "text"}}."""
    return {k: (v["answer"] if isinstance(v, dict) else v) for k, v in raw.items()}


def ci_mean(values, n_boot=2000, seed=0):
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += values[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def report(gold, preds, name, modes, with_rouge=False):
    print("=" * 70)
    print(f"CẤU HÌNH: {name}")
    if BACKEND != "nltk":
        print("⚠  Nền chấm DỰ PHÒNG — chạy selftest.py trên máy có nltk trước khi chốt.")
    print("=" * 70)

    results = {m: score_all(gold, preds, m, with_rouge=with_rouge) for m in modes}

    hdr = f"\n{'Kịch bản tham chiếu':<18}{'METEOR':>12}"
    print(hdr + (f"{'ROUGE-L':>12}" if with_rouge else ""))
    print("-" * (42 if with_rouge else 30))
    for m in modes:
        agg = results[m][0]
        line = f"{m:<18}{agg['meteor']:>12.4f}"
        if with_rouge:
            line += f"{agg['rougeL']:>12.4f}"
        print(line + ("  <- chính" if m == DEFAULT_MODE else ""))

    main_mode = DEFAULT_MODE if DEFAULT_MODE in results else modes[0]
    agg, per = results[main_mode]
    vals = [v["meteor"] for v in per.values()]
    lo, hi = ci_mean(vals)
    print(f"\nSố mẫu: {agg['n']}   KTC 95% của METEOR: [{lo:.4f}, {hi:.4f}]")
    if with_rouge:
        print("(ROUGE-L đã hỏng với tiếng Việt — đừng tối ưu theo nó. Xem README.)")

    # ---- Chẩn đoán độ dài (dùng kịch bản 'plain' cho số sạch) ----
    _, per_plain = score_all(gold, preds, "plain")
    ratios = sorted(v["len_pred"] / max(v["len_ref"], 1) for v in per_plain.values())
    n = len(ratios)
    short = sum(1 for r in ratios if r < 0.95)

    print("\n" + "-" * 42)
    print("CHẨN ĐOÁN ĐỘ DÀI  (t/r = độ dài sinh / độ dài tham chiếu)")
    print("-" * 42)
    print(f"  trung vị t/r          : {st.median(ratios):.2f}")
    print(f"  p10 / p90             : {ratios[int(n*0.1)]:.2f} / {ratios[int(n*0.9)]:.2f}")
    print(f"  số câu NGẮN hơn ref   : {short}/{n} ({100*short/n:.0f}%)")
    if st.median(ratios) < 0.95:
        print("  => ĐANG VIẾT QUÁ NGẮN. Đây là chỗ mất điểm rẻ nhất để sửa.")
        print("     Nhắm t/r khoảng 1.1–1.3 trước khi tối ưu bất cứ thứ gì khác.")
    elif st.median(ratios) > 1.6:
        print("  => Đang viết quá dài. Vẫn tốt hơn quá ngắn, nhưng nên siết lại.")
    else:
        print("  => Độ dài nằm trong vùng hợp lý.")

    # ---- Phân rã theo độ dài tham chiếu ----
    print("\n" + "-" * 42)
    print("PHÂN RÃ THEO ĐỘ DÀI THAM CHIẾU")
    print("-" * 42)
    buckets = [("ngắn  (<200 âm tiết)", 0, 200),
               ("vừa   (200–450)", 200, 450),
               ("dài   (>450)", 450, 10 ** 9)]
    for label, lo_b, hi_b in buckets:
        sel = [per[q]["meteor"] for q in per
               if lo_b <= per_plain[q]["len_ref"] < hi_b]
        if sel:
            print(f"  {label:<24} n={len(sel):>4}   METEOR={sum(sel)/len(sel):.4f}")

    # ---- Ca tệ nhất ----
    print("\n" + "-" * 42)
    print("5 CÂU ĐIỂM THẤP NHẤT  (nguyên liệu phân tích lỗi)")
    print("-" * 42)
    for qid, sc in sorted(per.items(), key=lambda kv: kv[1]["meteor"])[:5]:
        q = gold[qid]["question"][:56]
        print(f"  {qid}  METEOR={sc['meteor']:.3f}  "
              f"len {sc['len_pred']}/{sc['len_ref']}  | {q}...")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="tệp dự đoán JSON")
    ap.add_argument("--gold", default="data/dev_main.json",
                    help="data/dev_fast.json khi thí nghiệm nhanh")
    ap.add_argument("--name", default=None)
    ap.add_argument("--modes", default=DEFAULT_MODE,
                    help=f"'all' hoặc một trong {list(REF_MODES)}")
    ap.add_argument("--save", default=None, help="lưu điểm từng câu ra tệp JSON")
    ap.add_argument("--rouge", action="store_true",
                    help="tính thêm ROUGE-L (chậm, và đã hỏng với tiếng Việt "
                         "— chỉ bật khi cần số cho bài báo)")
    args = ap.parse_args()

    gold = load_json(args.gold)
    preds = normalize_preds(load_json(args.pred))
    modes = list(REF_MODES) if args.modes == "all" else [args.modes]

    results = report(gold, preds, args.name or Path(args.pred).stem, modes, args.rouge)

    if args.save:
        out = {m: {"agg": r[0], "per_sample": r[1]} for m, r in results.items()}
        Path(args.save).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nĐã lưu điểm từng câu vào {args.save}")


if __name__ == "__main__":
    main()
