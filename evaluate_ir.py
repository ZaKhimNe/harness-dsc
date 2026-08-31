"""
evaluate_ir.py — Chấm một tệp dự đoán Task 1 trên tập dev đã đóng băng.

Điểm số chỉ là hai dòng. Phần có giá trị hơn là CHẨN ĐOÁN bên dưới:

  - PRECISION và RECALL cạnh nhau, luôn luôn. Vì chưa biết BTC lấy cột nào,
    đọc một cột là tự bịt mắt một nửa.

  - Số câu VÔ HIỆU (k = 0 hoặc k > 5 tính trên danh sách thô). Mỗi câu như vậy
    là 0 điểm cứng, không phải điểm thấp. Đây là chỗ mất điểm ngu ngốc nhất.

  - Số câu có id TRÙNG LẶP. Trùng lặp bị phạt hai lần: phình mẫu số precision,
    và có thể đẩy len() thô vượt trần 5.

  - Phân rã theo SỐ GOLD của câu (1 gold vs nhiều gold). Câu nhiều gold là chỗ
    recall khó lên, và là lý do oracle_1 không đạt recall 1,0.

  - Đường cong recall/precision theo k. Trả lời thẳng câu hỏi chiến thuật:
    nộp mấy id thì hơn.

Cách dùng:
    python evaluate_ir.py --pred preds_ir/oracle_5.json
    python evaluate_ir.py --pred preds_ir/bm25.json --gold data_ir/dev_fast.json
"""

import argparse
import json
import random
import statistics as st
from pathlib import Path

from ir_metrics import DEFAULT_MODE, MAX_IDS, PRIMARY, REF_MODES, score_all, _as_list


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", default="data_ir/dev_main.json")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    gold = load_json(args.gold)
    preds = load_json(args.pred)
    name = Path(args.pred).stem

    agg, per = score_all(gold, preds, DEFAULT_MODE)
    n = agg["n"]

    print("=" * 70)
    print(f"CẤU HÌNH: {name}   (Task 1 — truy hồi)")
    print("=" * 70)
    print()
    print(f"{'Cách tổng hợp':<16}{'điểm':>10}")
    print("-" * 26)
    for m in REF_MODES:
        a, _ = score_all(gold, preds, m)
        star = "  <- mặc định" if m == DEFAULT_MODE else ""
        print(f"{m:<16}{a[PRIMARY]:>10.4f}{star}")
    print()
    lo, hi = ci_mean([v["recall"] for v in per.values()], args.boot)
    print(f"Recall    {agg['recall']:.4f}   KTC 95% [{lo:.4f}, {hi:.4f}]")
    lo, hi = ci_mean([v["precision"] for v in per.values()], args.boot)
    print(f"Precision {agg['precision']:.4f}   KTC 95% [{lo:.4f}, {hi:.4f}]")
    print(f"Số mẫu: {n}")

    print()
    print("-" * 42)
    print("CÂU HỎNG CỨNG  (0 điểm, không phải điểm thấp)")
    print("-" * 42)
    n_empty = sum(1 for v in per.values() if v["k"] == 0)
    n_over = sum(1 for v in per.values() if v["k"] > MAX_IDS)
    print(f"  k = 0 (rỗng)          : {n_empty:5d}  ({n_empty/n:.1%})")
    print(f"  k > {MAX_IDS} (vượt trần)    : {n_over:5d}  ({n_over/n:.1%})")
    print(f"  có id TRÙNG LẶP       : {agg['n_dup']:5d}  ({agg['n_dup']/n:.1%})")
    if agg["n_dup"]:
        print("  ⚠  Trùng lặp phạt HAI lần: phình mẫu số precision, và có thể")
        print("     đẩy len() thô vượt trần 5. Chạy guard.py --autofix.")
    print(f"  k trung bình          : {agg['k_mean']:.2f}")

    print()
    print("-" * 42)
    print("PHÂN RÃ THEO SỐ GOLD CỦA CÂU")
    print("-" * 42)
    groups = {}
    for qid, v in per.items():
        groups.setdefault("1 gold" if v["n_true"] == 1 else "nhiều gold", []).append(v)
    for g, rows in sorted(groups.items()):
        print(f"  {g:<12} n={len(rows):5d}   recall={st.mean(r['recall'] for r in rows):.4f}"
              f"   precision={st.mean(r['precision'] for r in rows):.4f}")

    print()
    print("-" * 42)
    print("ĐƯỜNG CONG THEO k  (cắt bài nộp hiện tại về k id đầu)")
    print("-" * 42)
    print(f"  {'k':>2}  {'recall':>8}  {'precision':>10}  {'trung bình':>11}")
    for k in range(1, MAX_IDS + 1):
        cut = {q: {"answer": _as_list(p)[:k]} for q, p in preds.items()}
        a, _ = score_all(gold, cut, "recall")
        print(f"  {k:>2}  {a['recall']:>8.4f}  {a['precision']:>10.4f}"
              f"  {(a['recall']+a['precision'])/2:>11.4f}")
    print()
    print("  Đọc bảng này TRƯỚC khi chọn k nộp. Nếu LB chấm recall -> lấy k=5.")
    print("  Nếu chấm precision -> k=1. Chưa biết -> đo trên public LB, submit")
    print("  không giới hạn nên miễn phí.")

    print()
    print("-" * 42)
    print("5 CÂU TRƯỢT HẲN  (recall = 0, có nộp id)")
    print("-" * 42)
    miss = [(q, v) for q, v in per.items() if v["recall"] == 0 and v["k"] > 0]
    for q, v in miss[:5]:
        g = gold[q]
        qt = g.get("question", "") if isinstance(g, dict) else ""
        print(f"  {q}  k={v['k']} n_true={v['n_true']}  | {qt[:56]}")
    if not miss:
        print("  (không có)")


if __name__ == "__main__":
    main()
