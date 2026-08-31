"""
baselines_ir.py — Trần và sàn cho Task 1. Dựng TRƯỚC khi dựng retriever.

Không có hai mốc này thì một con số recall đơn lẻ không nói lên điều gì.
Đặc biệt với Task 1, nơi mấy baseline tầm thường lại mạnh bất ngờ — và mạnh
theo cách chỉ lộ ra khi so precision với recall cạnh nhau.

  oracle_1        Đúng 1 văn bản gold đầu tiên.
                  -> TRẦN của precision. Recall < 1 vì 9% câu có nhiều hơn 1 gold.

  oracle_5        Toàn bộ gold, cắt về 5.
                  -> TRẦN tuyệt đối cả hai cột.

  oracle_pad5     Gold thật + độn id rác cho đủ đúng 5.
                  -> ĐO GIÁ CỦA VIỆC NỘP THỪA. Recall giữ nguyên, precision sập.
                     Đây là baseline quan trọng nhất: nó lượng hoá chính xác
                     canh bạc "nộp 5 hay nộp 1".

  oracle_dup      Gold thật, nhưng lặp lại id tới 6 lần.
                  -> Chứng minh trùng lặp giết điểm: len() thô vượt trần 5.
                     Phải ra ĐÚNG 0. Nếu không, harness sai.

  random_1        Một id ngẫu nhiên.       -> SÀN.
  random_5        Năm id ngẫu nhiên.       -> SÀN của chiến thuật rải mành mành.

  empty           Danh sách rỗng.
                  -> Phải ra 0 (điều kiện len > 0 của BTC). Phép kiểm harness.

Cách dùng:
    python baselines_ir.py --gold data_ir/dev_main.json --outdir preds_ir
"""

import argparse
import json
import random
from pathlib import Path


def gold_ids(item):
    if isinstance(item, dict):
        item = item.get("answer", [])
    if not isinstance(item, list):
        item = [item]
    return [str(x) for x in item]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data_ir/dev_main.json")
    ap.add_argument("--outdir", default="preds_ir")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    rng = random.Random(args.seed)

    # Kho id để lấy mẫu rác: gom mọi id gold trong tập.
    pool = sorted({i for v in gold.values() for i in gold_ids(v)})

    def noise(exclude, n):
        out = []
        while len(out) < n:
            c = rng.choice(pool)
            if c not in exclude and c not in out:
                out.append(c)
        return out

    builders = {
        "oracle_1":    lambda g: g[:1],
        "oracle_5":    lambda g: g[:5],
        "oracle_pad5": lambda g: (g[:5] + noise(g, max(0, 5 - len(g[:5]))))[:5],
        "oracle_dup":  lambda g: (g[:1] * 6),
        "random_1":    lambda g: noise([], 1),
        "random_5":    lambda g: noise([], 5),
        "empty":       lambda g: [],
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, fn in builders.items():
        preds = {qid: {"answer": fn(gold_ids(item))} for qid, item in gold.items()}
        (outdir / f"{name}.json").write_text(
            json.dumps(preds, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {name:14s} -> {outdir / (name + '.json')}")

    n_multi = sum(1 for v in gold.values() if len(gold_ids(v)) > 1)
    print(f"\n{len(gold)} câu, {n_multi} câu ({n_multi/len(gold):.1%}) có NHIỀU HƠN 1 gold")
    print(f"trung bình {sum(len(gold_ids(v)) for v in gold.values())/len(gold):.2f} gold/câu")
    print(f"\nTiếp theo:  python compare.py --task 1 --preds '{outdir}/*.json' --gold {args.gold}")


if __name__ == "__main__":
    main()
