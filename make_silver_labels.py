"""
make_silver_labels.py — Dựng NHÃN BẠC văn bản nguồn cho Task 2.

`train.json` của Task 2 chỉ có `question` + `answer`, KHÔNG có nhãn văn bản
nguồn. Nhưng gold answer trích NGUYÊN VĂN điều luật -> khớp ngược chuỗi về
corpus là lấy lại được nhãn đó.

Đo trên dev_fast: khớp được 291/300 câu = 97,0%.

Dùng để làm gì
--------------
1. Sinh tệp --retrieved oracle cho `baseline_template.py`, để đo TRẦN của
   kiến trúc template khi tầng ① hoàn hảo (tách bạch lỗi truy hồi khỏi lỗi
   định vị + template).
2. 7.000 cặp (câu hỏi -> văn bản) làm dữ liệu huấn luyện/đánh giá cho retriever
   của Task 2 — trước đây tưởng là không có.
3. Ước lượng trần recall cho nhóm IR trên chính phân bố câu hỏi của Task 2.

Cách làm: shingle 8 âm tiết, một lượt duy nhất qua 8.532 văn bản. Ngưỡng 3
shingle trùng — đủ chặt để không dính nhầm, đủ lỏng để chịu được sai khác
khoảng trắng.

    python make_silver_labels.py --gold data/dev_fast.json --out retrieved_oracle.json
    python make_silver_labels.py --gold ../../drive-download-20260806T033107Z-1-001/train.json \
                                 --out silver_train.json
"""

import argparse
import collections
import json
import zipfile
from pathlib import Path

DEFAULT_CORPUS = "../../drive-download-20260806T033107Z-1-001/selected-contexts.zip"


def shingles(tokens, k=8, step=1):
    return {hash(tuple(tokens[i:i + k])) for i in range(0, len(tokens) - k + 1, step)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="tệp {qid: {question, answer}}")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-hits", type=int, default=3)
    args = ap.parse_args()

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))

    # shingle của mọi gold answer -> qid. step=3 để tiết kiệm bộ nhớ.
    index = {}
    for qid, v in gold.items():
        ans = v.get("answer")
        if not isinstance(ans, str):
            continue                      # Task 1 có answer là list -> bỏ qua
        for s in shingles(ans.split(), 8, 3):
            index.setdefault(s, []).append(qid)
    print(f"{len(gold)} câu, {len(index)} shingle truy vấn")

    z = zipfile.ZipFile(args.corpus)
    names = [x for x in z.namelist() if x.endswith(".json")]
    best = collections.defaultdict(lambda: (0, None))

    for i, nm in enumerate(names):
        try:
            doc = json.loads(z.read(nm))
        except Exception:  # noqa: BLE001
            continue
        did = str(doc.get("id") or nm.split("_")[-1][:-5])
        toks = doc.get("passage", "").split()
        hits = collections.Counter()
        for j in range(0, len(toks) - 7, 2):
            q = index.get(hash(tuple(toks[j:j + 8])))
            if q:
                for x in q:
                    hits[x] += 1
        for qid, c in hits.items():
            if c > best[qid][0]:
                best[qid] = (c, did)
        if i % 2000 == 0:
            print(f"  {i}/{len(names)}  khớp {len(best)}", flush=True)

    out = {q: [v[1]] for q, v in best.items() if v[1] and v[0] >= args.min_hits}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    n = sum(1 for v in gold.values() if isinstance(v.get("answer"), str))
    print(f"\nđã ghi {args.out}  —  khớp {len(out)}/{n} ({len(out)/max(n,1):.1%})")


if __name__ == "__main__":
    main()
