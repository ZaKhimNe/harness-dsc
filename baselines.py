"""
baselines.py — Sinh các bài nộp tầm thường để kiểm tra harness và lấy mốc so sánh.

Dựng TRẦN và SÀN trước khi dựng hệ thống. Không có hai mốc này thì một con số
METEOR đơn lẻ không nói lên điều gì.

  oracle_copy       Chép y hệt câu trả lời chuẩn.
                    -> TRẦN. Không phải 1.0 vì lỗi str() trong scoring.py BTC.

  oracle_copy_q     Chép y hệt, nhưng LẶP LẠI CÂU HỎI ở đầu.
                    -> Cặp đối chứng của thí nghiệm quyết định. Nếu tệp tham
                       chiếu của BTC chứa cả 'question' thì bản này ăn điểm hơn
                       oracle_copy. So hai bản này trên public LB là cách duy
                       nhất biết được sự thật.

  oracle_80         Chép đúng nhưng chỉ giữ 80% độ dài.
                    -> Đo giá của việc viết hơi ngắn.

  oracle_half       Chép đúng nhưng cắt còn một nửa.
                    -> Đo giá của việc viết quá ngắn. Bài học đắt nhất.

  oracle_pad        Chép đủ rồi nối thêm template rác cho dài gấp rưỡi.
                    -> Đối chứng bất đối xứng: thừa rẻ hơn thiếu bao nhiêu lần.

  template_only     Chỉ khung câu, không có nội dung luật.
                    -> Bao nhiêu điểm đến từ riêng văn phong.

  echo_question     Chỉ lặp lại câu hỏi.
                    -> SÀN tuyệt đối.

Cách dùng:
    python baselines.py --gold data/dev_main.json --outdir preds
"""

import argparse
import json
from pathlib import Path

TEMPLATE = (
    "Căn cứ theo quy định của pháp luật hiện hành về vấn đề này thì nội dung "
    "được quy định cụ thể như sau: Theo đó, các cơ quan, tổ chức, cá nhân có "
    "liên quan phải thực hiện đầy đủ các quy định của pháp luật. Như vậy, "
    "trường hợp này được xác định theo quy định tại các văn bản pháp luật "
    "có liên quan."
)


def _keep(answer, frac):
    t = answer.split()
    return " ".join(t[: max(1, int(len(t) * frac))])


def _pad(answer, ratio):
    t = answer.split()
    need = int(len(t) * (ratio - 1))
    filler = (TEMPLATE.split() * (need // len(TEMPLATE.split()) + 1))[:need]
    return " ".join(t + filler)


BUILDERS = {
    "oracle_copy":   lambda q, a: a,
    "oracle_copy_q": lambda q, a: f"{q}\n{a}",
    "oracle_80":     lambda q, a: _keep(a, 0.80),
    "oracle_half":   lambda q, a: _keep(a, 0.50),
    "oracle_pad":    lambda q, a: _pad(a, 1.50),
    "template_only": lambda q, a: TEMPLATE,
    "echo_question": lambda q, a: q,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/dev_main.json")
    ap.add_argument("--outdir", default="preds")
    ap.add_argument("--only", nargs="*", default=None,
                    help="chỉ sinh một số baseline nhất định")
    args = ap.parse_args()

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    names = args.only or list(BUILDERS)
    for name in names:
        fn = BUILDERS[name]
        preds = {
            qid: {"answer": fn(item["question"], item["answer"])}
            for qid, item in gold.items()
        }
        p = outdir / f"{name}.json"
        p.write_text(json.dumps(preds, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Đã tạo {p}  ({len(preds)} câu)")

    print(f"\nChấm cả bộ:  python compare.py --preds '{outdir}/*.json'")


if __name__ == "__main__":
    main()
