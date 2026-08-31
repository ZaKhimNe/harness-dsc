"""
baseline_template.py — SÀN của Task 2: truy hồi + định vị điều luật + template cứng.

Đây là baseline để so mọi thứ khác. Không có mô hình sinh, không có GPU, không
có tham số học được. Nếu một hệ thống LLM không vượt được tệp này thì nó không
đáng tồn tại.

BA TẦNG — và tầng ở giữa mới là chỗ điểm số nằm
-----------------------------------------------
    ① TRUY HỒI VĂN BẢN   8.532 tệp -> 1..k văn bản      (đọc từ --retrieved)
    ② ĐỊNH VỊ ĐIỀU LUẬT  1 văn bản = trung vị 4.945 âm tiết, 13 Điều
                         gold answer =            347 âm tiết
                         -> dán cả văn bản là dài gấp ~14 lần. ĐIỂM SẬP.
    ③ GHÉP TEMPLATE      khung mở + nguyên văn Điều + đoạn kết

Tầng ② có ba cách cắm được, chọn bằng --locator, rồi để compare.py phân xử:

    article   tách theo "Điều N.", chấm lại từng Điều với câu hỏi   (mặc định)
    window    cửa sổ trượt ~350 âm tiết, không dựa vào cấu trúc
    whole     dán cả văn bản — ĐỐI CHỨNG ÂM, dùng để chứng minh tầng ② cần thiết

VÌ SAO TEMPLATE CÓ DẠNG NÀY — số đo trên dev_fast (300 câu)
------------------------------------------------------------
    A  chỉ nguyên văn điều luật, không khung        0,7344
    B1 + khung CHUNG (không cần cite thật)          0,7970   <- +0,063 MIỄN PHÍ
    B2 + khung có cite đúng                         0,8215   <- cite chỉ thêm +0,025
    C2 + đoạn kết                                   0,8378
    E  gold nguyên vẹn (TRẦN)                       1,0000

Khung không phải trang trí — nó là 0,104 điểm. Vì gold answer LUÔN có khung:
57,6% mở bằng "Căn cứ", 25,1% bằng "Theo", 85,8% chứa cụm "như sau:".

ĐƯỜNG CONG ĐỘ DÀI — vì sao có LEN_TARGET
-----------------------------------------
    cắt còn 80%   -> 0,7722        đủ + đệm 50%  -> 0,9189
    cắt còn 70%   -> 0,6683        đủ + đệm 150% -> 0,7804
    cắt còn 50%   -> 0,4751        đủ + đệm 300% -> 0,6309

Cắt 20% ≈ phình 2,5 lần. Một đơn vị THIẾU đắt bằng ~7,5 đơn vị THỪA
(hệ quả trực tiếp của alpha = 0,9 trong METEOR). Nên khi phân vân:
THÀ DÀI. Không bao giờ dừng ở 150 âm tiết.

ĐỊNH DẠNG --retrieved (chấp nhận cả ba, tự nhận diện)
------------------------------------------------------
    {"10001": ["280282", "56081"]}                     <- gọn nhất
    {"10001": {"answer": ["280282"]}}                  <- giống train.json Task 1
    {"10001": [{"id": "280282", "score": 12.3}]}       <- có điểm, dùng luôn thứ tự

Cách dùng
---------
    python baseline_template.py --retrieved retrieved_dev.json --out preds/tpl_article.json
    python baseline_template.py --retrieved retrieved_dev.json --locator window --out preds/tpl_window.json
    python baseline_template.py --retrieved retrieved_dev.json --locator whole  --out preds/tpl_whole.json
    python compare.py --preds 'preds/tpl_*.json' 'preds/oracle_copy.json' 'preds/template_only.json'
"""

import argparse
import json
import math
import re
import statistics as st
import zipfile
from collections import Counter
from pathlib import Path

# --------------------------------------------------------------------------
# Hằng số suy ra từ đo đạc, KHÔNG phải đoán. Xem docstring.
# --------------------------------------------------------------------------
LEN_FLOOR = 250    # dưới mức này gần như chắc chắn đang thiếu -> phạt nặng
LEN_TARGET = 480   # DÒ RA TRÊN dev_fast, không phải đoán. Xem bảng dưới
LEN_CAP = 1200     # trên mức này phần đệm bắt đầu đắt hơn phần được

# Dò LEN_TARGET trên dev_fast, truy hồi oracle (plain / độ dài trung vị sinh ra):
#   target   250     300     350     400    *480*    600
#   article  .4712   .4965   .5050   .5110  .5144   .5122   <- đỉnh phẳng 400–600
#   window   .4488   .4643   .4815   .4867  .4926   .5009
#
# NGHỊCH LÝ ĐÁNG NHỚ: độ dài tối ưu (~620 âm tiết) GẤP ĐÔI trung vị gold (305).
# Đường cong đệm ở docstring nói 2x mất 0,16 — nhưng đường cong đó giả định nội
# dung ĐÚNG TUYỆT ĐỐI. Ở đây nội dung là văn bản truy hồi, chưa chắc đúng, nên
# recall mới là ràng buộc: viết thêm còn rẻ hơn viết thiếu.
#   -> Nội dung càng kém tin cậy, càng nên viết dài.
#   -> Khi thay bằng LLM sinh câu trả lời tốt, PHẢI dò lại LEN_TARGET, nó sẽ giảm.

DEFAULT_CORPUS = "../../drive-download-20260806T033107Z-1-001/selected-contexts.zip"
DEFAULT_GOLD = "data/dev_main.json"

# Tên tệp corpus -> tên loại văn bản đọc được
DOC_KIND = [
    ("Nghi-dinh", "Nghị định"), ("Nghi-quyet", "Nghị quyết"),
    ("Thong-tu-lien-tich", "Thông tư liên tịch"), ("Thong-tu", "Thông tư"),
    ("Quyet-dinh", "Quyết định"), ("Phap-lenh", "Pháp lệnh"),
    ("Cong-van", "Công văn"), ("Chi-thi", "Chỉ thị"), ("Luat", "Luật"),
    ("Bo-luat", "Bộ luật"), ("Hien-phap", "Hiến pháp"),
]

# Từ hỏi và từ đệm — không mang thông tin định vị, bỏ khi chấm chunk
STOP = set("""là gì của và các có được cho về theo tại trong khi nào bao nhiêu thế
nào ra sao những một người này đó với thì mà hay hoặc như đối khoản điều mục
chương phần cần phải không tôi hỏi xin cho biết ạ vậy ai đâu""".split())


# --------------------------------------------------------------------------
# Tách âm tiết — PHẢI khớp cách BTC chấm: str.split(), không tách từ
# --------------------------------------------------------------------------
def syl(text):
    return text.split()


def norm_tokens(text):
    """Chuẩn hoá để CHẤM chunk (không dùng để sinh câu trả lời)."""
    t = re.sub(r"[^\w\s/]", " ", text.lower())
    return [w for w in t.split() if w not in STOP and len(w) > 1]


# --------------------------------------------------------------------------
# ① Đọc kết quả truy hồi — tự nhận diện ba định dạng
# --------------------------------------------------------------------------
def load_retrieved(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for qid, v in raw.items():
        if isinstance(v, dict):
            v = v.get("answer") or v.get("ids") or v.get("contexts") or []
        if not isinstance(v, list):
            v = [v]
        ids = []
        for item in v:
            if isinstance(item, dict):
                item = item.get("id") or item.get("context_id")
            if item is not None:
                ids.append(str(item))
        out[qid] = ids
    return out


class Corpus:
    """Đọc thẳng từ .zip. 93 MB, giải nén ra đĩa vừa chậm vừa vô ích."""

    def __init__(self, zip_path):
        self.z = zipfile.ZipFile(zip_path)
        self.names = set(self.z.namelist())
        self.cache = {}

    def get(self, doc_id):
        if doc_id in self.cache:
            return self.cache[doc_id]
        name = f"selected-contexts/context_{doc_id}.json"
        doc = None
        if name in self.names:
            try:
                doc = json.loads(self.z.read(name))
            except Exception:  # noqa: BLE001
                doc = None
        self.cache[doc_id] = doc
        return doc


# --------------------------------------------------------------------------
# ② Định vị — ba chiến thuật cắm được
# --------------------------------------------------------------------------
ART_RE = re.compile(r"(?m)^[ \t]*(Điều\s+\d+[a-zA-Z]?\s*[.:])")


def split_articles(passage):
    """Tách văn bản thành từng Điều. Trả [(nhãn, nội dung)]."""
    parts = ART_RE.split(passage)
    if len(parts) < 3:
        return []
    out = []
    for i in range(1, len(parts), 2):
        head, body = parts[i].strip(), parts[i + 1]
        chunk = (head + " " + body).strip()
        if len(syl(chunk)) >= 25:          # bỏ mục lục, tiêu đề rỗng
            out.append((head, chunk))
    return out


def split_windows(passage, width=None, stride=None):
    width = width or LEN_TARGET
    stride = stride or max(1, width // 2)
    toks = syl(passage)
    out = []
    for i in range(0, max(1, len(toks) - width // 2), stride):
        w = toks[i:i + width]
        if len(w) >= 60:
            out.append(("", " ".join(w)))
    return out or [("", passage)]


def fit_chunk(question, text, budget):
    """Một Điều đơn lẻ có thể dài 5.000 âm tiết trong khi gold chỉ 350.

    Khi đó KHÔNG dán cả Điều — cắt cửa sổ bên trong nó, chọn phần khớp câu hỏi
    nhất, rồi ghép lại THEO ĐÚNG THỨ TỰ GỐC (giữ tính liền mạch cho METEOR).
    """
    toks = syl(text)
    if len(toks) <= budget * 1.6:
        return text
    w = max(120, budget // 2)
    wins = []
    for i in range(0, max(1, len(toks) - w // 2), w):
        seg = toks[i:i + w]
        if len(seg) >= 40:
            wins.append((i, " ".join(seg)))
    ranked = rank_chunks(question, [("", t) for _, t in wins])
    order = {t: s for s, _, t in ranked}
    keep, total = [], 0
    for i, t in sorted(wins, key=lambda x: -order.get(x[1], 0)):
        if total >= budget:
            break
        keep.append((i, t))
        total += len(syl(t))
    keep.sort(key=lambda x: x[0])
    return "\n".join(t for _, t in keep)


def rank_chunks(question, chunks):
    """BM25-lite, IDF dựng tại chỗ trên các chunk ứng viên của CHÍNH câu này.

    Không cần index toàn cục — số chunk mỗi câu chỉ vài chục.
    """
    if not chunks:
        return []
    q = set(norm_tokens(question))
    docs = [Counter(norm_tokens(c[1])) for c in chunks]
    N = len(docs)
    df = Counter()
    for d in docs:
        for w in d:
            df[w] += 1
    avgdl = st.mean(sum(d.values()) for d in docs) or 1.0
    k1, b = 1.5, 0.75
    scored = []
    for (label, text), d in zip(chunks, docs):
        dl = sum(d.values()) or 1
        s = 0.0
        for w in q:
            if w not in d:
                continue
            idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
            tf = d[w]
            s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
        scored.append((s, label, text))
    scored.sort(key=lambda x: -x[0])
    return scored


# --------------------------------------------------------------------------
# ③ Template
# --------------------------------------------------------------------------
CLEAN_Q = re.compile(r"^(cho\s+tôi\s+hỏi|xin\s+hỏi|tôi\s+muốn\s+hỏi)[,\s]*", re.I)


def topic_of(question):
    q = CLEAN_Q.sub("", question.strip()).strip()
    return q.rstrip("?").strip()


def cite_of(doc, label):
    """Dựng chuỗi trích dẫn. Sai thì rơi về khung chung — chỉ mất 0,025."""
    if not doc:
        return "Căn cứ quy định của pháp luật hiện hành"
    name = doc.get("name") or ""
    kind = next((v for k, v in DOC_KIND if k.lower() in name.lower()), None)
    num = None
    m = re.search(r"(?m)^\s*Số:?\s*([\w./\-]+/[\w.\-]+)", doc.get("passage", "")[:3000])
    if m:
        num = m.group(1).strip()
    else:
        m = re.search(r"(\d+[a-zA-Z]?/\d{4}/[A-ZĐ\-]+)", name)
        if m:
            num = m.group(1)
    art = label.rstrip(".:").strip() if label else ""
    head = "Căn cứ"
    if art:
        head += f" {art}"
    if kind and num:
        head += f" {kind} {num}"
    elif kind:
        head += f" {kind}"
    elif not art:
        head += " quy định của pháp luật hiện hành"
    return head


def assemble(question, cite, bodies):
    topic = topic_of(question)
    parts = [f"{cite} quy định về {topic} như sau:"]
    parts += bodies
    parts.append(f"Theo đó, {topic} được thực hiện theo quy định nêu trên.")
    parts.append(
        f"Như vậy, căn cứ quy định của pháp luật hiện hành thì {topic.lower()} "
        "theo đúng nội dung đã được trích dẫn ở trên."
    )
    return "\n".join(parts)


def fallback(question):
    """T0 — không truy hồi được gì. Sàn template_only ≈ 0,083, vẫn hơn crash."""
    topic = topic_of(question)
    return assemble(question, "Căn cứ quy định của pháp luật hiện hành", [
        f"Pháp luật hiện hành có quy định cụ thể về {topic.lower()}.",
        "Việc thực hiện phải tuân thủ đầy đủ các điều kiện, trình tự, thủ tục, "
        "thẩm quyền và thời hạn do pháp luật quy định.",
        "Trường hợp có quy định khác nhau về cùng một vấn đề thì áp dụng văn bản "
        "có hiệu lực pháp lý cao hơn; trường hợp các văn bản do cùng một cơ quan "
        "ban hành thì áp dụng văn bản được ban hành sau.",
        "Cơ quan, tổ chức, cá nhân có liên quan có trách nhiệm thực hiện đúng quy "
        "định; nếu vi phạm thì tuỳ theo tính chất, mức độ mà bị xử lý theo quy định "
        "của pháp luật.",
    ])


# --------------------------------------------------------------------------
def build_one(question, doc_ids, corpus, locator):
    docs = [corpus.get(d) for d in doc_ids]
    docs = [d for d in docs if d and d.get("passage")]
    if not docs:
        return fallback(question), "T0_khong_truy_hoi"

    if locator == "whole":
        bodies = [d["passage"].strip() for d in docs[:1]]
        return assemble(question, cite_of(docs[0], ""), bodies), "whole"

    budget = LEN_TARGET

    cand = []
    for d in docs:
        chunks = split_articles(d["passage"]) if locator == "article" else []
        if not chunks:
            chunks = split_windows(d["passage"])
        for s, label, text in rank_chunks(question, chunks):
            cand.append((s, label, text, d))
    if not cand:
        return fallback(question), "T0_khong_tach_duoc"
    cand.sort(key=lambda x: -x[0])

    # Chính sách độ dài — kiểm TRƯỚC khi thêm, không phải sau.
    # Thêm sau khi đã đủ chính là lỗi làm độ dài vọt 2,3 lần ở bản đầu.
    bodies, total = [], 0
    for s, label, text, d in cand:
        if total >= budget:
            break
        piece = fit_chunk(question, text, budget - total)
        n = len(syl(piece))
        if bodies and total + n > LEN_CAP:
            break
        bodies.append(piece)
        total += n
    cite = cite_of(cand[0][3], cand[0][1])
    return assemble(question, cite, bodies), f"{locator}_x{len(bodies)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieved", required=True, help="kết quả tầng ① từ nhóm IR")
    ap.add_argument("--gold", default=DEFAULT_GOLD, help="tập dev — dùng để lấy DANH SÁCH qid")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--locator", choices=["article", "window", "whole"], default="article")
    ap.add_argument("--topk", type=int, default=3, help="số văn bản lấy từ tầng ①")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    retrieved = load_retrieved(args.retrieved)
    corpus = Corpus(args.corpus)

    preds, why, lens = {}, Counter(), []
    for qid, item in gold.items():
        ans, tag = build_one(item["question"], retrieved.get(qid, [])[: args.topk],
                             corpus, args.locator)
        preds[qid] = {"answer": ans}
        why[tag] += 1
        lens.append(len(syl(ans)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(preds, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- CHẨN ĐOÁN: quan trọng hơn bản thân tệp ----
    lens.sort()
    n = len(lens)
    ref = [len(syl(v["answer"])) for v in gold.values()]
    print(f"đã ghi {out}  —  {n} câu (đủ toàn bộ qid của {args.gold})")
    print()
    print("  đường đi:")
    for k, c in why.most_common():
        print(f"    {c:5d}  {c/n:5.1%}  {k}")
    print()
    print(f"  độ dài sinh ra : p10 {lens[n//10]}  trung vị {lens[n//2]}  p90 {lens[9*n//10]}")
    print(f"  độ dài gold    : p10 {sorted(ref)[n//10]}  trung vị {sorted(ref)[n//2]}"
          f"  p90 {sorted(ref)[9*n//10]}")
    short = sum(1 for a, b in zip(lens, sorted(ref)) if a < b * 0.8)
    print(f"  NGẮN hơn 80% gold: {short/n:.1%}   <- mỗi 20% thiếu mất ~0,22 điểm")
    if lens[n // 2] < LEN_FLOOR:
        print("  ⚠  trung vị dưới sàn 250 — đang mất điểm ở chỗ RẺ NHẤT để sửa.")
    print()
    print(f"Tiếp theo:  python evaluate.py --pred {out} --gold {args.gold}")


if __name__ == "__main__":
    main()
