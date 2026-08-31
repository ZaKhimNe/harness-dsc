"""
meteor_ref.py — Bản METEOR thuần Python, KHÔNG phụ thuộc nltk.

Vì sao cần: `btc_metrics.py` luôn ưu tiên `nltk` để khớp tuyệt đối với BTC.
Nhưng máy không có mạng (không tải được `wordnet`) thì cả harness đứng im.
Module này là bản dự phòng để harness vẫn chạy được ở mọi nơi.

SAO CHÉP THUẬT TOÁN của `nltk.translate.meteor_score` với tham số mặc định
(alpha=0.9, beta=3, gamma=0.5), gồm ba phần:
  - `_match_enums`  : ghép cặp token khớp CHÍNH XÁC, duyệt ngược từ cuối
  - `_count_chunks` : đếm số cụm liên tiếp ở cả hai phía
  - công thức F_mean và hình phạt phân mảnh

MỘT KHÁC BIỆT CÓ CHỦ Ý — đọc kỹ:
    nltk chạy ba tầng ghép cặp: khớp chính xác -> gốc từ (Porter) -> đồng nghĩa
    (WordNet). Bản này CHỈ có tầng đầu.

    Với tiếng Việt điều đó gần như không đổi kết quả: WordNet tiếng Anh không
    có synset cho từ tiếng Việt (tầng 3 chết hẳn), còn Porter hầu như không
    đụng tới âm tiết tiếng Việt. Nhưng "gần như" không phải "chắc chắn" —
    Porter có luật y->i nên `hay`/`hai` có thể ghép được ở tầng 2.

    ĐỪNG TIN, HÃY ĐO: chạy `python selftest.py` trên máy có nltk. Nó đối chiếu
    hai bản trên dữ liệu thật và in ra sai lệch tối đa. Chỉ dùng bản dự phòng
    để ra quyết định khi selftest báo sai lệch đủ nhỏ.
"""

ALPHA, BETA, GAMMA = 0.9, 3, 0.5


def _match_enums(enum_hyp, enum_ref):
    """Ghép cặp token khớp chính xác. Sao nguyên `nltk...meteor_score._match_enums`.

    Duyệt ngược từ cuối lên và pop khỏi cả hai danh sách — thứ tự duyệt này
    ảnh hưởng tới cách ghép khi có token lặp, nên phải giữ đúng.
    """
    word_match = []
    for i in range(len(enum_hyp) - 1, -1, -1):
        for j in range(len(enum_ref) - 1, -1, -1):
            if enum_hyp[i][1] == enum_ref[j][1]:
                word_match.append((enum_hyp[i][0], enum_ref[j][0]))
                enum_hyp.pop(i)
                enum_ref.pop(j)
                break
    return word_match, enum_hyp, enum_ref


def _count_chunks(matches):
    """Đếm số cụm. Một cụm = dãy token liên tiếp ở CẢ HAI câu, cùng thứ tự."""
    if not matches:
        return 0
    i, chunks = 0, 1
    while i < len(matches) - 1:
        if matches[i + 1][0] == matches[i][0] + 1 and matches[i + 1][1] == matches[i][1] + 1:
            i += 1
            continue
        i += 1
        chunks += 1
    return chunks


def single_meteor_score(reference_tokens, hypothesis_tokens,
                        alpha=ALPHA, beta=BETA, gamma=GAMMA):
    """METEOR cho một cặp. Đầu vào là danh sách token đã tách sẵn.

    nltk mặc định `preprocess=str.lower`, nên ta cũng hạ chữ thường.
    """
    enum_hyp = list(enumerate(w.lower() for w in hypothesis_tokens))
    enum_ref = list(enumerate(w.lower() for w in reference_tokens))

    matches, _, _ = _match_enums(enum_hyp, enum_ref)
    matches.sort(key=lambda pair: pair[0])
    m = len(matches)

    # nltk bắt ZeroDivisionError và trả 0.0
    if m == 0 or not hypothesis_tokens or not reference_tokens:
        return 0.0

    precision = m / len(hypothesis_tokens)
    recall = m / len(reference_tokens)
    fmean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
    penalty = gamma * (_count_chunks(matches) / m) ** beta
    return (1 - penalty) * fmean


def meteor_score(references_tokens, hypothesis_tokens, **kw):
    """Nhiều tham chiếu -> lấy điểm cao nhất. Giống chữ ký của nltk."""
    return max(single_meteor_score(r, hypothesis_tokens, **kw)
               for r in references_tokens)


# ---------------------------------------------------------------------------
# ROUGE-L dự phòng — CHỈ dùng khi thiếu gói `rouge_score` của BTC.
# Cảnh báo: ROUGE-L đã hỏng với tiếng Việt (tokenizer xoá mọi ký tự ngoài
# [a-z0-9]). Giữ lại để có con số, KHÔNG dùng để ra quyết định.
# ---------------------------------------------------------------------------
import re

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _rouge_tokenize(text):
    """Sao đúng `rouge_score.tokenize`: hạ chữ thường, xoá mọi ký tự ngoài a-z0-9."""
    return [t for t in _NON_ALNUM.sub(" ", text.lower()).split() if t]


def _lcs_len(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l_fmeasure(reference_text, hypothesis_text):
    a, b = _rouge_tokenize(reference_text), _rouge_tokenize(hypothesis_text)
    if not a or not b:
        return 0.0
    lcs = _lcs_len(a, b)
    if lcs == 0:
        return 0.0
    p, r = lcs / len(b), lcs / len(a)
    return 2 * p * r / (p + r)
