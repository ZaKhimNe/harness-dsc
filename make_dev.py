"""
make_dev.py — Tách và ĐÓNG BĂNG tập dev nội bộ từ `train.json` chính thức.

Chạy MỘT LẦN duy nhất, rồi không đụng tới nữa. Tập dev là toà án của cả nhóm:
mỗi người dùng một tập khác nhau thì các con số không so được, và cơ chế
bake-off mất nghĩa.

TẬP DEV HAI TẦNG — vì sao
-------------------------
Đo trên 2000 câu train thật, so hai KIẾN TRÚC khác nhau (trượt ở những câu
khác nhau, tương quan ≈ 0), độ lệch chuẩn của chênh lệch từng câu là ~0.50.
Từ đó suy ra xác suất chọn đúng cấu hình thắng:

    n dev  | chênh thật 0.01 | 0.02 | 0.04
    -------|-----------------|------|------
      150  |       60%       | 69%  | 84%
      300  |       64%       | 76%  | 92%
     1000  |       74%       | 90%  | 99%

Nhưng khi so CÙNG một hệ thống chỉ đổi một nút (dò t/r, sửa prompt), hai bản
tương quan rất cao, sd của chênh lệch chỉ ~0.037 — nhỏ hơn 13 lần — và 300 câu
đã quá đủ.

    dev_fast (300)  — thí nghiệm hàng ngày, chỉnh nút, sửa prompt
    dev_main (1000) — điểm quyết định: bake-off 08/09, chốt kiến trúc

`dev_fast` là TẬP CON LỒNG trong `dev_main`, nên hai số luôn so được với nhau
và không tốn thêm câu nào khỏi tập SFT.

Chi phí thật của tập dev lớn KHÔNG nằm ở lúc chấm (1000 câu mất dưới 1 giây)
mà ở lúc SINH câu trả lời bằng generator. Đó là lý do có tầng nhanh.

Sinh ra bốn tệp:
    dev_fast.json   300 câu   — tập con của dev_main
    dev_main.json  1000 câu   — toà án
    sft.json       6000 câu   — huấn luyện, KHÔNG được đụng vào dev
    dev.lock                  — vân tay SHA-256 + kiểm tra tính lồng nhau

Cách dùng:
    python make_dev.py --input ../../drive-download-20260806T033107Z-1-001/train.json
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

DEFAULT_INPUT = "../../drive-download-20260806T033107Z-1-001/train.json"


def sha256_of(obj):
    """Vân tay nội dung, không phụ thuộc thứ tự khoá."""
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT,
                    help="train.json chính thức của Task 2 (7000 câu)")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--n-main", type=int, default=1000, help="cỡ dev_main")
    ap.add_argument("--n-fast", type=int, default=300, help="cỡ dev_fast (tập con)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true",
                    help="ghi đè tập dev đã đóng băng — CÂN NHẮC KỸ")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    lock_path = outdir / "dev.lock"
    if lock_path.exists() and not args.force:
        raise SystemExit(
            f"{lock_path} đã tồn tại — tập dev ĐÃ ĐÓNG BĂNG.\n"
            "Sinh lại sẽ làm mọi con số đã đo trước đó không so được nữa.\n"
            "Nếu thật sự cần, thêm --force và báo cho cả nhóm."
        )

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"Không thấy {src}. Trỏ --input tới train.json của Task 2.")

    data = json.loads(src.read_text(encoding="utf-8"))
    if args.n_fast > args.n_main:
        raise SystemExit("--n-fast phải nhỏ hơn hoặc bằng --n-main.")
    if args.n_main > len(data):
        raise SystemExit(f"Chỉ có {len(data)} câu, không tách được {args.n_main}.")

    # sorted() trước khi shuffle -> kết quả không phụ thuộc thứ tự khoá trong tệp
    qids = sorted(data.keys())
    random.Random(args.seed).shuffle(qids)

    main_ids = qids[: args.n_main]
    fast_ids = main_ids[: args.n_fast]          # LỒNG trong dev_main
    sft_ids = qids[args.n_main:]

    dev_main = {k: data[k] for k in sorted(main_ids)}
    dev_fast = {k: data[k] for k in sorted(fast_ids)}
    sft = {k: data[k] for k in sorted(sft_ids)}

    # Bất biến phải luôn đúng, kiểm ngay tại đây
    assert set(dev_fast) < set(dev_main), "dev_fast phải là tập con thực sự của dev_main"
    assert not (set(dev_main) & set(sft)), "dev_main và sft KHÔNG được giao nhau"

    outdir.mkdir(parents=True, exist_ok=True)
    write(outdir / "dev_fast.json", dev_fast)
    write(outdir / "dev_main.json", dev_main)
    write(outdir / "sft.json", sft)

    lock = {
        "source": str(src),
        "seed": args.seed,
        "n_dev_fast": len(dev_fast),
        "n_dev_main": len(dev_main),
        "n_sft": len(sft),
        "sha256_dev_fast": sha256_of(dev_fast),
        "sha256_dev_main": sha256_of(dev_main),
        "sha256_source": sha256_of(data),
        "fast_is_subset_of_main": True,
    }
    write(lock_path, lock)

    print(f"dev_fast.json : {len(dev_fast):>5} câu   {lock['sha256_dev_fast'][:16]}...")
    print(f"dev_main.json : {len(dev_main):>5} câu   {lock['sha256_dev_main'][:16]}...")
    print(f"sft.json      : {len(sft):>5} câu")
    print(f"dev.lock      : đã ghi")
    print()
    print("dev_fast là TẬP CON của dev_main -> hai số luôn so được với nhau.")
    print("Commit cả bốn tệp vào git và KHÔNG sửa nữa.")
    print()
    print("Tiếp theo:  python baselines.py && python compare.py --preds 'preds/*.json'")


if __name__ == "__main__":
    main()
