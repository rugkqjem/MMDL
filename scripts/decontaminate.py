"""Remove MMMU / MMMU-Pro overlap from candidate training data before fine-tuning.

Four stages, as in the proposal:
  1. metadata   : drop rows whose source / id / image path names MMMU (complements reading each dataset card)
  2. text       : 13-gram overlap (GPT-3 style, boilerplate n-grams ignored) + MinHash near-duplicates
  3. image      : perceptual hash (pHash) distance + CLIP embedding cosine similarity
  4. review     : every flagged row goes to review.html / decisions.csv. By default ALL flagged rows are removed
                  (conservative); reviewers only switch obvious false positives to `keep` and re-run with --apply.

Input  : qwen-vl-finetune JSONL (what train_ft.py / prepare_mmmu_sft.py use):
         {"id": ..., "image": "a.png" | ["a.png", ...], "conversations": [{"from": "human", "value": ...}, ...]}
         plus optional metadata fields (e.g. "source") checked in stage 1. Convert HF datasets to this format first.
         Rows that also carry "question" / "options" (rl_prepare_virl39k.py) are compared on question + options,
         the same text the MMMU reference is built from, so the eval prompt boilerplate does not dilute MinHash.
Reference: MMMU dev/validation/test (pinned revision from infer.py) + MMMU-Pro (standard 10-option + vision).

Usage (from the repo root):
  # build the MMMU reference once (cached), then check a candidate set
  python scripts/decontaminate.py --data data/pathvqa/train.jsonl --image_root data/pathvqa/images \\
      --out outputs/decontam/pathvqa
  # after editing decisions.csv (remove -> keep for false positives)
  python scripts/decontaminate.py --data data/pathvqa/train.jsonl --out outputs/decontam/pathvqa --apply
Outputs in --out: clean.jsonl, flags.jsonl, decisions.csv, review.html, report.md
"""
import argparse
import base64
import csv
import hashlib
import io
import json
import os
import pickle
import re
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MMMU_PAT = re.compile(r"mmmu", re.I)
IMG_TAG = re.compile(r"<image\s*\d*>", re.I)


# ----------------------------------------------------------------------------- text utils
def norm_text(t):
    t = IMG_TAG.sub(" ", str(t)).lower()
    t = re.sub(r"[^\w]+", " ", t, flags=re.UNICODE)
    return " ".join(t.split())


def ngrams(words, n):
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


def minhash(words, num_perm, shingle=3):
    from datasketch import MinHash
    m = MinHash(num_perm=num_perm)
    for g in set(ngrams(words, shingle)) or {" ".join(words)}:
        m.update(g.encode())
    return m


# ----------------------------------------------------------------------------- image utils
def phash64(img):
    import imagehash
    bits = imagehash.phash(img.convert("RGB")).hash.flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def thumb_b64(img, size=160):
    im = img.convert("RGB")
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode()


def popcount64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


class Embedder:
    def __init__(self, name, device=None):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(name).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(name)

    def __call__(self, imgs):
        with self.torch.no_grad():
            x = self.proc(images=[i.convert("RGB") for i in imgs], return_tensors="pt").to(self.device)
            e = self.model.get_image_features(**x)
            e = e / e.norm(dim=-1, keepdim=True)
        return e.float().cpu().numpy()


# ----------------------------------------------------------------------------- reference (MMMU + MMMU-Pro)
def iter_reference(args):
    """Yields (ref_id, text, [PIL images]). Also usable offline via --ref_jsonl in the candidate format."""
    import ast
    if args.ref_jsonl:
        for rid, text, imgs in iter_candidates(args.ref_jsonl, args.ref_image_root):
            yield rid, text, imgs
        return
    from datasets import load_dataset
    import infer
    for split in args.mmmu_splits:
        for subj in infer.SUBJECTS:
            ds = load_dataset("MMMU/MMMU", subj, split=split, revision=infer.DATASET_REVISION,
                              cache_dir=args.data_root)
            for r in ds:
                opts = ast.literal_eval(r["options"]) if r.get("options") else []
                text = " ".join([r["question"], *map(str, opts), r.get("explanation") or ""])
                imgs = [r[f"image_{i}"] for i in range(1, 8) if r.get(f"image_{i}") is not None]
                yield r["id"], text, imgs
    if args.skip_pro:
        return
    for cfg in ("standard (10 options)", "vision"):
        try:
            ds = load_dataset("MMMU/MMMU_Pro", cfg, split="test", cache_dir=args.data_root)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] MMMU-Pro '{cfg}' not loaded ({e}); continuing without it", file=sys.stderr)
            continue
        for r in ds:
            opts = r.get("options")
            opts = ast.literal_eval(opts) if isinstance(opts, str) and opts.startswith("[") else (opts or [])
            text = " ".join([r.get("question") or "", *map(str, opts)])
            imgs = [r[k] for k in ["image", *[f"image_{i}" for i in range(1, 8)]] if r.get(k) is not None]
            yield f"pro/{cfg.split()[0]}/{r['id']}", text, imgs


def ref_config(args, embedder):
    return {"splits": list(args.mmmu_splits), "pro": not args.skip_pro, "ngram": args.ngram,
            "ngram_max_df": args.ngram_max_df, "min_words": args.min_words, "num_perm": args.num_perm,
            "minhash_threshold": args.minhash_threshold, "embed": args.embed_model if embedder else None}


def build_reference(args, embedder):
    t0 = time.time()
    from datasketch import MinHashLSH
    ref = {"ids": [], "text": [], "img_owner": [], "phash": [], "thumbs": [], "emb": []}
    gram_owner, gram_df = {}, defaultdict(int)
    lsh = MinHashLSH(threshold=args.minhash_threshold, num_perm=args.num_perm)
    mh = {}
    for k, (rid, text, imgs) in enumerate(iter_reference(args)):
        ref["ids"].append(rid)
        ref["text"].append(text[:600])
        w = norm_text(text).split()
        for g in set(ngrams(w, args.ngram)):
            hg = h64(g)
            gram_df[hg] += 1
            gram_owner.setdefault(hg, k)
        if len(w) >= args.min_words:
            m = minhash(w, args.num_perm)
            lsh.insert(str(k), m)
            mh[k] = m
        for im in imgs:
            ref["img_owner"].append(k)
            ref["phash"].append(phash64(im))
            ref["thumbs"].append(thumb_b64(im))
            if embedder:
                ref["emb"].append(embedder([im])[0])
        if (k + 1) % 1000 == 0:
            print(f"[ref] {k + 1} items, {len(ref['phash'])} images", file=sys.stderr)
    # n-grams shared by many reference items are boilerplate ("which of the following ..."), not leakage
    ref["grams"] = {g: o for g, o in gram_owner.items() if gram_df[g] < args.ngram_max_df}
    ref["lsh"], ref["mh"] = lsh, mh
    ref["cfg"] = ref_config(args, embedder)
    ref["phash"] = np.array(ref["phash"], dtype=np.uint64)
    ref["emb"] = np.stack(ref["emb"]) if ref["emb"] else None
    print(f"[ref] {len(ref['ids'])} items, {len(ref['phash'])} images, "
          f"{len(gram_owner) - len(ref['grams'])} boilerplate {args.ngram}-grams dropped, {time.time() - t0:.0f}s",
          file=sys.stderr)
    return ref


# ----------------------------------------------------------------------------- candidates
def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def row_images(row, image_root):
    from PIL import Image
    paths = row.get("image", row.get("images")) or []
    paths = [paths] if isinstance(paths, str) else paths
    out = []
    for p in paths:
        fp = p if os.path.isabs(p) else os.path.join(image_root or "", p)
        try:
            out.append(Image.open(fp))
        except OSError as e:
            print(f"[warn] {row.get('id')}: cannot open {fp} ({e})", file=sys.stderr)
    return out


def row_text(row, human_only=False):
    if row.get("question") is not None:  # structured row: question + options, built like the MMMU reference
        opts = row.get("options") or []
        return " ".join([str(row["question"]), *map(str, opts)])
    conv = row.get("conversations") or row.get("messages") or []
    parts = [c.get("value") or c.get("content") or "" for c in conv if isinstance(c, dict)
             and (not human_only or c.get("from", c.get("role")) in ("human", "user"))]
    return " ".join(str(p) for p in parts) or str(row.get("question", ""))


def iter_candidates(path, image_root):
    for i, row in enumerate(load_rows(path)):
        yield row.get("id", f"row{i}"), row_text(row), row_images(row, image_root)


def metadata_hit(row, fields):
    vals = [str(row.get(f, "")) for f in fields]
    imgs = row.get("image", row.get("images")) or []
    vals += [imgs] if isinstance(imgs, str) else list(imgs)
    for v in vals:
        if MMMU_PAT.search(v):
            return v[:120]
    return None


# ----------------------------------------------------------------------------- main checks
def check(args, ref, embedder):
    rows = load_rows(args.data)
    flags = []
    batch_imgs, batch_owner = [], []

    def flush_embed():
        if not (embedder and batch_imgs and ref["emb"] is not None):
            batch_imgs.clear(); batch_owner.clear(); return
        e = embedder(batch_imgs)
        sims = e @ ref["emb"].T
        best = sims.argmax(1)
        for j, (fi, img) in enumerate(zip(batch_owner, batch_imgs)):
            s = float(sims[j, best[j]])
            if s >= args.embed_threshold:
                k = ref["img_owner"][best[j]]
                flags[fi]["reasons"].append("embed")
                flags[fi]["evidence"]["embed"] = {"ref_id": ref["ids"][k], "cos": round(s, 4), "ref_img": int(best[j])}
                flags[fi]["cand_thumb"] = flags[fi].get("cand_thumb") or thumb_b64(img)
        batch_imgs.clear(); batch_owner.clear()

    for i, row in enumerate(rows):
        rid = row.get("id", f"row{i}")
        rec = {"index": i, "id": rid, "reasons": [], "evidence": {}}
        flags.append(rec)
        hit = metadata_hit(row, args.source_fields)
        if hit:
            rec["reasons"].append("metadata"); rec["evidence"]["metadata"] = hit
        text = row_text(row)
        rec["text"] = text[:600]
        w = norm_text(text).split()
        shared = defaultdict(int)
        for g in set(ngrams(w, args.ngram)):
            o = ref["grams"].get(h64(g))
            if o is not None:
                shared[o] += 1
        if shared:
            k, n = max(shared.items(), key=lambda kv: kv[1])
            rec["reasons"].append("ngram")
            rec["evidence"]["ngram"] = {"ref_id": ref["ids"][k], "shared_13grams": n, "ref_k": k}
        wq = norm_text(row_text(row, human_only=True)).split()  # MinHash on the question side only
        if len(wq) >= args.min_words:
            m = minhash(wq, args.num_perm)
            best = max(((ref["mh"][int(c)].jaccard(m), int(c)) for c in ref["lsh"].query(m)), default=None)
            if best and best[0] >= args.minhash_threshold:
                rec["reasons"].append("minhash")
                rec["evidence"]["minhash"] = {"ref_id": ref["ids"][best[1]], "jaccard": round(best[0], 3),
                                              "ref_k": best[1]}
        for img in row_images(row, args.image_root):
            if len(ref["phash"]):
                d = popcount64(ref["phash"] ^ np.uint64(phash64(img)))
                j = int(d.argmin())
                if d[j] <= args.phash_max_dist:
                    k = ref["img_owner"][j]
                    rec["reasons"].append("phash")
                    rec["evidence"]["phash"] = {"ref_id": ref["ids"][k], "dist": int(d[j]), "ref_img": j}
                    rec["cand_thumb"] = thumb_b64(img)
            if embedder:
                batch_imgs.append(img); batch_owner.append(i)
                if len(batch_imgs) >= args.batch:
                    flush_embed()
        if (i + 1) % 5000 == 0:
            print(f"[check] {i + 1}/{len(rows)}", file=sys.stderr)
    flush_embed()
    for rec in flags:
        rec["reasons"] = sorted(set(rec["reasons"]))
    return rows, [r for r in flags if r["reasons"]]


# ----------------------------------------------------------------------------- outputs
def write_outputs(args, rows, flagged, ref):
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "flags.jsonl"), "w", encoding="utf-8") as f:
        for r in flagged:
            f.write(json.dumps({k: v for k, v in r.items() if k != "cand_thumb"}, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "decisions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "id", "reasons", "matched_ref", "decision"])
        for r in flagged:
            ev = r["evidence"]
            ref_id = next((ev[k]["ref_id"] for k in ("ngram", "minhash", "phash", "embed") if k in ev), "")
            w.writerow([r["index"], r["id"], "+".join(r["reasons"]), ref_id, "remove"])
    # review page
    html = ["<html><head><meta charset='utf-8'><style>body{font-family:sans-serif;margin:24px}"
            ".c{display:grid;grid-template-columns:120px 1fr 1fr;gap:12px;border-bottom:1px solid #ddd;padding:12px 0}"
            "img{max-width:160px;border:1px solid #ccc}.r{color:#c0392b;font-weight:bold}small{color:#666}</style>"
            "</head><body><h2>Decontamination review</h2>"
            f"<p>{len(flagged)} flagged of {len(rows)}. Default decision = remove. Edit decisions.csv to keep false "
            "positives, then re-run with --apply.</p>"]
    for r in flagged[: args.max_review]:
        ev = r["evidence"]
        k = next((ev[x].get("ref_k") for x in ("ngram", "minhash") if x in ev), None)
        ref_txt = ref["text"][k] if k is not None else ""
        ref_img_idx = next((ev[x]["ref_img"] for x in ("phash", "embed") if x in ev), None)
        if ref_img_idx is not None and not ref_txt:
            ref_txt = ref["text"][ref["img_owner"][ref_img_idx]]
        ref_thumb = f"<img src='data:image/jpeg;base64,{ref['thumbs'][ref_img_idx]}'>" if ref_img_idx is not None else ""
        cand_thumb = f"<img src='data:image/jpeg;base64,{r['cand_thumb']}'>" if r.get("cand_thumb") else ""
        esc = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;")
        html.append(f"<div class='c'><div><b>#{r['index']}</b><br><span class='r'>{'+'.join(r['reasons'])}</span>"
                    f"<br><small>{esc(json.dumps({x: {kk: vv for kk, vv in ev[x].items() if kk not in ('ref_k', 'ref_img')} if isinstance(ev[x], dict) else ev[x] for x in ev}, ensure_ascii=False))}</small></div>"
                    f"<div><b>candidate</b> {esc(r['id'])}<br>{cand_thumb}<p>{esc(r['text'][:400])}</p></div>"
                    f"<div><b>MMMU</b><br>{ref_thumb}<p>{esc(ref_txt[:400])}</p></div></div>")
    html.append("</body></html>")
    with open(os.path.join(args.out, "review.html"), "w", encoding="utf-8") as f:
        f.write("".join(html))
    apply_decisions(args, rows)


def apply_decisions(args, rows=None):
    rows = rows if rows is not None else load_rows(args.data)
    dec_path = os.path.join(args.out, "decisions.csv")
    with open(dec_path, encoding="utf-8") as f:
        decisions = list(csv.DictReader(f))
    drop = {int(d["index"]) for d in decisions if d["decision"].strip().lower() != "keep"}
    kept = [r for i, r in enumerate(rows) if i not in drop]
    with open(os.path.join(args.out, "clean.jsonl"), "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_reason = defaultdict(int)
    for d in decisions:
        for x in d["reasons"].split("+"):
            by_reason[x] += 1
    n_keep = sum(d["decision"].strip().lower() == "keep" for d in decisions)
    lines = [f"# Decontamination report: `{args.data}`", "",
             f"- candidates: **{len(rows)}**",
             f"- flagged: **{len(decisions)}** ({', '.join(f'{k} {v}' for k, v in sorted(by_reason.items()))})",
             f"- kept after review (false positives): **{n_keep}**",
             f"- removed: **{len(drop)}** → clean set **{len(kept)}** rows (`clean.jsonl`)", "",
             f"Thresholds: {args.ngram}-gram any overlap (boilerplate df ≥ {args.ngram_max_df} ignored), "
             f"MinHash Jaccard ≥ {args.minhash_threshold}, pHash distance ≤ {args.phash_max_dist}, "
             f"CLIP cosine ≥ {args.embed_threshold}."]
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="candidate JSONL (qwen-vl-finetune format)")
    p.add_argument("--image_root", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--apply", action="store_true", help="only rebuild clean.jsonl/report.md from decisions.csv")
    p.add_argument("--source_fields", nargs="*", default=["source", "data_source", "dataset", "subset", "id"])
    # reference
    p.add_argument("--mmmu_splits", nargs="+", default=["dev", "validation", "test"])
    p.add_argument("--skip_pro", action="store_true")
    p.add_argument("--data_root", default=None, help="HF datasets cache dir")
    p.add_argument("--ref_cache", default="outputs/decontam/mmmu_ref.pkl")
    p.add_argument("--ref_jsonl", default=None, help="offline reference in the candidate format (testing)")
    p.add_argument("--ref_image_root", default=None)
    # thresholds
    p.add_argument("--ngram", type=int, default=13)
    p.add_argument("--ngram_max_df", type=int, default=10)
    p.add_argument("--min_words", type=int, default=8, help="texts shorter than this skip MinHash")
    p.add_argument("--num_perm", type=int, default=128)
    p.add_argument("--minhash_threshold", type=float, default=0.8)
    p.add_argument("--phash_max_dist", type=int, default=8)
    p.add_argument("--embed_model", default="openai/clip-vit-base-patch32")
    p.add_argument("--embed_threshold", type=float, default=0.95)
    p.add_argument("--no_embed", action="store_true", help="skip CLIP similarity (pHash only)")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max_review", type=int, default=2000)
    args = p.parse_args()

    if args.apply:
        apply_decisions(args)
        return
    embedder = None if args.no_embed else Embedder(args.embed_model)
    ref = None
    if args.ref_cache and os.path.exists(args.ref_cache) and not args.ref_jsonl:
        with open(args.ref_cache, "rb") as f:
            ref = pickle.load(f)
        if ref.get("cfg") != ref_config(args, embedder):
            print("[ref] cache built with different settings -> rebuilding", file=sys.stderr)
            ref = None
        else:
            print(f"[ref] loaded cache {args.ref_cache}", file=sys.stderr)
    if ref is None:
        ref = build_reference(args, embedder)
        if args.ref_cache and not args.ref_jsonl:
            os.makedirs(os.path.dirname(args.ref_cache) or ".", exist_ok=True)
            with open(args.ref_cache, "wb") as f:
                pickle.dump(ref, f)
    rows, flagged = check(args, ref, embedder)
    write_outputs(args, rows, flagged, ref)


if __name__ == "__main__":
    main()
