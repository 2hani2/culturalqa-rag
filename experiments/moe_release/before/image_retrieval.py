"""CLIP image retrieval with optional trained MoE rescoring.

This adapts the question -> image relevance -> top-K idea from RetVQA.
It does not implement RetVQA's Faster R-CNN relevance encoder or MI-BART.
Both conditions start from the same CLIP shortlist of distinct images.
"""

import csv
import datetime
import json
import os
from collections import OrderedDict

from dataset_config import is_active_dataset, validate_dataset_filter, excluded_dataset_in_query
from project_paths import ROOT

import torch
import torch.nn.functional as F


def _image_ref(record):
    for key in ("image_url", "image_path", "images"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _image_key(ref):
    if ref.startswith(("http://", "https://")):return ref
    if os.path.isfile(ref):
        from research_protocol import image_digest
        return 'image:'+image_digest(ref)
    return os.path.realpath(ref)


def _valid_vector(value):
    vector = torch.as_tensor(value).detach().cpu().float().reshape(-1)
    if not torch.isfinite(vector).all() or vector.norm().item() <= 1e-12:
        return None
    return vector


class ImagePoolRetriever:
    """Encode each distinct image once and rank using CLIP cosine similarity."""

    def __init__(self, image_encoder, records, dataset_filter=None):
        validate_dataset_filter(dataset_filter)
        self.image_encoder = image_encoder
        self.datasets_searched = sorted({r.get("dataset") for r in records if is_active_dataset(r.get("dataset"))})
        groups = OrderedDict()
        for record in records:
            if not is_active_dataset(record.get("dataset")):
                continue
            ref = _image_ref(record)
            if not ref:
                continue
            group = groups.setdefault(_image_key(ref), {"image_ref": ref, "_records": []})
            group["_records"].append(record)
        self.groups = list(groups.values())
        self.datasets_with_images = sorted({r["dataset"] for g in self.groups for r in g["_records"]})
        self._indexed_groups = None
        self._matrix = None

    def _ensure_index(self):
        if self._indexed_groups is not None:
            return
        if not self.image_encoder.enable_live_fetch:
            raise ValueError("Image retrieval requires image encoding to be enabled.")
        groups, vectors = [], []
        for group in self.groups:
            try:
                vector = _valid_vector(self.image_encoder.encode(group["image_ref"]))
            except Exception as exc:
                print(f"[image retrieval] skipped {group['image_ref']}: {exc}")
                continue
            # Failed loads are cached as zeros by ImageEncoder. They must not
            # outrank a valid image whose cosine similarity happens to be negative.
            if vector is None:
                continue
            groups.append(group)
            vectors.append(vector)
        self._indexed_groups = groups
        self._matrix = F.normalize(torch.stack(vectors), dim=-1) if vectors else None
        print(f"[image retrieval] {len(groups)} usable distinct images; "
              f"{len(self.groups) - len(groups)} missing/invalid image embeddings")

    def retrieve(self, question, top_k=6):
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        self._ensure_index()
        if self._matrix is None:
            return []
        query = _valid_vector(self.image_encoder.encode_text_for_retrieval(question))
        if query is None:
            raise ValueError("CLIP returned an invalid question embedding.")
        scores = self._matrix @ F.normalize(query, dim=0)
        order = sorted(range(len(scores)),
                       key=lambda i: (-float(scores[i]), self._indexed_groups[i]["image_ref"]))
        hits = []
        for index in order[:top_k]:
            group = self._indexed_groups[index]
            first = group["_records"][0]
            hits.append({
                "image_ref": group["image_ref"],
                "image_name": first.get("image_name") or os.path.basename(group["image_ref"]),
                "dataset": first.get("dataset"),
                "record_id": first.get("id"),
                "associated_record_count": len(group["_records"]),
                "clip_score": float(scores[index]),
                "_records": group["_records"],
            })
        return hits


class ImageSearch:
    """Use CLIP for both conditions; optionally rescore that shortlist with MoE.

    Several QA records may share one image. For MoE, evaluate their associated
    text/image representations and use the highest relevance score per image.
    The selected record ID identifies that best associated text, not a new image.
    """

    def __init__(self, image_encoder, records, dataset_filter=None,
                 text_encoder=None, moe=None, top_k=3, candidate_k=6, batch_size=16):
        if top_k < 1 or candidate_k < top_k or batch_size < 1:
            raise ValueError("Require 1 <= top_k <= candidate_k and batch_size >= 1.")
        self.pool = ImagePoolRetriever(image_encoder, records, dataset_filter)
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.moe = moe
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.batch_size = batch_size

    @torch.no_grad()
    def _rescore(self, question, hits):
        if self.moe is None or self.text_encoder is None:
            raise ValueError("MoE image rescoring requires a trained model and a text encoder.")
        from structured_store import record_to_text

        query_text = self.text_encoder.encode(question)
        query_image = torch.zeros(1, self.image_encoder.embed_dim)
        for hit in hits:
            image = self.image_encoder.encode(hit["image_ref"]).reshape(1, -1).float()
            best_score, best_record = float("-inf"), None
            records = hit["_records"]
            for start in range(0, len(records), self.batch_size):
                batch = records[start:start + self.batch_size]
                # Match the existing BERT encoder's short-context use.
                texts = [record_to_text(record)[:2000] for record in batch]
                text_vectors = self.text_encoder.encode(texts)
                count = len(batch)
                scores = self.moe.score_relevance(
                    query_text.expand(count, -1), query_image.expand(count, -1),
                    text_vectors, image.expand(count, -1),
                )
                if not torch.isfinite(scores).all():
                    raise ValueError("MoE produced non-finite relevance scores.")
                index = int(torch.argmax(scores))
                score = float(scores[index])
                if score > best_score:
                    best_score, best_record = score, batch[index]
            hit["moe_score"] = best_score
            hit["record_id"] = best_record.get("id")
            hit["dataset"] = best_record.get("dataset")

    def search(self, question, use_moe=False, verbose=True):
        excluded=excluded_dataset_in_query(question)
        if excluded:
            if verbose:print(f'[image retrieval] {excluded} is excluded.')
            return []
        hits = self.pool.retrieve(question, top_k=self.candidate_k)
        if use_moe and hits:
            self._rescore(question, hits)
            hits.sort(key=lambda hit: (-hit["moe_score"], -hit["clip_score"], hit["image_ref"]))
        output = []
        for rank, hit in enumerate(hits[:self.top_k], start=1):
            item = {key: value for key, value in hit.items() if key != "_records"}
            item["rank"] = rank
            item["moe_score"] = hit.get("moe_score")
            item["ranking_score"] = hit["moe_score"] if use_moe else hit["clip_score"]
            item["moe_rescore_used"] = bool(use_moe)
            output.append(item)
            if verbose:
                extra = f" | MoE={item['moe_score']:.4f}" if use_moe else ""
                print(f"[image] {rank}: CLIP={item['clip_score']:.4f}{extra} | {item['image_ref']}")
        if verbose and not output:
            print("[image retrieval] No usable images across the active datasets.")
        return output


def load_image_search(dataset_filter=None, use_moe=False, top_k=3, candidate_k=6,
                      records_cache=str(ROOT / "faiss_store/raw_records_cache.json"),
                      checkpoint_path=str(ROOT / "checkpoints/moe_fusion.pt")):
    """Image-only entry: no FAISS text search, reader, or Groq key required."""
    from encoders import ImageEncoder, TextEncoder
    from moe_fusion import MoEFusion

    if not os.path.isfile(records_cache):
        raise FileNotFoundError(f"Missing {records_cache}. Build the dataset index first.")
    if use_moe and not os.path.isfile(checkpoint_path):
        raise FileNotFoundError("No trained MoE checkpoint. Run train_moe.py first, "
                                "or use main_without_moe.py for CLIP image retrieval.")
    with open(records_cache, encoding="utf-8") as stream:
        records = json.load(stream)
    from reranker_models import load_model
    moe = load_model("moe", checkpoint_path)[0] if use_moe else None
    text_encoder = TextEncoder() if use_moe else None
    return ImageSearch(ImageEncoder(enable_live_fetch=True), records, dataset_filter,
                       text_encoder=text_encoder, moe=moe, top_k=top_k, candidate_k=candidate_k)


def run_image_questions(search, questions, use_moe=False, output_path=None):
    results = []
    for index, question in enumerate(questions, start=1):
        print(f"[image batch] {index}/{len(questions)}: {question}")
        try:
            hits = search.search(question, use_moe=use_moe)
            results.append({"question": question, "images": hits, "message": ""})
        except Exception as exc:
            message = f"[error] {exc}"
            print(message)
            results.append({"question": question, "images": [], "message": message})
    if output_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"image_results_{timestamp}.csv"
    fields = ["question", "rank", "image_ref", "dataset", "record_id", "clip_score",
              "moe_score", "ranking_score", "moe_rescore_used", "message"]
    with open(output_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            if not result["images"]:
                writer.writerow({"question": result["question"], "moe_rescore_used": use_moe,
                                 "message": result["message"] or "No usable images in the selected pool."})
            for hit in result["images"]:
                row = {field: hit.get(field, "") for field in fields}
                row["question"] = result["question"]
                writer.writerow(row)
    print(f"[image batch] saved image results to {output_path}")
    return results
