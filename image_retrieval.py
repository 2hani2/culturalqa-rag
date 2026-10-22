"""Shared CLIP image retrieval, independent of the text MoE comparison.

This adapts the question -> image relevance -> top-K idea from RetVQA.
It does not implement RetVQA's Faster R-CNN relevance encoder or MI-BART.
Both text launchers expose the same CLIP retrieval of distinct images.
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
    """Shared CLIP image retrieval, independent of the two text ranking policies."""

    def __init__(self,image_encoder,records,dataset_filter=None,top_k=3,candidate_k=12):
        if not 1<=top_k<=candidate_k:
            raise ValueError('Require 1 <= top_k <= candidate_k.')
        self.pool=ImagePoolRetriever(image_encoder,records,dataset_filter)
        self.top_k=top_k;self.candidate_k=candidate_k

    def search(self,question,verbose=True):
        if excluded_dataset_in_query(question):return []
        hits=self.pool.retrieve(question,top_k=self.candidate_k)
        output=[]
        for rank,hit in enumerate(hits[:self.top_k],1):
            item={k:v for k,v in hit.items() if k!='_records'}
            item.update(rank=rank,ranking_score=hit['clip_score'],moe_rescore_used=False,ranking_policy='clip_shared')
            output.append(item)
            if verbose:print(f"[image] {rank}: CLIP={item['clip_score']:.4f} | {item['image_ref']}")
        if verbose and not output:print('[image retrieval] No usable images across the active datasets.')
        return output


def load_image_search(dataset_filter=None,top_k=3,candidate_k=12,
                      records_cache=str(ROOT/'faiss_store/raw_records_cache.json')):
    """Image-only entry: no text reader or MoE model is needed."""
    from encoders import ImageEncoder
    with open(records_cache,encoding='utf-8') as stream:records=json.load(stream)
    return ImageSearch(ImageEncoder(enable_live_fetch=True),records,dataset_filter,top_k,candidate_k)


def run_image_questions(search, questions, output_path=None):
    results = []
    for index, question in enumerate(questions, start=1):
        print(f"[image batch] {index}/{len(questions)}: {question}")
        try:
            hits = search.search(question)
            results.append({"question": question, "images": hits, "message": ""})
        except Exception as exc:
            message = f"[error] {exc}"
            print(message)
            results.append({"question": question, "images": [], "message": message})
    if output_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"image_results_{timestamp}.csv"
    fields = ["question", "rank", "image_ref", "dataset", "record_id", "clip_score",
              "ranking_score", "ranking_policy", "moe_rescore_used", "message"]
    with open(output_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            if not result["images"]:
                writer.writerow({"question": result["question"], "moe_rescore_used": False,
                                 "message": result["message"] or "No usable images in the selected pool."})
            for hit in result["images"]:
                row = {field: hit.get(field, "") for field in fields}
                row["question"] = result["question"]
                writer.writerow(row)
    print(f"[image batch] saved image results to {output_path}")
    return results
