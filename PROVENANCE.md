# Provenance review

## Release scope

This release contains the in-house LoRA dataset **preparation pipeline v0.1.4** only.

## Separate Dataset Collector

The private working archive also contains `LoRA Dataset Collector v3.3.1`. Its supplied files do not contain an author name, copyright line, license file, upstream repository URL or attribution notice. Public searches using distinctive UI text and code identifiers did not identify an upstream source during this review.

The Collector was also compared against the supplied LoRA Dataset Studio Windows package. Searches for distinctive Collector identifiers such as `FaceVerifier`, `FACE_REVIEW_LIMIT`, `face_review_manifest`, `relevant_record`, `parse_bing_records`, `Preferred long side`, and `Curated supplemental folder` produced no matches in that inspected package. This does **not** prove independent authorship; it only means the available evidence does not establish that the Collector came from LoRA Dataset Studio.

For that reason, the Collector is excluded from the public package until its provenance/license can be established or a separately authored replacement is created.

## LoRA Dataset Studio

LoRA Dataset Studio is an upstream third-party project by `perfectgf`. The inspected Windows package contains PolyForm Noncommercial 1.0.0 licensing. No LoRA Dataset Studio code is included here.

## Preparation pipeline

The supplied preparation-pipeline files identify their own version history (v0.1.2-v0.1.4) and contain no third-party source attribution indicating that the scripts themselves were copied from another project. They invoke third-party libraries/models normally through their public APIs. The public package therefore licenses these preparation scripts under MIT while leaving all third-party components under their own terms.

This provenance note is an engineering audit, not legal advice.
