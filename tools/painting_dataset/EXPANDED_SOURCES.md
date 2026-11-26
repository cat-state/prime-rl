# Expanded painting sources

## Validated NGA expansion

`build_nga.py OUTPUT --count 1000 --images 100` pins the official NGA repository commit, streams its object and published-image CSVs without retaining duplicate CSV archives, joins accessioned paintings to primary images marked `openaccess=1`, and keeps the joined source rows in the manifest. Deterministic artwork-key hashes define the split; repeated artwork keys and exact duplicate image files are rejected. A hash-sorted selection caps artists at 30 metadata records and 8 downloaded images. Metadata-only rows have null image paths and are excluded from train/heldout JSONL files.

Completed remotely in `/root/prime-rl/loop-validation/painting-data-nga-v1`: **1000 metadata records, 100 decoded images, 87 train / 13 heldout, 75 artists among images, 29 MiB total**. All downloaded images have a minimum dimension of at least 128 pixels and aspect ratio no greater than 4. Artwork keys and image SHA256 hashes are unique. This validates acquisition and structural splits, not manual artistic quality or near-duplicate freedom across other corpora. Do not merge with Met/PD12M heldout sets without catalog/perceptual matching.

Met expansion reached searchable catalog snapshots but individual object retrievals repeatedly returned 404/502 or zero-byte timeouts. That fetcher was stopped after NGA succeeded; cached snapshots and `acquisition_status.json` remain in `/root/prime-rl/loop-validation/painting-data-expanded-v1`. The original working six-image dataset was untouched.

## Museum seed collection

`build_expanded.py OUTPUT --count 100 --images 100` queries the Met's current v1.1 search endpoint, interleaves 16 subject/culture searches, caps examples per artist, and admits only records marked public domain with object type Painting and downloadable images. It checks decoding, minimum dimension, extreme aspect ratio, and duplicate image hashes. It retains museum metadata, download URLs, artwork Wikidata IDs where supplied, image SHA256, and original dimensions. Stable splits are by artwork key; the original six example assignments are preserved. All derived crops must inherit their source artwork's split. This is automatic curation, not a manual claim that every image is famous or aesthetically suitable.

The museum API is mutable. Cached source responses, recorded metadata/image hashes and candidate-ID inventory are the reproducible snapshot. Do not delete these then assume a fresh API search is the same dataset. Retry failures are recorded in `errors.json`; `manifest.json` declares completeness. JSONL files contain downloaded images only. Metadata-only rows have null image paths and cannot silently enter training.

Storage inspected before expansion on 2026-11-26: entire `/root/simple_pt` 311 GiB, `/root/prime-rl` 25 GiB, `/root/.cache` 48 GiB, filesystem 15 GiB free. No cleanup performed. Images use museum web-large variants; downloads are individually capped at 12 MB and builds require 2 GiB free. Initial target is only 100 images. A larger metadata manifest can use `--count 1000 --images 100 --artist-cap 30 --search-limit 500` in a new output directory; curate and audit availability before committing to image expansion. Same artwork can appear under alternate catalog records or in multiple museums; cross-source perceptual matching remains necessary before merging corpora.

Primary sources:

- [Met API documentation](https://metmuseum.github.io/) documents public-domain image access, the new paginated v1.1 endpoint, and object metadata. [Open Access policy](https://www.metmuseum.org/hubs/open-access).
- [Art Institute API](https://api.artic.edu/docs/) supports public-domain filtering and IIIF images. It returned HTTP 403 during the initial dataset work; it is a future provider, not a downloaded source in this manifest.
- [National Gallery of Art official open-data repository](https://github.com/NationalGalleryOfArt/opendata) supplies CC0 collection metadata. The builder joins image availability and the `published_images.openaccess` flag explicitly; its metadata license alone does not substitute for individual image provenance.

## What “Public Diffusion dataset” means

[Spawning's announcement](https://spawning.substack.com/p/it-takes-a-village-to-raise-a-dataset) distinguishes its Public Diffusion model from the published Public Domain 12M dataset. The concrete available dataset is [Spawning/PD12M](https://huggingface.co/datasets/Spawning/PD12M), with [official datasheet](https://huggingface.co/datasets/Spawning/PD12M/blob/main/Datasheet.pdf). Its metadata license is CDLA-Permissive-2.0; image records carry public-domain/CC0 license URLs. It includes many non-painting images and machine-generated captions. It is not DiffusionDB. Public Diffusion may use a larger evolving collection; do not imply PD12M is the exact full training corpus of every Public Diffusion release.

`stage_pd12m.py OUTPUT --rows 1000` obtains a bounded metadata sample, records the observed dataset revision and API page hashes, and retains caption-matched painting candidates with accepted license URLs. It downloads no image payloads. These candidates are **not training-ready**: catalog artwork identities are absent, supplied MD5 hashes remain unverified, and source captions do not establish the actual work or artist. Rows therefore have split `unassigned`. Resolve catalog identities and perform image/perceptual deduplication against museum held-out examples before use. Do not invent artwork titles from captions or assign random image-level evaluation splits.

Observed source revision: `867988b01138799b89d3ffdd5b4f7e1455951f32`. The viewer endpoint is not revision-addressed, so the sampler checks that the repository revision is unchanged before/after and records hashes of each response. A future large-scale build should read revision-pinned parquet files and retain the source licensing/attribution fields.
