# Data sources

Keep raw VCC H5AD files and STRING downloads outside Git. Use separate paths for training, validation and test. The test responses remain closed during model selection.

- VCC 2025: [Arc Institute Virtual Cell Challenge](https://github.com/ArcInstitute/arc-virtual-cell-atlas/tree/main/virtual-cell-challenge). The validation run used local training and validation H5AD files. Their SHA-256 values are below; they identify our local files but have not yet been checked against an official checksum manifest.
- Human STRING v12.0 functional network: [protein information](https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz), [combined scores](https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz), and [detailed evidence channels](https://stringdb-downloads.org/download/protein.links.detailed.v12.0/9606.protein.links.detailed.v12.0.txt.gz). Follow the data providers' terms and attribution requirements.

| Local input | SHA-256 |
| --- | --- |
| Training H5AD | `a09977104fefb622368ca74b50c9d3c1e891733e6c83db07acfca49b0219c02b` |
| Validation H5AD | `376f0bab27d9f22e00dfd888668d3602006d035edc182721c378f4802c77bfe1` |
| STRING protein info | `144de4b0d98c6a7dfde6ddc2591cf88657f27b989eadff4f501450c3ed1f0f1c` |
| STRING combined links | `3e22f32572211aa341d5b4bd08d30c32e693e294603202120936872f87719d4f` |
| STRING detailed links | `28a2ea3b22c964e691c0f2f934c5e3e397a4b0a12c833d8787fa02b7847b91b6` |

Preparation scripts record their own inputs and checksums. They use exact unique STRING preferred-name matches; investigate missing or ambiguous mappings instead of silently substituting aliases.
