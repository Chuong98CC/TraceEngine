"""Helper functions for PT2 export and inference (see package ``infer_pt2``).

- ``img_utils.py``    image loading / letterbox / tensorization (pre-processing)
- ``flow_utils.py``   crop + rescale of the raw flow (post-processing)
- ``export_utils.py`` bf16 normalize patch + iteration-unrolling wrapper
"""
