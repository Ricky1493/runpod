"""
face-gpu-worker — stateless InsightFace inference for the ephemeral RunPod GPU
burst tier.

Contract: a presigned URL goes in, bounding boxes and 512-dim L2-normalized
embeddings come out.

WHAT THIS SERVICE MUST NEVER DO (plan §3, §11 of the implementation plan):
  * connect to MSSQL or PostgreSQL — the drivers are not installed
  * read or write a FAISS index
  * generate production face crops
  * write .bin embedding files
  * set isInsightFace
  * make any outbound request other than to the presigned URLs it is given

India remains the orchestrator and the only writer of durable state.

The one thing that makes this service correct rather than merely fast is
NUMERICAL PARITY with India's CPU path. preprocess.py and quality.py are
deliberate mirrors of core/image_processor.py and core/face_detector.py from the
India repository, guarded by committed reference fixtures and by the parity
harness. An embedding from the wrong model, or from a differently-resized image,
is numerically plausible and silently incompatible with an 11.3M-face index.
"""

import os

__version__ = os.environ.get("IMAGE_VERSION", "unknown")
