"""RallyReview computer-vision pipeline.

Turns a single-rally badminton video into court-space trajectories:

    court detection -> player tracking -> shuttle tracking -> shot detection
    -> court-coordinate mapping -> RallyAnalysis

Assumes the standard broadcast view: a static camera behind and above one
baseline with the whole court in frame. See docs/cv-pipeline.md.
"""
