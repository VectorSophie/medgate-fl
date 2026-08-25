"""Alternative coarse ontology, for the coarse-ontology sensitivity sweep
(P1 requirement, repair pass 4 -- medgate/data/synthetic.py's COARSE_MAP
comment pointed here "once Phase 2 starts"; this closes that placeholder
rather than leaving it as a permanent TODO).

The PRIMARY ontology (medgate.data.synthetic.COARSE_MAP) groups AK with
keratinocytic lesions (it is the precursor keratinocytic lesion). This
ALTERNATIVE ontology instead uses a benign/malignant/ambiguous split and
keeps AK in its own "ambiguous" bucket rather than folding it into
keratinocytic -- exactly the candidate named in earlier drafts of this
paper's coarse-ontology-sensitivity TODO, now actually run rather than
only proposed. Neither ontology is asserted as clinically authoritative;
both are experimental taxonomies used to test whether this project's
capability-isolation findings are sensitive to where the coarse/fine
boundary is drawn.
"""
from medgate.data.synthetic import FINE_CLASSES

ALTERNATIVE_COARSE_MAP = {
    "benign": ["NV", "DF", "VASC", "BKL"],
    "malignant": ["MEL", "BCC", "SCC"],
    "ambiguous": ["AK"],  # kept in its own bucket, not folded into either -- the whole point of this alternative
}
ALTERNATIVE_COARSE_CLASSES = list(ALTERNATIVE_COARSE_MAP.keys())
ALTERNATIVE_FINE_TO_COARSE_IDX = {
    fine: coarse_idx
    for coarse_idx, (_, fines) in enumerate(ALTERNATIVE_COARSE_MAP.items())
    for fine in fines
}
assert set(ALTERNATIVE_FINE_TO_COARSE_IDX) == set(FINE_CLASSES), "alternative coarse map must cover every fine class"
