from typing import List, Tuple

import re

try:
    import sacrebleu
    _HAS_SACREBLEU = True
except Exception:
    _HAS_SACREBLEU = False


def _normalize_text(s: str) -> str:
    """Lowercase, remove punctuation and extra whitespace for robust matching in EM/F1."""
    if s is None:
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9а-яёқғўҳҗӣәөӯәА-ЯЁЌё]+", " ", s)
    s = " ".join(s.split())
    return s.strip()


def exact_match(prediction: str, ground_truth: str) -> int:
    """Return 1 if normalized strings match exactly, else 0."""
    return int(_normalize_text(prediction) == _normalize_text(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between prediction and ground truth (as used in SQuAD eval).

    Tokens are simple whitespace tokens after normalization.
    """
    pred_tokens = _normalize_text(prediction).split()
    gt_tokens = _normalize_text(ground_truth).split()
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = 0
    from collections import Counter
    pred_counts = Counter(pred_tokens)
    gt_counts = Counter(gt_tokens)
    for t in pred_counts:
        common += min(pred_counts[t], gt_counts.get(t, 0))
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu_score(references: List[str], hypotheses: List[str]) -> float:
    """Compute corpus BLEU score. references and hypotheses are lists of strings.

    Uses sacrebleu if available for stable scoring; otherwise falls back to a simple unigram BLEU approximation.
    """
    if not hypotheses:
        return 0.0
    if _HAS_SACREBLEU:
        # sacrebleu expects list of references (each its own list) and list of hyps
        refs = [references]
        try:
            score = sacrebleu.corpus_bleu(hypotheses, refs)
            return float(score.score)
        except Exception:
            pass

    # Fallback: simplistic BLEU using 1-gram precision averaged (not a replacement for sacrebleu, but usable)
    def _ngram_precision(ref, hyp):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        from collections import Counter
        ref_counts = Counter(ref_tokens)
        hyp_counts = Counter(hyp_tokens)
        if not hyp_tokens:
            return 0.0
        common = 0
        for t in hyp_counts:
            common += min(hyp_counts[t], ref_counts.get(t, 0))
        return common / len(hyp_tokens)

    precisions = []
    for r, h in zip(references, hypotheses):
        precisions.append(_ngram_precision(r, h))
    # return percentage-like score
    return float(sum(precisions) / len(precisions) * 100.0)


def recommend_accuracy(recommended: List[dict], ground_truth_name: str) -> int:
    """Return 1 if ground_truth_name appears in recommended results, else 0.

    recommended: list of dicts with 'name' key. Ground truth matching is case-insensitive and partial-match tolerant.
    """
    if not recommended:
        return 0
    gt = ground_truth_name.lower()
    for r in recommended:
        if r.get('name') and gt in r.get('name').lower():
            return 1
    return 0
