# scoring_utils.py - Soft scoring adjustments and safety utilities
"""
Safe, parameterized scoring adjustment functions for rephrased content
matching. Replaces hard thresholds with smooth, tunable adjustments.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SoftScoreAdjuster:
    """
    Safe, parameterized score adjustment using soft functions instead of
    hard thresholds and fixed boosts.
    """
    
    @staticmethod
    def sigmoid(x: float, midpoint: float = 0.5, steepness: float = 10.0) -> float:
        """
        Smooth sigmoid function for soft thresholding.
        
        Args:
            x: Input value (typically 0-1 score)
            midpoint: Center point of sigmoid (where output = 0.5)
            steepness: Steepness of transition (higher = sharper)
        
        Returns:
            Value between 0 and 1
        """
        return 1.0 / (1.0 + np.exp(-steepness * (x - midpoint)))
    
    @staticmethod
    def soft_boost(base_score: float, 
                   trigger_value: float,
                   trigger_threshold: float,
                   max_boost: float,
                   use_sigmoid: bool = True) -> float:
        """
        Apply soft boost to score based on trigger condition.
        
        Uses sigmoid to smoothly activate boost instead of hard threshold.
        Output is always clamped to [0, 1].
        
        Args:
            base_score: Original score
            trigger_value: Value that triggers the boost (e.g., semantic score)
            trigger_threshold: Threshold for boost activation
            max_boost: Maximum boost amount (percentage of base_score)
            use_sigmoid: If True, use sigmoid smoothing; else hard threshold
        
        Returns:
            Adjusted score in [0, 1]
        """
        if use_sigmoid:
            # Sigmoid activation: smooth transition around threshold
            activation = SoftScoreAdjuster.sigmoid(
                trigger_value, 
                midpoint=trigger_threshold,
                steepness=10.0
            )
        else:
            # Hard threshold
            activation = 1.0 if trigger_value > trigger_threshold else 0.0
        
        boost_amount = activation * max_boost
        adjusted = base_score + boost_amount
        return np.clip(adjusted, 0.0, 1.0)
    
    @staticmethod
    def multiplicative_boost(base_score: float,
                            trigger_value: float,
                            trigger_threshold: float,
                            multiplier: float,
                            use_sigmoid: bool = True) -> float:
        """
        Apply multiplicative boost (percentage increase).
        
        Args:
            base_score: Original score
            trigger_value: Trigger value (e.g., semantic score)
            trigger_threshold: Activation threshold
            multiplier: Multiplicative factor (e.g., 1.2 = 20% boost)
            use_sigmoid: Use sigmoid smoothing or hard threshold
        
        Returns:
            Adjusted score in [0, 1]
        """
        if use_sigmoid:
            activation = SoftScoreAdjuster.sigmoid(trigger_value, trigger_threshold, 10.0)
        else:
            activation = 1.0 if trigger_value > trigger_threshold else 0.0
        
        # Interpolate between base and boosted version
        boosted = base_score * multiplier
        adjusted = base_score + activation * (boosted - base_score)
        return np.clip(adjusted, 0.0, 1.0)
    
    @staticmethod
    def detect_paraphrase_pattern(scores: Dict[str, float],
                                  semantic_min: float = 0.60,
                                  field_score_max: float = 0.30,
                                  bm25_min: float = 0.50) -> Tuple[bool, float]:
        """
        Detect paraphrase/rephrased content using score pattern.
        
        Pattern: High semantic + Low field_score + Medium-high BM25
        = Strong match on meaning, weak match on per-field wording, good keyword overlap
        
        Returns:
            (is_paraphrase, confidence_score)
            - is_paraphrase: True if pattern matches
            - confidence: How strongly the pattern matches (0-1)
        """
        semantic    = scores.get('semantic',    0.0)
        field_score = scores.get('field_score', scores.get('lexical', 0.0))  # backward compat
        bm25        = scores.get('bm25',        0.0)
        
        # Check if pattern matches
        semantic_ok    = semantic    > semantic_min
        field_score_ok = field_score < field_score_max
        bm25_ok        = bm25        > bm25_min
        
        is_pattern = semantic_ok and field_score_ok and bm25_ok
        
        # Confidence: how well all criteria are met
        if is_pattern:
            semantic_strength    = (semantic    - semantic_min)    / (1.0 - semantic_min)
            field_score_strength = (field_score_max - field_score) / field_score_max
            bm25_strength        = (bm25        - bm25_min)        / (1.0 - bm25_min)
            
            confidence = np.mean([semantic_strength, field_score_strength, bm25_strength])
            confidence = np.clip(confidence, 0.0, 1.0)
        else:
            confidence = 0.0
        
        return is_pattern, confidence


def compute_adaptive_threshold(semantic_scores: np.ndarray,
                               base_threshold: float = 0.30,
                               high_semantic_cutoff: float = 0.70,
                               reduction_factor: float = 0.80,
                               min_threshold: float = 0.20) -> float:
    """
    Compute adaptive threshold based on semantic score distribution.
    
    If we have strong semantic matches, lower the threshold to surface them.
    
    Args:
        semantic_scores: Array of semantic similarity scores
        base_threshold: Default threshold
        high_semantic_cutoff: If max(semantic) > this, lower threshold
        reduction_factor: Multiply threshold by this (0-1)
        min_threshold: Floor for threshold
    
    Returns:
        Adaptive threshold value
    """
    if len(semantic_scores) == 0:
        return base_threshold
    
    max_semantic = np.max(semantic_scores)
    
    if max_semantic > high_semantic_cutoff:
        # Good semantic matches exist, lower the threshold
        adaptive = base_threshold * reduction_factor
        adaptive = max(adaptive, min_threshold)
        return adaptive
    
    return base_threshold


def normalize_weights_safe(raw_weights: Dict[str, float],
                           min_weight: float = 0.05,
                           max_weight: float = 0.50,
                           use_softmax: bool = True,
                           temperature: float = 1.0) -> Dict[str, float]:
    """
    Safely normalize weights with bounds and optionalsmoothing.
    
    Args:
        raw_weights: Dict of method -> score
        min_weight: Minimum weight for any method
        max_weight: Maximum weight for any method
        use_softmax: If True, use softmax smoothing; else simple normalization
        temperature: Softmax temperature (lower = sharper)
    
    Returns:
        Dict of method -> normalized weight in [min_weight, max_weight]
    """
    methods = list(raw_weights.keys())
    scores = np.array([raw_weights[m] for m in methods])
    
    # Handle all-zero case
    if np.sum(scores) == 0:
        return {m: 1.0 / len(methods) for m in methods}
    
    if use_softmax:
        # Softmax with temperature
        exp_scores = np.exp(scores / (temperature + 1e-10))
        weights = exp_scores / (np.sum(exp_scores) + 1e-10)
    else:
        # Simple normalization
        weights = scores / (np.sum(scores) + 1e-10)
    
    # Apply min/max bounds
    clipped = np.clip(weights, min_weight, max_weight)
    
    # Re-normalize after clipping to ensure sum = 1
    clipped = clipped / (np.sum(clipped) + 1e-10)
    
    return {m: float(w) for m, w in zip(methods, clipped)}


def validate_score_dict(scores: Dict[str, float], 
                       expected_methods: list = None) -> bool:
    """
    Validate score dictionary structure.
    
    Args:
        scores: Dict to validate
        expected_methods: Optional list of expected method names
    
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(scores, dict):
        logger.warning(f"Scores is not dict: {type(scores)}")
        return False
    
    for method, score in scores.items():
        if not isinstance(score, (int, float, np.number)):
            logger.warning(f"Score for {method} is not numeric: {type(score)}")
            return False
        
        if not (0.0 <= score <= 1.0):
            logger.warning(f"Score for {method} out of bounds: {score}")
            return False
    
    if expected_methods:
        missing = set(expected_methods) - set(scores.keys())
        if missing:
            logger.warning(f"Missing methods in scores: {missing}")
            return False
    
    return True


def safe_rerank(all_final_scores: np.ndarray,
                all_method_scores: Dict[str, np.ndarray],
                config: 'RerankingConfig') -> Tuple[np.ndarray, Dict[str, list]]:
    """
    Safely re-rank cases detecting rephrased content.
    
    Args:
        all_final_scores: Array of final scores
        all_method_scores: Dict mapping method -> array of scores
        config: RerankingConfig with thresholds and parameters
    
    Returns:
        (re_ranked_scores, detection_log)
        - re_ranked_scores: Adjusted scores
        - detection_log: Dict with indices and reasons for boosts
    """
    re_ranked = all_final_scores.copy()
    detection_log = {
        'detected_indices': [],
        'confidence_scores': [],
        'boost_amounts': [],
        'boost_reasons': []
    }
    
    if not config.enable_reranking:
        return re_ranked, detection_log
    
    semantic    = all_method_scores.get('semantic',    np.zeros_like(all_final_scores))
    field_score = all_method_scores.get('field_score',
                  all_method_scores.get('lexical',     np.zeros_like(all_final_scores)))  # backward compat
    bm25        = all_method_scores.get('bm25',        np.zeros_like(all_final_scores))
    
    # Detect paraphrase pattern for each case
    for idx in range(len(all_final_scores)):
        scores_dict = {
            'semantic':    semantic[idx],
            'field_score': field_score[idx],
            'bm25':        bm25[idx]
        }
        
        is_paraphrase, confidence = SoftScoreAdjuster.detect_paraphrase_pattern(
            scores_dict,
            semantic_min=config.semantic_threshold,
            field_score_max=config.lexical_threshold,
            bm25_min=config.bm25_threshold
        )
        
        if is_paraphrase and confidence > 0.3:  # Only boost if reasonably confident
            old_score = re_ranked[idx]
            
            if config.use_additive_boost:
                boost_amount = config.additive_boost_amount
                new_score = min(old_score + boost_amount, config.boost_max_cap)
            else:
                new_score = min(old_score * config.boost_multiplier, config.boost_max_cap)
            
            re_ranked[idx] = new_score
            
            detection_log['detected_indices'].append(idx)
            detection_log['confidence_scores'].append(float(confidence))
            detection_log['boost_amounts'].append(float(new_score - old_score))
            detection_log['boost_reasons'].append(
                f"semantic={semantic[idx]:.3f}, field_score={field_score[idx]:.3f}, bm25={bm25[idx]:.3f}"
            )
            
            if config.verbose:
                logger.info(
                    f"🔄 Rephrased content detected at {idx}: "
                    f"confidence={confidence:.3f}, "
                    f"boosted {old_score:.3f} -> {new_score:.3f}"
                )
    
    return re_ranked, detection_log
# scoring_utils.py - Add these functions at the end of the file

def normalize_scores(scores):
    """
    Safely normalize scores to [0, 1] range
    Handles edge cases like all scores equal
    """
    import numpy as np
    
    scores = np.array(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    if max_score == min_score:
        # If all scores are equal, return 0.5 for all
        return np.full_like(scores, 0.5, dtype=float)
    
    return (scores - min_score) / (max_score - min_score + 1e-9)


def calculate_heading_similarity(query_heading: str, case_heading: str) -> float:
    """
    Calculate heading similarity score [0, 1]
    Uses multiple strategies for robustness
    """
    if not query_heading or not case_heading:
        return 0.0
    
    # Convert to lowercase for comparison
    q = query_heading.lower().strip()
    c = case_heading.lower().strip()
    
    # Exact match gets 1.0
    if q == c:
        return 1.0
    
    # Check if one contains the other
    if q in c or c in q:
        # Length ratio determines score
        shorter = min(len(q), len(c))
        longer = max(len(q), len(c))
        return shorter / longer
    
    # Token-based similarity
    q_tokens = set(q.split())
    c_tokens = set(c.split())
    
    if not q_tokens or not c_tokens:
        return 0.0
    
    # Jaccard similarity
    intersection = len(q_tokens & c_tokens)
    union = len(q_tokens | c_tokens)
    
    return intersection / union if union > 0 else 0.0


def adaptive_alpha(heading_similarity: float, base_alpha: float = 0.65, alpha_range: float = 0.3) -> float:
    """
    Adjust alpha based on heading match strength
    Strong heading match → More weight to retrieval (preserve exact match)
    Weak heading match → More weight to CE (find semantic matches)
    """
    import numpy as np
    
    # heading_sim = 1.0 → alpha = base_alpha - alpha_range/2
    # heading_sim = 0.0 → alpha = base_alpha + alpha_range/2
    min_alpha = base_alpha - (alpha_range / 2)
    max_alpha = base_alpha + (alpha_range / 2)
    
    # Inverse relationship: higher heading_sim = lower alpha
    alpha = base_alpha - (heading_similarity * alpha_range / 2)
    return np.clip(alpha, min_alpha, max_alpha)