# per_case_optimizer_enhanced.py - Enhanced version with soft scoring and config
"""
Enhanced per-case optimizer with:
- Config-driven parameterization
- Soft scoring adjustments (sigmoid instead of hard thresholds)
- Rephrased content detection
- Safe weight normalization
"""

import numpy as np
from config import SimilarityConfig, load_config
from scoring_utils import (
    SoftScoreAdjuster, 
    normalize_weights_safe,
    compute_adaptive_threshold
)
from per_case_optimizer_v2 import PerCaseOptimizerV2
import logging

logger = logging.getLogger(__name__)


class EnhancedPerCaseOptimizer(PerCaseOptimizerV2):
    """
    Extended PerCaseOptimizerV2 with:
    - Configuration-driven parameters
    - Soft scoring adjustments
    - Safe weight computation
    - Rephrased content awareness
    """
    
    def __init__(self, 
                 corpus_embeddings: np.ndarray,
                 corpus_metadata,
                 config: SimilarityConfig = None,
                 **kwargs):
        """
        Initialize enhanced optimizer.
        
        Args:
            corpus_embeddings: Corpus embeddings array
            corpus_metadata: Corpus metadata DataFrame
            config: SimilarityConfig instance (loaded from file if None)
            **kwargs: Additional args passed to parent class
        """
        super().__init__(corpus_embeddings, corpus_metadata, **kwargs)
        
        # Load or create configuration
        self.config = config if config is not None else load_config()
        
        logger.info(f"✅ Enhanced optimizer initialized with config")
        logger.debug(f"   Reranking enabled: {self.config.enable_reranking}")
        logger.debug(f"   Adaptive threshold enabled: {self.config.enable_adaptive_threshold}")
        logger.debug(f"   Dynamic weighting enabled: {self.config.enable_dynamic_weighting}")
    
    def _adjust_method_score_with_config(self, 
                                         method: str,
                                         base_score: float,
                                         case_features: dict) -> float:
        """
        Apply configuration-driven soft score adjustments.
        
        Replaces hard thresholds with sigmoid-based soft adjustments.
        
        Args:
            method: Method name ('semantic', 'lexical', etc.)
            base_score: Original score
            case_features: Extracted features from case
        
        Returns:
            Adjusted score in [0, 1]
        """
        adjustment = 0.0
        
        # Semantic adjustments - boost for rephrased content
        if method == 'semantic':
            cfg = self.config.semantic
            
            text_length = case_features.get('text_length', 0)
            
            # 1. STRICT DISABLE: No semantic boost for very short queries
            if text_length < 30:
                return base_score
                
            tech_density = case_features.get('technical_term_density', 0)
            description_complexity = case_features.get('description_complexity', 0)
            
            # 2. Gate condition: Must be long AND complex, OR have strong technical/rephrased signal
            is_long_and_complex = (text_length > cfg.min_text_length_long) and \
                                  (description_complexity > cfg.min_description_complexity)
            
            strong_technical_signal = (tech_density > cfg.min_technical_density)
            has_abstract_concepts = case_features.get('has_abstract_concepts', False)
            
            if is_long_and_complex or strong_technical_signal or has_abstract_concepts:
                # Boost for technical content
                if tech_density > cfg.min_technical_density:
                    adjustment += SoftScoreAdjuster.soft_boost(
                        base_score,
                        trigger_value=tech_density,
                        trigger_threshold=cfg.min_technical_density,
                        max_boost=cfg.min_technical_density_boost,
                        use_sigmoid=True
                    ) - base_score
                
                # Boost for complex descriptions
                if description_complexity > cfg.min_description_complexity:
                    adjustment += SoftScoreAdjuster.soft_boost(
                        base_score,
                        trigger_value=description_complexity,
                        trigger_threshold=cfg.min_description_complexity,
                        max_boost=cfg.description_complexity_boost,
                        use_sigmoid=True
                    ) - base_score
                
                # Boost for long text
                if text_length > cfg.min_text_length_long:
                    adjustment += cfg.long_text_boost * (
                        min(text_length / 200, 1.0)  # Scale by length
                    )
        
        # Lexical adjustments - reduce for rephrased content
        elif method == 'lexical':
            cfg = self.config.lexical
            
            # Reduce weight for concrete examples (likely to be rephrased)
            if case_features.get('has_concrete_examples', False):
                adjustment += cfg.concrete_examples_boost
            
            # Boost for short text only
            text_length = case_features.get('text_length', 0)
            if text_length < cfg.short_text_threshold:
                adjustment += cfg.short_text_boost
            
            # Penalize technical content in lexical (likely different wording)
            tech_density = case_features.get('technical_term_density', 0)
            if tech_density > cfg.technical_penalty_threshold:
                adjustment += cfg.technical_penalty  # Negative
        
        # Cross-encoder adjustments - great for rephrases
        elif method == 'cross_encoder':
            cfg = self.config.cross_encoder
            
            # Boost for high complexity
            complexity = case_features.get('description_complexity', 0)
            if complexity > cfg.min_high_complexity:
                adjustment += cfg.high_complexity_boost * (
                    min(complexity / 100, 1.0)
                )
            
            # Boost for abstract concepts
            if case_features.get('has_abstract_concepts', False):
                adjustment += cfg.abstract_concepts_boost
            
            # Boost for technical terms
            tech_density = case_features.get('technical_term_density', 0)
            if tech_density > cfg.technical_threshold:
                adjustment += SoftScoreAdjuster.soft_boost(
                    base_score,
                    trigger_value=tech_density,
                    trigger_threshold=cfg.technical_threshold,
                    max_boost=cfg.technical_boost,
                    use_sigmoid=True
                ) - base_score
        
        # BM25 adjustments
        elif method == 'bm25':
            cfg = self.config.bm25
            
            # Reduce boost for short text
            text_length = case_features.get('text_length', 0)
            if text_length < cfg.short_text_threshold:
                adjustment += cfg.short_text_boost
            
            # Boost for vocabulary specificity
            vocab_spec = case_features.get('vocabulary_specificity', 0)
            if vocab_spec > cfg.vocabulary_specificity_threshold:
                adjustment += SoftScoreAdjuster.soft_boost(
                    base_score,
                    trigger_value=vocab_spec,
                    trigger_threshold=cfg.vocabulary_specificity_threshold,
                    max_boost=cfg.vocabulary_specificity_boost,
                    use_sigmoid=True
                ) - base_score
            
            # Boost for technical terms
            tech_density = case_features.get('technical_term_density', 0)
            if tech_density > cfg.technical_threshold:
                adjustment += cfg.technical_boost
        
        # Apply adjustment and clip
        adjusted = base_score + adjustment
        return np.clip(adjusted, 0.0, 1.0)
    
    def _compute_weights_from_scores_safe(self, 
                                          method_scores: dict) -> dict:
        """
        Safely compute normalized weights from scores.
        
        Uses:
        - Dynamic scaling for semantic when high
        - Softmax smoothing with temperature
        - Min/max weight constraints
        - Safe normalization
        
        Args:
            method_scores: Dict mapping method -> score
        
        Returns:
            Dict mapping method -> normalized weight
        """
        cfg = self.config.dynamic_weighting
        
        # Create ordered score array
        methods = self.methods if hasattr(self, 'methods') else list(method_scores.keys())
        scores = {m: method_scores.get(m, 0.5) for m in methods}
        
        # Apply dynamic semantic boost if enabled
        if self.config.enable_dynamic_weighting:
            semantic_score = scores.get('semantic', 0.0)
            if semantic_score > cfg.semantic_boost_cutoff:
                # Boost semantic weight
                scores['semantic'] = scores['semantic'] * cfg.semantic_boost_multiplier
        
        # Use safe normalization with softmax
        return normalize_weights_safe(
            scores,
            min_weight=cfg.min_weight,
            max_weight=cfg.max_weight,
            use_softmax=cfg.use_softmax_smoothing,
            temperature=cfg.temperature
        )
    
    def optimize_for_case_enhanced(self, 
                                   case_embedding: np.ndarray,
                                   case_metadata: dict) -> tuple:
        """
        Enhanced optimization with soft adjustments and safety checks.
        
        Wraps parent's optimize_for_case and applies config-driven
        enhancements.
        
        Args:
            case_embedding: Query embedding
            case_metadata: Query metadata
        
        Returns:
            (method_scores, weights, analysis)
        """
        # Get base optimization from parent
        method_scores, weights, analysis = self.optimize_for_case(
            case_embedding, 
            case_metadata
        )
        
        # Extract case features
        query_features = self._extract_query_features(case_metadata)
        
        # Apply soft adjustments to each method score
        adjusted_scores = {}
        for method in method_scores:
            original = method_scores[method]
            adjusted = self._adjust_method_score_with_config(
                method, 
                original,
                query_features
            )
            adjusted_scores[method] = adjusted
            
            if self.config.enable_dynamic_weighting:
                logger.debug(
                    f"{method}: {original:.3f} -> {adjusted:.3f} "
                    f"({adjusted-original:+.3f})"
                )
        
        # Recompute weights with safe normalization
        if self.config.enable_dynamic_weighting:
            adjusted_weights = self._compute_weights_from_scores_safe(
                adjusted_scores
            )
        else:
            adjusted_weights = weights
        
        # Update analysis
        analysis['enhanced'] = True
        analysis['query_features'] = query_features
        analysis['adjusted_method_scores'] = adjusted_scores
        analysis['adjusted_weights'] = adjusted_weights
        analysis['config_used'] = self.config.to_dict()
        
        return adjusted_scores, adjusted_weights, analysis


# Convenience function
def create_enhanced_optimizer(corpus_embeddings, corpus_metadata, 
                             config_path: str = None) -> EnhancedPerCaseOptimizer:
    """
    Create enhanced optimizer with automatic config loading.
    
    Args:
        corpus_embeddings: Corpus embeddings
        corpus_metadata: Corpus metadata
        config_path: Optional path to config JSON
    
    Returns:
        EnhancedPerCaseOptimizer instance
    """
    if config_path:
        config = SimilarityConfig.from_json(config_path)
    else:
        config = load_config()
    
    return EnhancedPerCaseOptimizer(
        corpus_embeddings,
        corpus_metadata,
        config=config
    )
