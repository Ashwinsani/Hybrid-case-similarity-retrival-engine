# config.py - Centralized Configuration for Rephrased Content Matching
"""
Production-ready configuration system for similarity matching with
rephrased content detection and adaptive scoring.

All thresholds, boosts, and parameters are externalized here for easy tuning
without modifying core logic.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List
import json
import os


@dataclass
class SemanticAdjustmentConfig:
    """Semantic similarity adjustment parameters"""
    min_technical_density_boost: float = 0.15  # Boost at technical_term_density > 0.2
    min_technical_density: float = 0.2
    
    description_complexity_boost: float = 0.12  # Boost at description_complexity > 50
    min_description_complexity: float = 50
    
    long_text_boost: float = 0.10  # Boost for text_length > 80
    min_text_length_long: float = 80


@dataclass
class LexicalAdjustmentConfig:
    """Lexical similarity adjustment parameters - reduced for rephrased content"""
    concrete_examples_boost: float = 0.08  # Reduced from 0.15
    short_text_boost: float = 0.05  # Reduced from 0.08
    short_text_threshold: float = 100
    
    technical_penalty: float = -0.10  # Penalize technical terms in lexical
    technical_penalty_threshold: float = 0.15


@dataclass
class CrossEncoderAdjustmentConfig:
    """Cross-encoder/semantic ranking adjustments - great for paraphrases"""
    high_complexity_boost: float = 0.25  # Boost at description_complexity > 50
    min_high_complexity: float = 50
    
    abstract_concepts_boost: float = 0.12  # Increased from 0.08
    technical_boost: float = 0.10
    technical_threshold: float = 0.1


@dataclass
class BM25AdjustmentConfig:
    """BM25 adjustment parameters"""
    short_text_boost: float = 0.15  # Reduced from 0.20
    short_text_threshold: float = 30
    
    vocabulary_specificity_boost: float = 0.15  # Increased from 0.10
    vocabulary_specificity_threshold: float = 0.7
    
    technical_boost: float = 0.10
    technical_threshold: float = 0.15


@dataclass
class RerankingConfig:
    """Rephrased content detection and re-ranking parameters"""
    # Pattern: semantic high + lexical low + BM25 high = likely rephrased
    semantic_threshold: float = 0.60  # Min semantic score
    lexical_threshold: float = 0.30   # Max lexical score for paraphrase detection
    bm25_threshold: float = 0.50      # Min BM25 score
    
    boost_multiplier: float = 1.20    # Boost final score by this factor (20% boost)
    boost_max_cap: float = 1.0        # Cap at 1.0 after boosting
    
    # Alternative: use additive boost instead of multiplicative
    use_additive_boost: bool = False
    additive_boost_amount: float = 0.15  # Add this to score instead of multiplying
    
    # Logging
    verbose: bool = True
    log_rephrased_detections: bool = True


@dataclass
class AdaptiveThresholdConfig:
    """Adaptive threshold configuration based on score distribution"""
    base_threshold: float = 0.30          # Default threshold
    high_semantic_cutoff: float = 0.70    # If max_semantic > this, lower threshold
    threshold_reduction_factor: float = 0.80  # Multiply threshold by this factor
    min_threshold: float = 0.20           # Floor for adaptive threshold


@dataclass
class DynamicWeightingConfig:
    """Parameters for dynamic weight computation"""
    # Boost semantic when it's already scoring well
    semantic_boost_cutoff: float = 0.60
    semantic_boost_multiplier: float = 1.20
    
    # Min/max weight constraints
    min_weight: float = 0.05
    max_weight: float = 0.50
    
    # Smoothing for weight distribution
    use_softmax_smoothing: bool = True
    temperature: float = 1.0  # Temperature for softmax (lower = sharper distribution)


@dataclass
class ValidationConfig:
    """Configuration for validation and testing"""
    # Ground truth pairs format: list of (rephrased_text, original_text) tuples
    run_validation_on_startup: bool = False
    validation_data_path: str = "validation_data.json"
    
    # Thresholds for validation success
    min_rank_for_original: int = 5  # Original should rank in top-5
    min_score_for_original: float = 0.60  # Original should score >= 0.60
    
    # Logging
    log_validation_results: bool = True


@dataclass
class SimilarityConfig:
    """Main configuration container"""
    # Adjustment configs
    semantic: SemanticAdjustmentConfig = field(default_factory=SemanticAdjustmentConfig)
    lexical: LexicalAdjustmentConfig = field(default_factory=LexicalAdjustmentConfig)
    cross_encoder: CrossEncoderAdjustmentConfig = field(default_factory=CrossEncoderAdjustmentConfig)
    bm25: BM25AdjustmentConfig = field(default_factory=BM25AdjustmentConfig)
    
    # Feature detection
    reranking: RerankingConfig = field(default_factory=RerankingConfig)
    adaptive_threshold: AdaptiveThresholdConfig = field(default_factory=AdaptiveThresholdConfig)
    dynamic_weighting: DynamicWeightingConfig = field(default_factory=DynamicWeightingConfig)
    
    # Validation
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    
    # Global flags
    enable_reranking: bool = True
    enable_adaptive_threshold: bool = True
    enable_dynamic_weighting: bool = True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self, path: str):
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✅ Configuration saved to {path}")
    
    @staticmethod
    def from_json(path: str) -> 'SimilarityConfig':
        """Load configuration from JSON file"""
        if not os.path.exists(path):
            print(f"⚠️ Config file not found: {path}, using defaults")
            return SimilarityConfig()
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Reconstruct nested dataclass objects
        config = SimilarityConfig()
        for key, value in data.items():
            if hasattr(config, key) and isinstance(value, dict):
                # Recursively reconstruct nested dataclasses
                nested_type = type(getattr(config, key))
                setattr(config, key, nested_type(**value))
            elif hasattr(config, key):
                setattr(config, key, value)
        
        return config


# Default instance
DEFAULT_CONFIG = SimilarityConfig()
 
def load_config(config_path: str = None) -> SimilarityConfig:
    """
    Load configuration from file or use defaults
    
    Args:
        config_path: Optional path to config JSON file
    
    Returns:
        SimilarityConfig instance
    """
    if config_path and os.path.exists(config_path):
        return SimilarityConfig.from_json(config_path)
    
    # Check for config.json in current directory
    if os.path.exists('similarity_config.json'):
        return SimilarityConfig.from_json('similarity_config.json')
    
    return DEFAULT_CONFIG


def save_default_config(output_path: str = 'similarity_config.json'):
    """Save default configuration to file for reference"""
    DEFAULT_CONFIG.to_json(output_path)
    print(f"📝 Default configuration saved to {output_path}")
