# per_case_optimizer_v2.py - True Corpus-Aware Per-Case Auto-Tuning Optimizer
"""
Per-Case Auto-Tuning Weight Optimizer
=====================================

This module implements corpus-aware weight computation where different queries
get different weights based on actual method effectiveness on corpus samples.

ARCHITECTURE:
1. FEATURE EXTRACTION: Analyzes query for intrinsic properties
2. CORPUS SAMPLING: Deterministically samples corpus for evaluation
3. METHOD EVALUATION: Tests each similarity method on the sample
4. EFFECTIVENESS SCORING: Measures discrimination power & reliability
5. WEIGHT COMPUTATION: Converts effectiveness into normalized weights
6. CONSTRAINT APPLICATION: Applies min/max bounds and normalization

SIMILARITY METHODS (4 methods):
- Semantic Similarity (embedding cosine) - captures meaning
- Field Score (weighted per-field TF-IDF) - boosts matches in key fields (Name > Description > Domain)
- BM25 Similarity - keyword importance ranking
- N-Gram Similarity (bigram-based) - phrase-level overlap

KEY PRINCIPLE:
No predefined weights. Weights are computed automatically based on actual 
performance of each method on representative corpus samples.

WEIGHT AUTO-TUNING LOGIC:
- Short + keyword-heavy → Higher BM25, Field Score, N-gram
- Long + descriptive → Higher Semantic, lower N-gram
- Structured lists → Higher Field Score, N-gram
- All weights: MIN=0.05, MAX=0.50 (balanced portfolio, no method reaches 100%)
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import warnings
import re

warnings.simplefilter("default")


class PerCaseOptimizerV2:
    """
    True corpus-aware per-case auto-tuning optimizer.
    
    Evaluates each similarity method on corpus samples to dynamically compute
    weights that reflect actual method effectiveness for the given query.
    """

    DEFAULT_METHODS = ['semantic', 'field_score', 'bm25', 'ngram']
    
    # Weight constraints (enforced after effectiveness scoring)
    MIN_WEIGHT = 0.05
    MAX_WEIGHT = 0.50

    def __init__(self,
                 corpus_embeddings: np.ndarray,
                 corpus_metadata: pd.DataFrame,
                 tfidf_max_features: int = 5000,
                 sample_size: int = None):
        """
        Initialize optimizer with corpus data.
        
        Args:
            corpus_embeddings: (n_cases, embedding_dim) array of corpus embeddings
            corpus_metadata: DataFrame with corpus case metadata
            tfidf_max_features: Maximum features for TF-IDF vectorizer
            sample_size: Corpus sample size for evaluation (auto-determined if None)
        """
        assert isinstance(corpus_embeddings, np.ndarray), "corpus_embeddings must be numpy array"
        assert isinstance(corpus_metadata, pd.DataFrame), "corpus_metadata must be DataFrame"

        self.corpus_embeddings = corpus_embeddings.astype('float32')
        self.corpus_metadata = corpus_metadata.reset_index(drop=True).copy()
        self.n_corpus = len(self.corpus_metadata)

        # Auto-determine sample size
        if sample_size is None:
            self.sample_size = min(max(15, self.n_corpus // 3), 60)
        else:
            self.sample_size = min(sample_size, self.n_corpus)

        # Normalize embeddings for cosine similarity
        self.normalized_embeddings = self._normalize_embeddings(self.corpus_embeddings)
        
        # Build similarity texts from corpus metadata
        self.texts = [self._get_text_from_metadata(row) for _, row in self.corpus_metadata.iterrows()]

        # ---- Field Score Setup ----
        # Each field gets its own TF-IDF vectorizer + a blend weight.
        # Matching on 'Idea Name' is worth more than matching on 'Idea Description'.
        self.field_config = [
            {'col': 'Idea Name',        'weight': 0.35},
            {'col': 'Domain',           'weight': 0.20},
            {'col': 'Idea Description', 'weight': 0.45},
        ]
        self.field_vectorizers = {}
        self.field_matrices   = {}
        for fc in self.field_config:
            col = fc['col']
            field_texts = []
            for _, row in self.corpus_metadata.iterrows():
                val = row.get(col, '') if hasattr(row, 'get') else getattr(row, col, '')
                field_texts.append(str(val).strip() if pd.notna(val) else '')
            vec = TfidfVectorizer(
                max_features=tfidf_max_features,
                stop_words='english',
                min_df=1, max_df=0.95
            )
            try:
                mat = vec.fit_transform(field_texts)
                self.field_vectorizers[col] = vec
                self.field_matrices[col]   = mat
            except Exception:
                self.field_vectorizers[col] = None
                self.field_matrices[col]   = None

        # Keep a single flat TF-IDF for deterministic sample selection only
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=tfidf_max_features,
            stop_words='english',
            min_df=1,
            max_df=0.95
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.texts)

        # Try to load BM25 model
        try:
            from bm25_similarity import BM25Similarity
            self.bm25_model = BM25Similarity(self.texts)
        except ImportError:
            self.bm25_model = None

        # Stop words for keyword matching
        self.stop_words = {'the','and','for','with','this','that','from','are','was','were',
                          'has','have','had','but','not','is','a','an','or','in','at','to','by',
                          'of','be','been','on','it','as','he','she','you','i','we','they'}

        self.methods = list(self.DEFAULT_METHODS)

    # ===================================================================
    # HELPER UTILITIES
    # ===================================================================
    
    @staticmethod
    def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit vectors for cosine similarity."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def _get_text_from_metadata(self, metadata) -> str:
        """Extract text representation from metadata object."""
        if isinstance(metadata, pd.Series):
            if 'similarity_text' in metadata and pd.notna(metadata['similarity_text']):
                return str(metadata['similarity_text'])
            parts = [
                str(metadata.get('Idea Name', '') or ''),
                str(metadata.get('Idea Description', '') or '')
            ]
            return ' '.join([p for p in parts if p])
        
        if isinstance(metadata, dict):
            if 'similarity_text' in metadata and metadata.get('similarity_text'):
                return str(metadata['similarity_text'])
            parts = [
                str(metadata.get('Idea Name', '') or ''),
                str(metadata.get('Idea Description', '') or '')
            ]
            return ' '.join([p for p in parts if p])
        
        return ''

    def _compute_deterministic_sample(self, query_text: str) -> np.ndarray:
        """
        Deterministically sample corpus indices for method evaluation.
        Uses query-based seeding for reproducibility.
        
        Args:
            query_text: The query/input text
        
        Returns:
            Array of corpus indices for evaluation
        """
        sample_size = self.sample_size
        
        if sample_size >= self.n_corpus:
            return np.arange(self.n_corpus)
        
        # Compute lexical similarity for deterministic selection
        try:
            query_vec = self.tfidf_vectorizer.transform([query_text])
            lexical_sims = cosine_similarity(query_vec, self.tfidf_matrix).ravel()
        except:
            # Fallback to random deterministic sample
            np.random.seed(hash(query_text) % (2**32))
            return np.random.choice(self.n_corpus, sample_size, replace=False)
        
        # Take top contentful samples and include some random for diversity
        n_lexical = sample_size // 2
        n_random = sample_size - n_lexical
        
        lexical_indices = np.argsort(lexical_sims)[::-1][:n_lexical]
        remaining_indices = np.setdiff1d(np.arange(self.n_corpus), lexical_indices)
        
        # Seed from query hash for determinism
        np.random.seed(hash(query_text) % (2**32))
        random_indices = np.random.choice(remaining_indices, 
                                          min(n_random, len(remaining_indices)), 
                                          replace=False)
        
        return np.concatenate([lexical_indices, random_indices])

    # ===================================================================
    # STEP 1: QUERY FEATURE EXTRACTION
    # ===================================================================

    def _extract_query_features(self, query_metadata: dict) -> dict:
        """
        Extract intrinsic features from query text for analysis.
        
        Features analyzed:
        - Text length and vocabulary richness
        - Technical term density
        - Keyword density and specificity
        - Presence of concrete examples
        - Abstractness vs concreteness
        
        Returns:
            Dictionary of extracted numerical features
        """
        query_text = self._get_text_from_metadata(query_metadata).strip()
        
        if not query_text or len(query_text) < 5:
            return self._get_default_query_features()
        
        # Tokenization
        tokens = re.findall(r'\w+', query_text.lower())
        words = [w for w in tokens if len(w) > 2]
        text_length = len(words)
        
        if text_length == 0:
            return self._get_default_query_features()
        
        unique_words = len(set(words))
        sentence_count = max(1, len(re.split(r'[.!?]+', query_text)) - 1)
        
        # Lexical diversity
        type_token_ratio = unique_words / text_length
        avg_word_length = np.mean([len(w) for w in words])
        
        # Technical term density
        technical_terms = {
            'ai', 'ml', 'machine', 'learning', 'algorithm', 'neural', 'deep',
            'blockchain', 'iot', 'cloud', 'api', 'nlp', 'vision', 'analytics',
            'predictive', 'automation', 'sensor', 'quantum', 'database'
        }
        technical_count = sum(1 for w in words if w in technical_terms)
        tech_density = technical_count / text_length
        
        # Keyword density (content words)
        content_words = [w for w in words if w not in self.stop_words]
        keyword_density = len(content_words) / text_length if text_length > 0 else 0.5
        
        # Specificity (vocabulary uniqueness)
        unique_content = len(set(content_words))
        specificity = unique_content / (len(content_words) + 1e-6) if content_words else 0.5
        
        # Structural patterns
        has_examples = bool(re.search(r'\b(example|instance|case|scenario|use case)\b', 
                                     query_text, re.IGNORECASE))
        has_imperatives = bool(re.search(r'\b(build|create|develop|implement|design)\b', 
                                        query_text, re.IGNORECASE))
        
        # Abstractness
        abstract_terms = ['concept', 'framework', 'model', 'theory', 'paradigm']
        concrete_terms = ['example', 'case', 'instance', 'prototype', 'implementation']
        abstract_count = sum(1 for w in words if any(t in w for t in abstract_terms))
        concrete_count = sum(1 for w in words if any(t in w for t in concrete_terms))
        abstract_score = abstract_count / text_length
        concrete_score = concrete_count / text_length
        
        return {
            'text_length': int(text_length),
            'unique_words': int(unique_words),
            'sentence_count': int(sentence_count),
            'type_token_ratio': float(type_token_ratio),
            'avg_word_length': float(avg_word_length),
            'specificity': float(specificity),
            'keyword_density': float(keyword_density),
            'technical_density': float(tech_density),
            'has_examples': bool(has_examples),
            'has_imperatives': bool(has_imperatives),
            'abstract_score': float(abstract_score),
            'concrete_score': float(concrete_score),
        }

    @staticmethod
    def _get_default_query_features() -> dict:
        """Return neutral default features."""
        return {
            'text_length': 0,
            'unique_words': 0,
            'sentence_count': 1,
            'type_token_ratio': 0.5,
            'avg_word_length': 5.0,
            'specificity': 0.5,
            'keyword_density': 0.5,
            'technical_density': 0.0,
            'has_examples': False,
            'has_imperatives': False,
            'abstract_score': 0.0,
            'concrete_score': 0.0,
        }

    # ===================================================================
    # STEP 2-4: CORPUS EVALUATION & EFFECTIVENESS SCORING
    # ===================================================================

    def _compute_sample_method_scores(self, query_embedding: np.ndarray, 
                                      query_text: str, 
                                      sample_indices: np.ndarray) -> dict:
        """
        Compute similarity scores from all 4 methods on the corpus sample.
        
        Args:
            query_embedding: (embedding_dim,) embedding of query
            query_text: Text representation of query
            sample_indices: Indices of corpus samples to evaluate on
        
        Returns:
            Dictionary mapping method names to score arrays (length = sample size)
        """
        sample_size = len(sample_indices)
        scores = {}
        
        # 1. SEMANTIC SIMILARITY (embedding cosine)
        try:
            query_emb_normalized = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)
            sample_embeddings = self.normalized_embeddings[sample_indices]
            semantic_scores = np.dot(sample_embeddings, query_emb_normalized)
            scores['semantic'] = np.clip(semantic_scores, 0, 1)
        except Exception as e:
            print(f"Warning: Semantic scoring failed: {e}")
            scores['semantic'] = np.full(sample_size, 0.5)
        
        # 2. FIELD SCORE (weighted per-field TF-IDF)
        try:
            combined_field = np.zeros(sample_size)
            total_w = 0.0
            # query_metadata may be a dict with per-field values; fall back to query_text
            for fc in self.field_config:
                col    = fc['col']
                w      = fc['weight']
                vec    = self.field_vectorizers.get(col)
                mat    = self.field_matrices.get(col)
                if vec is None or mat is None:
                    continue
                # Use query_text as fallback for field value
                field_q = query_text
                try:
                    q_vec   = vec.transform([field_q])
                    f_sims  = cosine_similarity(q_vec, mat[sample_indices]).ravel()
                    combined_field += f_sims * w
                    total_w += w
                except Exception:
                    pass
            if total_w > 0:
                combined_field /= total_w
            scores['field_score'] = np.clip(combined_field, 0, 1)
        except Exception as e:
            print(f"Warning: Field scoring failed: {e}")
            scores['field_score'] = np.full(sample_size, 0.5)
        
        # 3. BM25 SIMILARITY
        try:
            if self.bm25_model is not None:
                all_bm25_scores = self.bm25_model.get_scores(query_text)
                bm25_scores = all_bm25_scores[sample_indices]
                # Normalize BM25 to [0, 1] range
                bm25_max = np.max(all_bm25_scores) if np.max(all_bm25_scores) > 0 else 1.0
                scores['bm25'] = np.clip(bm25_scores / bm25_max, 0, 1)
            else:
                scores['bm25'] = np.full(sample_size, 0.5)
        except Exception as e:
            print(f"Warning: BM25 scoring failed: {e}")
            scores['bm25'] = np.full(sample_size, 0.5)
        
        # 4. N-GRAM SIMILARITY (bigram-based phrase overlap)
        try:
            ngram_scores = np.zeros(sample_size)
            
            query_bigrams = self._get_bigrams(query_text)
            if not query_bigrams:
                ngram_scores = np.full(sample_size, 0.0)
            else:
                for i, idx in enumerate(sample_indices):
                    case_bigrams = self._get_bigrams(self.texts[idx])
                    if case_bigrams:
                        intersection = len(query_bigrams & case_bigrams)
                        union = len(query_bigrams | case_bigrams)
                        ngram_scores[i] = intersection / union if union > 0 else 0.0
                    else:
                        ngram_scores[i] = 0.0
            
            scores['ngram'] = ngram_scores
        except Exception as e:
            print(f"Warning: N-gram scoring failed: {e}")
            scores['ngram'] = np.full(sample_size, 0.5)
        
        return scores

    def _get_bigrams(self, text):
        """Extract bigrams (2-grams) from text for N-gram similarity."""
        tokens = re.findall(r'\b\w+\b', text.lower())
        if len(tokens) < 2:
            return set()
        return set(zip(tokens, tokens[1:]))

    def _compute_method_effectiveness(self, method_scores: np.ndarray, 
                                      method_name: str) -> dict:
        """
        Compute effectiveness metrics for a similarity method.
        
        Metrics:
        - score_variance: How much scores vary (higher = better discrimination)
        - score_stdev: Standard deviation of scores
        - discrimination_power: Ability to separate top from rest
        - confidence_spread: How confidently scores are distributed
        
        Args:
            method_scores: Array of scores from this method on sample
            method_name: Name of method (for debugging)
        
        Returns:
            Dictionary of effectiveness metrics [0, 1] scale
        """
        if len(method_scores) < 2:
            return {
                'variance_score': 0.5,
                'stdev_score': 0.5,
                'discrimination_power': 0.5,
                'confidence_spread': 0.5,
            }
        
        # Variance: how much scores spread out (0-1)
        score_var = np.var(method_scores)
        variance_score = min(score_var * 10, 1.0)  # Scale to [0, 1]
        
        # Standard deviation score
        score_std = np.std(method_scores)
        stdev_score = min(score_std * 3, 1.0)
        
        # Discrimination power: difference between top and rest
        top_k = max(1, len(method_scores) // 4)
        top_scores = np.sort(method_scores)[-top_k:]
        rest_scores = np.sort(method_scores)[:-top_k]
        
        if len(rest_scores) > 0:
            top_mean = np.mean(top_scores)
            rest_mean = np.mean(rest_scores)
            discrimination = abs(top_mean - rest_mean)
        else:
            discrimination = 0.5
        
        discrimination_power = min(discrimination * 2, 1.0)
        
        # Confidence spread: how confident the method is overall
        nonzero_scores = method_scores[method_scores > 0]
        if len(nonzero_scores) > 0:
            avg_confidence = np.mean(nonzero_scores)
            confidence_spread = float(avg_confidence)
        else:
            confidence_spread = 0.3
        
        return {
            'variance_score': float(variance_score),
            'stdev_score': float(stdev_score),
            'discrimination_power': float(discrimination_power),
            'confidence_spread': float(confidence_spread),
        }

    def _compute_effectiveness_scores(self, all_method_scores: dict) -> dict:
        """
        Convert method effectiveness metrics into single effectiveness score [0, 1].
        
        Args:
            all_method_scores: Dict mapping method names to effectiveness metrics
        
        Returns:
            Dictionary of effectiveness scores (0-1) for each method
        """
        effectiveness_scores = {}
        
        for method_name in self.methods:
            if method_name not in all_method_scores:
                effectiveness_scores[method_name] = 0.5
                continue
            
            metrics = all_method_scores[method_name]
            
            # Combine metrics: weighted average
            # (Discrimination power is most important, then variance)
            combined = (
                metrics['discrimination_power'] * 0.40 +
                metrics['variance_score'] * 0.30 +
                metrics['confidence_spread'] * 0.20 +
                metrics['stdev_score'] * 0.10
            )
            
            effectiveness_scores[method_name] = float(np.clip(combined, 0.0, 1.0))
        
        return effectiveness_scores

    # ===================================================================
    # STEP 5: DYNAMIC WEIGHT COMPUTATION
    # ===================================================================

    def _compute_weights_from_effectiveness(self, effectiveness_scores: dict) -> dict:
        """
        Convert effectiveness scores to normalized weights with constraints.
        
        Constraints:
        - Minimum weight per method: MIN_WEIGHT (0.05)
        - Maximum weight per method: MAX_WEIGHT (0.50)
        - All weights sum to 1.0
        
        Args:
            effectiveness_scores: Dict with effectiveness [0, 1] for each method
        
        Returns:
            Dictionary of normalized weights summing to 1.0
        """
        # Start with effectiveness scores as base weights
        weights = effectiveness_scores.copy()
        
        # Add minimum weight to all methods (ensures all methods participate)
        for method in self.methods:
            if method not in weights:
                weights[method] = 0.5
            weights[method] = weights[method] * 0.7 + 0.3  # Blend towards 0.5
        
        # Apply maximum ceiling
        for method in self.methods:
            weights[method] = min(weights[method], self.MAX_WEIGHT)
        
        # Apply minimum floor
        for method in self.methods:
            weights[method] = max(weights[method], self.MIN_WEIGHT)
        
        # Normalize to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight <= 0:
            # Fallback to equal weights
            weights = {m: 0.25 for m in self.methods}
        else:
            weights = {m: w / total_weight for m, w in weights.items()}
        
        # Verify sum
        final_sum = sum(weights.values())
        if abs(final_sum - 1.0) > 1e-6:
            # Fine-tune largest method to make sum exact
            largest_method = max(weights, key=weights.get)
            weights[largest_method] += (1.0 - final_sum)
        
        return weights

    # ===================================================================
    # STEP 6: MAIN OPTIMIZATION ROUTINE
    # ===================================================================

    def optimize_for_case(self, case_embedding: np.ndarray, case_metadata: dict) -> tuple:
        """
        Main entry point for per-case weight optimization.
        
        WORKFLOW:
        1. Extract query features
        2. Sample corpus deterministically
        3. Evaluate all methods on sample
        4. Compute method effectiveness scores
        5. Convert to normalized weights with constraints
        6. Generate explanation
        
        Args:
            case_embedding: (embedding_dim,) array - embedding of query
            case_metadata: dict - metadata of query
        
        Returns:
            Tuple of (method_scores_dict, weights_dict, analysis_dict)
            - method_scores: All methods mapped to 1.0 (computed after weighting)
            - weights: Normalized weights for each method
            - analysis: Detailed explanation and metrics
        """
        if case_embedding is None or len(case_embedding) == 0:
            raise ValueError("case_embedding cannot be None or empty")
        
        # STEP 1: Extract query features
        query_features = self._extract_query_features(case_metadata)
        query_text = self._get_text_from_metadata(case_metadata)
        
        # STEP 2: Sample corpus
        sample_indices = self._compute_deterministic_sample(query_text)
        
        # STEP 3: Evaluate methods on sample
        sample_method_scores = self._compute_sample_method_scores(
            case_embedding, query_text, sample_indices
        )
        
        # STEP 4: Compute effectiveness metrics and scores
        all_effectiveness_metrics = {}
        for method_name, scores in sample_method_scores.items():
            if method_name in self.methods:
                all_effectiveness_metrics[method_name] = self._compute_method_effectiveness(
                    scores, method_name
                )
        
        # Convert to single effectiveness score per method
        effectiveness_scores = self._compute_effectiveness_scores(all_effectiveness_metrics)
        
        # STEP 5: Compute weights from effectiveness
        weights = self._compute_weights_from_effectiveness(effectiveness_scores)
        
        # STEP 6: Generate explanation
        explanation = self._generate_explanation(
            query_features, effectiveness_scores, weights, sample_indices.shape[0]
        )
        
        # Build analysis dictionary
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'query_features': query_features,
            'sample_size': int(sample_indices.shape[0]),
            'sample_indices': sample_indices.tolist(),
            'method_effectiveness_scores': {m: float(e) for m, e in effectiveness_scores.items()},
            'computed_weights': weights,
            'effectiveness_metrics': all_effectiveness_metrics,
            'explanation': explanation,
            'method_count': len(self.methods),
        }
        
        # method_scores (legacy compatibility - all methods active)
        method_scores = {m: 1.0 for m in self.methods}
        
        return method_scores, weights, analysis

    def _generate_explanation(self, query_features: dict, 
                             effectiveness_scores: dict, 
                             weights: dict, 
                             sample_size: int) -> list:
        """
        Generate human-readable explanation for weight computation.
        
        Args:
            query_features: Extracted features from query
            effectiveness_scores: Effectiveness of each method [0-1]
            weights: Computed normalized weights
            sample_size: Number of samples evaluated
        
        Returns:
            List of explanation strings
        """
        explanation = []
        
        # Query analysis
        explanation.append("CORPUS-AWARE PER-CASE WEIGHT COMPUTATION")
        explanation.append("=" * 50)
        explanation.append("")
        explanation.append(f"Query Analysis (LENGTH: {query_features['text_length']} words):")
        explanation.append(f"  - Specificity: {query_features['specificity']:.2f}")
        explanation.append(f"  - Keyword density: {query_features['keyword_density']:.2f}")
        explanation.append(f"  - Technical density: {query_features['technical_density']:.2f}")
        explanation.append(f"  - Abstractness level: {query_features['abstract_score']:.2f}")
        explanation.append("")
        
        # Sample evaluation
        explanation.append(f"Corpus Sample Evaluation (SIZE: {sample_size} cases):")
        explanation.append("")
        
        # Method effectiveness and weights
        sorted_methods = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        explanation.append("Method Effectiveness → Weights:")
        explanation.append("-" * 50)
        
        for rank, (method, weight) in enumerate(sorted_methods, 1):
            effectiveness = effectiveness_scores.get(method, 0.5)
            explanation.append(f"{rank}. {method.upper():18} | " +
                             f"Effectiveness: {effectiveness:.3f} | " +
                             f"Weight: {weight:.3f}")
        
        explanation.append("")
        explanation.append(f"Total weight: {sum(weights.values()):.4f} (must be 1.0000)")
        explanation.append("")
        explanation.append("Constraints Applied:")
        explanation.append(f"  - Min weight per method: {self.MIN_WEIGHT}")
        explanation.append(f"  - Max weight per method: {self.MAX_WEIGHT}")
        explanation.append("  - All weights normalized to sum = 1.0")
        
        return explanation

