# multi_similarity_engine_enhanced.py - Enhanced engine with safe re-ranking
"""
Enhanced multi-similarity engine with:
- Safe re-ranking for rephrased content
- Adaptive thresholding
- Detailed logging and validation
"""

# At the top of multi_similarity_engine_enhanced.py
import numpy as np
import logging
from multi_similarity_engine import EnhancedCosineSimilarityMatcher
from config import SimilarityConfig, load_config
from scoring_utils import safe_rerank, compute_adaptive_threshold, normalize_scores, calculate_heading_similarity, adaptive_alpha

logger = logging.getLogger(__name__)

class EnhancedSimilarityMatcher(EnhancedCosineSimilarityMatcher):
    """
    Extended multi-similarity matcher with:
    - Configuration-driven parameters
    - Safe rephrased content re-ranking
    - Adaptive thresholding
    - Enhanced logging
    """
    
    def __init__(self, corpus_dir="case_embeddings", config: SimilarityConfig = None):
        """
        Initialize enhanced matcher.
        
        Args:
            corpus_dir: Path to corpus embeddings directory
            config: SimilarityConfig instance (loads from file if None)
        """
        super().__init__(corpus_dir)
        
        # Load or use provided config
        self.config = config if config is not None else load_config()
        
        logger.info(f"✅ Enhanced matcher initialized")
        logger.debug(f"   Reranking enabled: {self.config.enable_reranking}")
        logger.debug(f"   Adaptive threshold enabled: {self.config.enable_adaptive_threshold}")
    
    # ===== FIRST: The blended reranking method =====
    def rerank_with_blended_scores(self, 
                                candidates, 
                                query_text,
                                query_heading,
                                base_alpha=0.65,
                                heading_bonus_weight=0.05,
                                top_k=10):
        """
        Step 1: Normalize scores
        Step 2: Combine (α * CE + (1-α) * Retrieval)
        Step 3: Add heading bonus
        Step 4: Sort
        
        Args:
            candidates: List of candidate cases with retrieval_scores
            query_text: Full query text for CE
            query_heading: Query heading for similarity calculation
            base_alpha: Base weight for CE vs retrieval (0.65 = 65% CE, 35% retrieval)
            heading_bonus_weight: Max heading bonus (0.05 = 5%)
            top_k: Number of results to return
        """
        import numpy as np
        from scoring_utils import normalize_scores, calculate_heading_similarity, adaptive_alpha
        
        if not candidates:
            return []
        
        print(f"\n🔄 RERANKING with Blended Scores")
        print(f"   Base alpha: {base_alpha} (CE), {1-base_alpha} (Retrieval)")
        print(f"   Heading bonus max: {heading_bonus_weight*100}%")
        
        # Step 1: Get CE scores for all candidates
        pairs = [[query_text, c.get('similarity_text', '')] for c in candidates]
        ce_scores = self.cross_encoder.predict(pairs)
        
        # Extract retrieval scores
        retrieval_scores = [c.get('final_score', 0) for c in candidates]
        
        # Step 1: Normalize both score sets
        ce_norm = normalize_scores(ce_scores)
        retrieval_norm = normalize_scores(retrieval_scores)
        
        # Calculate heading similarities for each candidate
        heading_sims = []
        for candidate in candidates:
            case_heading = candidate.get('case_name', '') or candidate.get('Idea Name', '')
            sim = calculate_heading_similarity(query_heading, case_heading)
            heading_sims.append(sim)
        
        # Calculate adaptive alpha for each candidate (based on heading match)
        alphas = [adaptive_alpha(sim, base_alpha) for sim in heading_sims]
        
        # Step 2: Combine with candidate-specific alpha
        combined_scores = [
            alphas[i] * ce_norm[i] + (1 - alphas[i]) * retrieval_norm[i]
            for i in range(len(candidates))
        ]
        
        # Step 3: Add heading bonus (additive)
        heading_bonuses = [sim * heading_bonus_weight for sim in heading_sims]
        final_scores = [combined_scores[i] + heading_bonuses[i] for i in range(len(candidates))]
        
        # Step 4: Sort by final score
        ranked_indices = np.argsort(final_scores)[::-1]
        
        # Create reranked list with debug info
        reranked = []
        for rank, idx in enumerate(ranked_indices[:top_k]):
            candidate = candidates[idx].copy()
            candidate.update({
                'rank': rank + 1,
                'original_rank': idx + 1,
                'ce_score_raw': float(ce_scores[idx]),
                'ce_score_norm': float(ce_norm[idx]),
                'retrieval_score_raw': float(retrieval_scores[idx]),
                'retrieval_score_norm': float(retrieval_norm[idx]),
                'alpha_used': float(alphas[idx]),
                'heading_similarity': float(heading_sims[idx]),
                'heading_bonus': float(heading_bonuses[idx]),
                'combined_score': float(combined_scores[idx]),
                'final_score': float(final_scores[idx])
            })
            reranked.append(candidate)
        
        # Print debug info
        print(f"\n📊 RERANKING RESULTS:")
        print(f"{'Rank':<5} {'Case Name':<30} {'CE':<6} {'Ret':<6} {'α':<5} {'Head':<6} {'Final':<6}")
        print("-" * 70)
        
        for r in reranked[:5]:
            print(f"{r['rank']:<5} {r['case_name'][:28]:<30} "
                  f"{r['ce_score_norm']:.3f} {r['retrieval_score_norm']:.3f} "
                  f"{r['alpha_used']:.3f} {r['heading_similarity']:.3f} {r['final_score']:.3f}")
        
        return reranked

    # ===== THEN: The enhanced_find_similar_cases method =====
    def enhanced_find_similar_cases(self, 
                                   user_folder=None, 
                                   top_k=55, 
                                   similarity_threshold=0.3) -> dict:
        """
        Enhanced case finding with rephrased content detection and re-ranking.
        
        WORKFLOW:
        1. Run base similarity search (with skip_rerank=True)
        2. Apply blended reranking with adaptive alpha and heading bonus
        3. Apply adaptive threshold
        4. Return top-k ranked cases
        
        Args:
            user_folder: User embedding folder path
            top_k: Number of top cases to return
            similarity_threshold: Base threshold for inclusion
        
        Returns:
            Dict with similar cases and detailed scoring info
        """
        # Get base results from parent - WITH skip_rerank=True to prevent double reranking
        results = super().enhanced_find_similar_cases(
            user_folder=user_folder,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            skip_rerank=True  # ← THIS IS CRITICAL!
        )
        
        if not results or 'similar_cases' not in results:
            return results
        
        # Extract score data for re-ranking
        try:
            all_scores = results.get('_all_scores', {})
            all_final_scores = np.array(results.get('_all_final_scores', []))
            
            if len(all_final_scores) == 0:
                logger.warning("No score data available for re-ranking")
                return results
            
            # ===== Get query information for blended reranking =====
            query_text = results.get('user_input', {}).get('similarity_text', '')
            if not query_text:
                query_text = results.get('user_input', {}).get('similarity_text_preview', '')
            
            query_heading = results.get('user_input', {}).get('idea_name', '')
            
            # Store original similar cases before reranking
            original_similar_cases = results['similar_cases'].copy() if results['similar_cases'] else []
            
            print(f"\n🔍 DEBUG - Enhanced Reranking:")
            print(f"   Query Heading: {query_heading}")
            print(f"   Query Text Preview: {query_text[:100]}...")
            print(f"   Original cases count: {len(original_similar_cases)}")
            print(f"   Cross-encoder available: {self.cross_encoder is not None}")
            print(f"   Reranking enabled in config: {self.config.enable_reranking}")
            
            # ===== Apply your blended reranking approach =====
            if self.config.enable_reranking and self.cross_encoder and original_similar_cases:
                print("✅ APPLYING BLENDED RERANKING...")
                
                # Use the new reranking method
                reranked_cases = self.rerank_with_blended_scores(
                    candidates=original_similar_cases,
                    query_text=query_text,
                    query_heading=query_heading,
                    base_alpha=0.65,           # 65% CE, 35% retrieval default
                    heading_bonus_weight=0.05,  # Max 5% bonus from heading
                    top_k=top_k
                )
                
                # Update results with reranked cases
                results['similar_cases'] = reranked_cases
                results['_reranking_method'] = 'blended_with_adaptive_alpha_and_heading_bonus'
                results['_reranking_applied'] = True
                
                logger.info(f"✅ Blended reranking complete. Top case: {reranked_cases[0]['case_name'] if reranked_cases else 'None'}")
                
            else:
                print("❌ Blended reranking NOT applied:")
                if not self.config.enable_reranking:
                    print("   - Reranking disabled in config")
                if not self.cross_encoder:
                    print("   - Cross-encoder not available")
                if not original_similar_cases:
                    print("   - No original cases found")
            
            # Apply adaptive threshold if enabled
            if self.config.enable_adaptive_threshold and all_scores:
                semantic_scores = np.array(all_scores.get('semantic', []))
                adaptive_threshold = compute_adaptive_threshold(
                    semantic_scores,
                    base_threshold=self.config.adaptive_threshold.base_threshold,
                    high_semantic_cutoff=self.config.adaptive_threshold.high_semantic_cutoff,
                    reduction_factor=self.config.adaptive_threshold.threshold_reduction_factor,
                    min_threshold=self.config.adaptive_threshold.min_threshold
                )
                
                if adaptive_threshold < similarity_threshold:
                    logger.info(
                        f"🎯 Adaptive threshold: {similarity_threshold:.3f} -> "
                        f"{adaptive_threshold:.3f}"
                    )
                    results['adaptive_threshold_used'] = adaptive_threshold
                    results['threshold_reason'] = (
                        f"Max semantic score {np.max(semantic_scores):.3f} > "
                        f"{self.config.adaptive_threshold.high_semantic_cutoff}"
                    )
                    
                    # Filter cases by adaptive threshold
                    filtered_cases = [
                        c for c in results['similar_cases']
                        if c.get('final_score', 0) >= adaptive_threshold
                    ]
                    results['similar_cases'] = filtered_cases[:top_k]
            
            # Add configuration info to results
            results['config_used'] = {
                'reranking_enabled': self.config.enable_reranking,
                'adaptive_threshold_enabled': self.config.enable_adaptive_threshold,
                'reranking_config': {
                    'semantic_threshold': self.config.reranking.semantic_threshold,
                    'lexical_threshold': self.config.reranking.lexical_threshold,
                    'bm25_threshold': self.config.reranking.bm25_threshold,
                    'boost_multiplier': self.config.reranking.boost_multiplier,
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error during re-ranking: {e}", exc_info=True)
            logger.warning("⚠️ Returning original results without re-ranking")
        
        return results
    
    def validate_match_quality(self, results: dict, expected_case_name: str = None) -> dict:
        """
        Validate match quality against expectations.
        
        Args:
            results: Results dict from enhanced_find_similar_cases
            expected_case_name: Optional expected top case name
        
        Returns:
            Dict with validation details
        """
        validation = {
            'total_matches': len(results.get('similar_cases', [])),
            'expected_case_found': False,
            'expected_case_rank': -1,
            'expected_case_score': 0.0,
            'average_top_5_score': 0.0,
            'validation_passed': False
        }
        
        similar_cases = results.get('similar_cases', [])
        
        if not similar_cases:
            logger.warning("⚠️ No similar cases found")
            return validation
        
        # Average score of top-5
        top_5_scores = [c.get('final_score', 0) for c in similar_cases[:5]]
        validation['average_top_5_score'] = float(np.mean(top_5_scores)) if top_5_scores else 0.0
        
        # Check for expected case
        if expected_case_name:
            for rank, case in enumerate(similar_cases):
                if expected_case_name.lower() in case.get('case_name', '').lower():
                    validation['expected_case_found'] = True
                    validation['expected_case_rank'] = rank + 1
                    validation['expected_case_score'] = case.get('final_score', 0.0)
                    break
        
        # Validation pass: expected in top-5 with good score
        cfg = self.config.validation
        if expected_case_name:
            validation['validation_passed'] = (
                validation['expected_case_rank'] <= cfg.min_rank_for_original and
                validation['expected_case_score'] >= cfg.min_score_for_original
            )
        else:
            # No expectation: just check we have matches with decent scores
            validation['validation_passed'] = (
                len(similar_cases) > 0 and
                validation['average_top_5_score'] >= cfg.min_score_for_original
            )
        
        if cfg.log_validation_results:
            logger.info(f"✅ Match validation:")
            logger.info(f"   Total matches: {validation['total_matches']}")
            logger.info(f"   Top-5 avg score: {validation['average_top_5_score']:.3f}")
            if expected_case_name:
                logger.info(
                    f"   Expected case: rank={validation['expected_case_rank']}, "
                    f"score={validation['expected_case_score']:.3f}, "
                    f"found={validation['expected_case_found']}"
                )
            logger.info(f"   Validation: {'PASS' if validation['validation_passed'] else 'FAIL'}")
        
        return validation


def create_enhanced_matcher(corpus_dir="case_embeddings",
                           config_path: str = None) -> EnhancedSimilarityMatcher:
    """
    Create enhanced matcher with automatic config loading.
    
    Args:
        corpus_dir: Corpus directory path
        config_path: Optional config JSON path
    
    Returns:
        EnhancedSimilarityMatcher instance
    """
    if config_path:
        config = SimilarityConfig.from_json(config_path)
    else:
        config = load_config()
    
    return EnhancedSimilarityMatcher(corpus_dir, config=config)