# validation_utils.py - Testing and validation utilities
"""
Validation and testing utilities for similarity matching system.
Includes:
- Rephrased content test cases
- A/B testing framework
- Metrics computation
- Offline evaluation
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Single test case: (rephrased_text, original_text, expected_min_rank, expected_min_score)"""
    rephrased: str
    original: str
    expected_min_rank: int = 5
    expected_min_score: float = 0.60
    domain: str = "general"
    description: str = ""


class RephraseTestDataset:
    """
    Collection of test cases for rephrased content matching.
    """
    
    def __init__(self):
        self.test_cases: List[TestCase] = [
            # Agricultural domain
            TestCase(
                rephrased=(
                    "AI-Driven Smart Agriculture Forecasting System\n"
                    "This project focuses on developing an AI-powered prediction system that "
                    "analyzes environmental data, soil parameters, and market trends to forecast "
                    "crop yields, pest risks, and weather impacts. The system integrates data from "
                    "IoT sensors, satellite imagery, and historical yield records to help farmers "
                    "and policymakers make data-driven agricultural decisions."
                ),
                original=(
                    "AI-based agricultural forecasting\n"
                    "Develop an AI-powered model to predict crop yields across districts using "
                    "historical agricultural data, weather patterns, and satellite imagery."
                ),
                expected_min_rank=3,
                expected_min_score=0.70,
                domain="agriculture",
                description="Environmental data analysis with different terminology"
            ),
            # Healthcare domain
            TestCase(
                rephrased=(
                    "Comprehensive AI-Driven Diagnostic Platform\n"
                    "An intelligent system that leverages deep learning and computer vision "
                    "technologies to analyze medical imaging data, including X-rays, CT scans, "
                    "and MRI images, for early disease detection and patient risk stratification."
                ),
                original=(
                    "AI-powered medical imaging analysis\n"
                    "Build a machine learning model that analyzes medical images to detect diseases."
                ),
                expected_min_rank=5,
                expected_min_score=0.65,
                domain="healthcare",
                description="Same concept, different implementation details"
            ),
            # Environmental domain
            TestCase(
                rephrased=(
                    "Advanced Environmental Monitoring and Prediction System\n"
                    "Deploy a comprehensive platform utilizing satellite remote sensing, "
                    "IoT sensor networks, and advanced analytics to monitor environmental changes, "
                    "predict climate impacts, and support evidence-based policy decisions."
                ),
                original=(
                    "Environmental data monitoring\n"
                    "Track and analyze environmental metrics using sensor networks and satellites."
                ),
                expected_min_rank=5,
                expected_min_score=0.60,
                domain="environment",
                description="Paraphrased with additional technical details"
            ),
            # Financial domain
            TestCase(
                rephrased=(
                    "Intelligent Fraud Detection and Risk Assessment Framework\n"
                    "Develop a sophisticated ML-based system for real-time transaction monitoring, "
                    "anomaly detection, and fraud pattern recognition using ensemble methods "
                    "and graph neural networks for network analysis."
                ),
                original=(
                    "Fraud detection system\n"
                    "Create a machine learning model to detect fraudulent transactions."
                ),
                expected_min_rank=5,
                expected_min_score=0.65,
                domain="finance",
                description="Similar goal, technical vocabulary expansion"
            ),
        ]
    
    def save(self, path: str):
        """Save test cases to JSON"""
        data = [
            {
                'rephrased': tc.rephrased,
                'original': tc.original,
                'expected_min_rank': tc.expected_min_rank,
                'expected_min_score': tc.expected_min_score,
                'domain': tc.domain,
                'description': tc.description
            }
            for tc in self.test_cases
        ]
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"✅ Test dataset saved to {path}")
    
    @staticmethod
    def load(path: str) -> 'RephraseTestDataset':
        """Load test cases from JSON"""
        dataset = RephraseTestDataset()
        dataset.test_cases = []
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        for item in data:
            dataset.test_cases.append(TestCase(
                rephrased=item['rephrased'],
                original=item['original'],
                expected_min_rank=item.get('expected_min_rank', 5),
                expected_min_score=item.get('expected_min_score', 0.60),
                domain=item.get('domain', 'general'),
                description=item.get('description', '')
            ))
        
        logger.info(f"✅ Loaded {len(dataset.test_cases)} test cases from {path}")
        return dataset


@dataclass
class EvaluationMetrics:
    """Metrics for a single test case evaluation"""
    test_id: int
    domain: str
    description: str
    rephrased: str
    original: str
    
    # Results
    original_rank: int  # Rank of original in results (1-indexed)
    original_score: float  # Score of original
    top_1_score: float  # Score of top result
    top_5_avg_score: float  # Average score of top 5
    
    # Judgements
    rank_passed: bool  # Did original rank <= expected?
    score_passed: bool  # Did original score >= expected?
    overall_passed: bool  # Both rank and score passed?
    
    # Metadata
    reranking_boosted: bool = False
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    bm25_score: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'test_id': self.test_id,
            'domain': self.domain,
            'description': self.description,
            'original_rank': self.original_rank,
            'original_score': float(self.original_score),
            'top_1_score': float(self.top_1_score),
            'top_5_avg_score': float(self.top_5_avg_score),
            'rank_passed': self.rank_passed,
            'score_passed': self.score_passed,
            'overall_passed': self.overall_passed,
            'reranking_boosted': self.reranking_boosted,
            'semantic_score': float(self.semantic_score),
            'lexical_score': float(self.lexical_score),
            'bm25_score': float(self.bm25_score),
        }


class OfflineEvaluator:
    """
    Offline evaluation framework for A/B testing.
    Compare old vs new system on test cases.
    """
    
    def __init__(self, test_dataset: RephraseTestDataset):
        self.test_dataset = test_dataset
        self.results_v1 = []  # Old system results
        self.results_v2 = []  # New system results
    
    def evaluate_case(self,
                     test_id: int,
                     test_case: TestCase,
                     search_results: Dict,
                     expected_min_rank: Optional[int] = None,
                     expected_min_score: Optional[float] = None) -> EvaluationMetrics:
        """
        Evaluate a single test case against search results.
        
        Args:
            test_id: ID of test case
            test_case: TestCase object
            search_results: Dict with 'similar_cases' from similarity matcher
            expected_min_rank: Override expected rank (uses test_case if None)
            expected_min_score: Override expected score (uses test_case if None)
        
        Returns:
            EvaluationMetrics object
        """
        expected_rank = expected_min_rank or test_case.expected_min_rank
        expected_score = expected_min_score or test_case.expected_min_score
        
        similar_cases = search_results.get('similar_cases', [])
        
        # Find original case in results
        original_rank = -1
        original_score = 0.0
        
        for rank, case in enumerate(similar_cases):
            # Match by case name (simplified)
            if test_case.original[:30].lower() in case.get('case_name', '').lower():
                original_rank = rank + 1
                original_score = case.get('final_score', 0.0)
                break
        
        # If not found, rank is beyond all results
        if original_rank == -1:
            original_rank = len(similar_cases) + 1
            original_score = 0.0
        
        # Top-k metrics
        top_1_score = similar_cases[0].get('final_score', 0.0) if similar_cases else 0.0
        top_5_scores = [
            similar_cases[i].get('final_score', 0.0)
            for i in range(min(5, len(similar_cases)))
        ]
        top_5_avg = float(np.mean(top_5_scores)) if top_5_scores else 0.0
        
        # Judge pass/fail
        rank_passed = original_rank <= expected_rank
        score_passed = original_score >= expected_score
        overall_passed = rank_passed and score_passed
        
        # Extract method scores if available
        method_scores = search_results.get('_all_scores', {})
        reranking_log = search_results.get('_reranking_log', {})
        
        # Check if original was boosted
        boosted_indices = reranking_log.get('detected_indices', [])
        original_boosted = (original_rank - 1) in boosted_indices if original_rank > 0 else False
        
        metrics = EvaluationMetrics(
            test_id=test_id,
            domain=test_case.domain,
            description=test_case.description,
            rephrased=test_case.rephrased[:100],
            original=test_case.original[:100],
            original_rank=original_rank,
            original_score=original_score,
            top_1_score=top_1_score,
            top_5_avg_score=top_5_avg,
            rank_passed=rank_passed,
            score_passed=score_passed,
            overall_passed=overall_passed,
            reranking_boosted=original_boosted,
            semantic_score=method_scores.get('semantic', [0] * original_rank)[
                original_rank - 1 if original_rank > 0 else 0
            ] if isinstance(method_scores.get('semantic'), list) else 0.0,
            lexical_score=method_scores.get('lexical', [0] * original_rank)[
                original_rank - 1 if original_rank > 0 else 0
            ] if isinstance(method_scores.get('lexical'), list) else 0.0,
            bm25_score=method_scores.get('bm25', [0] * original_rank)[
                original_rank - 1 if original_rank > 0 else 0
            ] if isinstance(method_scores.get('bm25'), list) else 0.0,
        )
        
        return metrics
    
    def compute_summary(self, results: List[EvaluationMetrics]) -> Dict:
        """
        Compute aggregate evaluation metrics.
        
        Args:
            results: List of EvaluationMetrics
        
        Returns:
            Dict with summary statistics
        """
        if not results:
            return {}
        
        passed = sum(1 for r in results if r.overall_passed)
        rank_passed = sum(1 for r in results if r.rank_passed)
        score_passed = sum(1 for r in results if r.score_passed)
        boosted = sum(1 for r in results if r.reranking_boosted)
        
        ranks = [r.original_rank for r in results]
        scores = [r.original_score for r in results]
        
        return {
            'total_tests': len(results),
            'passed': passed,
            'pass_rate': passed / len(results),
            'rank_passed_rate': rank_passed / len(results),
            'score_passed_rate': score_passed / len(results),
            'boosted_count': boosted,
            'avg_original_rank': float(np.mean(ranks)),
            'avg_original_score': float(np.mean(scores)),
            'median_original_rank': float(np.median(ranks)),
            'max_original_rank': int(np.max(ranks)),
            'min_original_rank': int(np.min(ranks)),
            'by_domain': self._summary_by_domain(results)
        }
    
    @staticmethod
    def _summary_by_domain(results: List[EvaluationMetrics]) -> Dict:
        """Compute summary by domain"""
        by_domain = {}
        for r in results:
            if r.domain not in by_domain:
                by_domain[r.domain] = []
            by_domain[r.domain].append(r)
        
        summary = {}
        for domain, domain_results in by_domain.items():
            passed = sum(1 for r in domain_results if r.overall_passed)
            summary[domain] = {
                'tests': len(domain_results),
                'passed': passed,
                'pass_rate': passed / len(domain_results),
                'avg_rank': float(np.mean([r.original_rank for r in domain_results])),
                'avg_score': float(np.mean([r.original_score for r in domain_results])),
            }
        
        return summary
    
    def save_results(self, path: str, results: List[EvaluationMetrics]):
        """Save evaluation results to JSON"""
        data = {
            'summary': self.compute_summary(results),
            'individual_results': [r.to_dict() for r in results]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"✅ Evaluation results saved to {path}")
    
    @staticmethod
    def load_results(path: str) -> Dict:
        """Load evaluation results from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        return data


def compare_systems(old_results: List[EvaluationMetrics],
                   new_results: List[EvaluationMetrics]) -> Dict:
    """
    Compare two system versions.
    
    Args:
        old_results: Results from old system
        new_results: Results from new system
    
    Returns:
        Dict with comparison metrics (improvement, regression, etc.)
    """
    evaluator = OfflineEvaluator(RephraseTestDataset())
    
    old_summary = evaluator.compute_summary(old_results)
    new_summary = evaluator.compute_summary(new_results)
    
    old_pass_rate = old_summary.get('pass_rate', 0)
    new_pass_rate = new_summary.get('pass_rate', 0)
    
    return {
        'old_system': old_summary,
        'new_system': new_summary,
        'improvements': {
            'pass_rate_delta': new_pass_rate - old_pass_rate,
            'pass_rate_improvement_pct': (
                ((new_pass_rate - old_pass_rate) / old_pass_rate * 100)
                if old_pass_rate > 0 else 0
            ),
            'avg_rank_improvement': (
                old_summary.get('avg_original_rank', 0) -
                new_summary.get('avg_original_rank', 0)
            ),
            'avg_score_improvement': (
                new_summary.get('avg_original_score', 0) -
                old_summary.get('avg_original_score', 0)
            ),
        }
    }
