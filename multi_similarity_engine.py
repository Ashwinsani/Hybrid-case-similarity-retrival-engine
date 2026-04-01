# COMPLETE FIXED multi_similarity_engine.py with threshold sensitivity
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import json
from datetime import datetime
import traceback
import sys
import joblib

# Helper function for safe printing (handles encoding issues on Windows)
def safe_print(message):
    """Print with fallback for Unicode encoding issues"""
    try:
        print(message)
    except UnicodeEncodeError:
        # Fallback: remove emojis and print
        fallback = message.encode('ascii', 'ignore').decode('ascii')
        print(fallback if fallback else "[Output not displayable]")

try:
    from bm25_similarity import BM25Similarity
except ImportError as e:
    print(f"[WARNING] Could not import BM25Similarity: {e}")
    BM25Similarity = None

try:
    from sentence_transformers import CrossEncoder
except ImportError as e:
    print(f"[WARNING] Could not import CrossEncoder: {e}")
    CrossEncoder = None

from sklearn.feature_extraction.text import TfidfVectorizer

class EnhancedCosineSimilarityMatcher:
    def __init__(self, corpus_dir="case_embeddings"):
        """
        Initialize with corpus embeddings and metadata
        """
        try:
            print("📁 Loading corpus embeddings...")
        except UnicodeEncodeError:
            print("[Loading corpus embeddings...]")
        try:
            # Load the main dataset embeddings
            embeddings_path = os.path.join(corpus_dir, "embeddings.npy")
            metadata_path = os.path.join(corpus_dir, "metadata.csv")
            
            if not os.path.exists(embeddings_path):
                raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
            self.corpus_embeddings = np.load(embeddings_path)
            self.corpus_metadata = pd.read_csv(metadata_path)
            
            print(f"[OK] Corpus loaded: {len(self.corpus_embeddings)} cases")

            # Initialize similarity models
            # Build similarity text from available columns
            self.texts = self._build_similarity_texts(self.corpus_metadata)
            
            print("[Initializing similarity models...]")
            
            # Initialize BM25 with error handling
            try:
                if BM25Similarity is not None:
                    self.bm25 = BM25Similarity(self.texts)
                else:
                    print("[WARNING] BM25Similarity not available, skipping BM25 initialization")
                    self.bm25 = None
            except Exception as e:
                print(f"[WARNING] BM25 initialization failed: {e}. Proceeding without it.")
                self.bm25 = None

            # Field Score: separate TF-IDF vectorizers per field with boost weights
            # field_weights: controls how much each field contributes to field_score
            self.field_config = [
                {'col': 'Idea Name',        'weight': 0.35},
                {'col': 'Domain',           'weight': 0.20},
                {'col': 'Idea Description', 'weight': 0.45},
            ]
            self.field_vectorizers = {}
            self.field_matrices = {}
            
            tfidf_path = os.path.join(corpus_dir, "tfidf_data.pkl")
            if os.path.exists(tfidf_path):
                print("📦 Loading cached TF-IDF models...")
                try:
                    cached_data = joblib.load(tfidf_path)
                    self.field_vectorizers = cached_data['vectorizers']
                    self.field_matrices = cached_data['matrices']
                    print("[OK] TF-IDF models loaded from cache")
                except Exception as e:
                    print(f"[WARNING] Failed to load TF-IDF cache: {e}. Rebuilding...")
                    self._build_and_cache_tfidf(tfidf_path)
            else:
                self._build_and_cache_tfidf(tfidf_path)
            
            # Initialize Cross-Encoder for final re-ranking
            self.cross_encoder = None
            try:
                if CrossEncoder is not None:
                    print("[Loading] Cross-Encoder model for re-ranking...")
                    self.cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-base')
                    print("[OK] Cross-Encoder initialized for re-ranking")
                else:
                    print("[WARNING] CrossEncoder not available, skipping Cross-Encoder initialization")
            except Exception as e:
                print(f"[WARNING] Cross-Encoder initialization failed: {e}. Proceeding without re-ranking.")
                self.cross_encoder = None

            print("[OK] Similarity models initialized.")

        except Exception as e:
            print(f"[ERROR] Failed to load corpus or initialize models: {e}")
            raise
    
    def _build_and_cache_tfidf(self, cache_path):
        """Build TF-IDF models and cache them to disk."""
        print("⚙️ Building TF-IDF models...")
        for fc in self.field_config:
            col = fc['col']
            field_texts = [
                str(row[col]).strip() if col in row.index and pd.notna(row[col]) else ''
                for _, row in self.corpus_metadata.iterrows()
            ]
            vec = TfidfVectorizer(max_features=3000, stop_words='english')
            try:
                mat = vec.fit_transform(field_texts)
                self.field_vectorizers[col] = vec
                self.field_matrices[col] = mat
            except Exception as e:
                print(f"[WARNING] Field vectorizer for '{col}' failed: {e}")
                self.field_vectorizers[col] = None
                self.field_matrices[col] = None
        
        try:
            joblib.dump({
                'vectorizers': self.field_vectorizers,
                'matrices': self.field_matrices
            }, cache_path)
            print("[OK] TF-IDF models cached successfully")
        except Exception as e:
            print(f"[WARNING] Failed to cache TF-IDF models: {e}")

    def get_latest_user_folder(self, base_dir="user_embeddings"):
        """Find the most recent user embedding folder"""
        print(f"[Searching] Looking for latest user folder in: {base_dir}")
        
        if not os.path.exists(base_dir):
            print(f"[WARNING] Directory not found: {base_dir}")
            return None
        
        # Get all USER_ folders
        user_folders = []
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and item.startswith("USER_"):
                user_folders.append(item_path)
        
        if not user_folders:
            print("[WARNING] No user folders found")
            return None
        
        print(f"[OK] Found {len(user_folders)} user folders")
        
        # Try to sort by modification time
        try:
            user_folders.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            latest_folder = user_folders[0]
            print(f"📅 Latest folder: {os.path.basename(latest_folder)}")
            return latest_folder
        except Exception as e:
            print(f"[WARNING] Could not sort by mtime: {e}")
            # Fallback: sort alphabetically
            user_folders.sort(reverse=True)
            latest_folder = user_folders[0]
            print(f"📅 Latest folder (alphabetical): {os.path.basename(latest_folder)}")
            return latest_folder
    
    def _build_similarity_texts(self, metadata_df):
        """Build similarity texts from available metadata columns - SAME FORMAT as user input"""
        texts = []
        for _, row in metadata_df.iterrows():
            parts = []
            
            # Use SAME fields in SAME order as _extract_similarity_text for user input
            for col in ['Idea Name', 'Domain', 'Idea Description']:
                if col in row and pd.notna(row[col]):
                    text = str(row[col]).strip()
                    if text:
                        parts.append(text)
            
            # Join all parts
            full_text = ' '.join(parts) if parts else ''
            texts.append(full_text)
        
        return texts
    
    def _load_user_embeddings(self, user_folder):
        """Load user embeddings and metadata"""
        try:
            print(f"📥 Loading user data from: {user_folder}")
            
            # Try multiple possible filenames
            possible_filenames = ["embeddings.npy", "embedding.npy"]
            embeddings_path = None
            
            for filename in possible_filenames:
                test_path = os.path.join(user_folder, filename)
                if os.path.exists(test_path):
                    embeddings_path = test_path
                    print(f"[OK] Found embedding file: {filename}")
                    break
            
            if not embeddings_path:
                raise FileNotFoundError(f"No embedding file found in {user_folder}")
            
            metadata_path = os.path.join(user_folder, "metadata.json")
            info_path = os.path.join(user_folder, "info.json")
            
            # Load files
            user_embeddings = np.load(embeddings_path)
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                user_metadata = json.load(f)
                
            with open(info_path, 'r', encoding='utf-8') as f:
                user_info = json.load(f)
        
            print(f"[OK] User data loaded: {user_info.get('user_id', 'Unknown')}")
            
            return user_embeddings, user_metadata, user_info
            
        except Exception as e:
            print(f"[ERROR] Failed to load user session: {e}")
            raise
    
    def _extract_similarity_text(self, user_metadata):
        """
        Extract similarity text using the SAME logic as corpus building.
        This ensures user input similarity_text matches corpus similarity_text format.
        """
        similarity_text = ""
    
        # Try to get similarity_text field first (if available)
        if 'similarity_text' in user_metadata and user_metadata['similarity_text']:
            return str(user_metadata['similarity_text']).strip()
        
        # Build from available fields in the SAME order as corpus
        parts = []
        for col in ['Idea Name', 'Domain', 'Idea Description']:
            if col in user_metadata and user_metadata[col]:
                text = str(user_metadata[col]).strip()
                if text:
                    parts.append(text)
        
        # Join all parts
        similarity_text = " ".join(parts)
        return similarity_text
    
    def _calculate_semantic_scores(self, user_embedding):
        """Calculate semantic scores for all cases in the corpus using Scikit-Learn's cosine similarity."""
        # Ensure dimensionality matches
        user_dim = user_embedding.shape[0]
        corpus_dim = self.corpus_embeddings.shape[1]
        
        if user_dim != corpus_dim:
            min_dim = min(user_dim, corpus_dim)
            user_embedding = user_embedding[:min_dim]
            corpus_embeddings = self.corpus_embeddings[:, :min_dim]
        else:
            corpus_embeddings = self.corpus_embeddings

        # Calculate cosine similarity
        similarities = cosine_similarity([user_embedding], corpus_embeddings)[0]
        return similarities

    def _calculate_field_scores(self, user_metadata):
        """Calculate field-weighted scores for all cases in the corpus.

        Each field (Idea Name, Domain, Idea Description) is scored independently
        using its own TF-IDF vectorizer, then blended using configurable weights.
        This gives importance boosts to key fields (e.g. title match > body match).
        """
        n = len(self.corpus_metadata)
        combined = np.zeros(n)
        total_weight = 0.0

        for fc in self.field_config:
            col = fc['col']
            weight = fc['weight']
            vec = self.field_vectorizers.get(col)
            mat = self.field_matrices.get(col)

            if vec is None or mat is None:
                continue

            # Get the query value for this field
            if isinstance(user_metadata, dict):
                field_query = str(user_metadata.get(col, '')).strip()
            else:
                field_query = str(user_metadata).strip()  # fallback: treat whole string as query

            if not field_query:
                continue

            try:
                query_vec = vec.transform([field_query])
                field_sims = cosine_similarity(query_vec, mat).ravel()
                combined += field_sims * weight
                total_weight += weight
            except Exception as e:
                print(f"[WARNING] Field score for '{col}' failed: {e}")

        if total_weight > 0:
            combined /= total_weight

        return combined

    def _calculate_bm25_scores(self, query_text):
        """Calculate BM25 scores for all cases in the corpus."""
        if self.bm25 is not None:
            return self.bm25.get_scores(query_text)
        # Fallback to zeros if BM25 not available
        return np.zeros(len(self.corpus_metadata))

    def _rerankwith_cross_encoder(self, query_text, cases):
        """Re-rank cases using Cross-Encoder for improved relevance.
        
        Cross-Encoder provides fine-grained neural ranking based on query-case pairs.
        This is applied as a final refinement step after initial filtering.
        
        Uses stsb-roberta-base which is trained for semantic textual similarity (0-1 scale),
        making it appropriate for comparing rephrased/similar idea descriptions.
        CE score is blended (70%) with the original final_score (30%) to avoid
        completely overriding high-confidence multi-method matches.
        """
        if not self.cross_encoder or not cases:
            return cases
        
        try:
            print(f"[Re-ranking] Using Cross-Encoder to re-rank {len(cases)} cases...")
            
            # Prepare query-case pairs using FULL original corpus text (not truncated display text)
            query_case_pairs = []
            for case in cases:
                # Use the full text from self.texts via embedding_idx for accurate scoring
                emb_idx = case.get('embedding_idx')
                if emb_idx is not None and emb_idx < len(self.texts):
                    case_text = self.texts[emb_idx]  # full original corpus text
                else:
                    case_text = case.get('similarity_text', '')  # fallback
                query_case_pairs.append([query_text, case_text])
            
            # Get cross-encoder scores (stsb-roberta-base returns values in 0-1 range)
            ce_scores = self.cross_encoder.predict(query_case_pairs)
            
            # Normalize CE scores to [0, 1] in case of any out-of-range values
            ce_min, ce_max = ce_scores.min(), ce_scores.max()
            if ce_max - ce_min > 1e-9:
                ce_scores_norm = (ce_scores - ce_min) / (ce_max - ce_min)
            else:
                ce_scores_norm = np.ones_like(ce_scores) * 0.5
            
            # Blend CE score with original final_score:
            # 60% CE (semantic rerank) + 40% original combined score (multi-method trust)
            CE_WEIGHT = 0.60
            for i, case in enumerate(cases):
                original_score = case.get('final_score', 0.0)
                blended = CE_WEIGHT * float(ce_scores_norm[i]) + (1 - CE_WEIGHT) * original_score
                case['cross_encoder_score'] = round(float(ce_scores_norm[i]), 4)
                case['cross_encoder_raw'] = round(float(ce_scores[i]), 4)
                case['reranked_score'] = round(blended, 4)
            
            # Re-rank by blended score
            reranked_cases = sorted(cases, key=lambda x: x['reranked_score'], reverse=True)
            
            # Update ranks
            for idx, case in enumerate(reranked_cases, 1):
                case['rank'] = idx
            
            print(f"[OK] Re-ranking complete. Top case CE score: {reranked_cases[0].get('cross_encoder_score', 0):.4f}, blended: {reranked_cases[0].get('reranked_score', 0):.4f}")
            return reranked_cases
            
        except Exception as e:
            print(f"[WARNING] Cross-Encoder re-ranking failed: {e}. Returning original ranking.")
            return cases
    
    def _get_bigrams(self, text):
        """Extract bigrams (2-grams) from text."""
        tokens = re.findall(r'\b\w+\b', text.lower())
        if len(tokens) < 2:
            return set()
        return set(zip(tokens, tokens[1:]))

    def _calculate_ngram_scores(self, query_text):
        """Calculate N-gram (bigram) similarity for all cases in the corpus.
        
        Measures phrase-level overlap by comparing sequence of words.
        More sophisticated than simple keyword matching.
        """
        if not query_text or not query_text.strip():
            return np.zeros(len(self.corpus_metadata))
        
        try:
            query_bigrams = self._get_bigrams(query_text)
            
            if not query_bigrams:
                return np.zeros(len(self.corpus_metadata))
            
            scores = np.zeros(len(self.corpus_metadata))
            
            for idx, case_text in enumerate(self.texts):
                if not case_text or not case_text.strip():
                    scores[idx] = 0.0
                    continue
                
                case_bigrams = self._get_bigrams(case_text)
                
                if not case_bigrams:
                    scores[idx] = 0.0
                    continue
                
                # Calculate Jaccard similarity on bigrams
                intersection = len(query_bigrams & case_bigrams)
                union = len(query_bigrams | case_bigrams)
                
                if union > 0:
                    scores[idx] = intersection / union
                else:
                    scores[idx] = 0.0
            
            return scores
            
        except Exception as e:
            print(f"[WARNING] N-gram scoring failed: {e}")
            return np.zeros(len(self.corpus_metadata))

    def _apply_confidence_scoring(self, final_scores, scores):
        """
        DISABLED: Confidence scoring removed - using natural raw scores only.
        """
        return final_scores

    def _calculate_all_final_scores(self, user_embedding, query_text, weight_distribution, user_metadata=None):
        """Calculate final scores for ALL cases in corpus by combining all similarity methods."""
        print("[Calculating] Computing all final scores with weighted averaging...")

        # Get scores from all methods (semantic, field_score, bm25, ngram)
        semantic_scores = self._calculate_semantic_scores(user_embedding)

        # Field score uses per-field metadata if available, else falls back to query_text
        field_meta = user_metadata if user_metadata else query_text
        field_scores_raw = self._calculate_field_scores(field_meta)

        bm25_scores_raw = self._calculate_bm25_scores(query_text)
        ngram_scores = self._calculate_ngram_scores(query_text)

        # Normalize all scores to 0-1 range (min-max)
        def minmax_normalize(scores):
            score_min = np.min(scores)
            score_max = np.max(scores)
            if score_max - score_min > 1e-9:
                return (scores - score_min) / (score_max - score_min)
            else:
                return np.ones_like(scores) * 0.5

        if np.max(np.abs(bm25_scores_raw)) > 1e-9:
            bm25_scores = minmax_normalize(bm25_scores_raw)
        else:
            bm25_scores = np.zeros_like(bm25_scores_raw)

        if np.max(field_scores_raw) > 1e-9:
            field_scores = minmax_normalize(field_scores_raw)
        else:
            field_scores = np.zeros_like(field_scores_raw)

        if np.max(semantic_scores) > 1e-9:
            semantic_scores = minmax_normalize(semantic_scores)

        if np.max(ngram_scores) > 1e-9:
            ngram_scores = minmax_normalize(ngram_scores)

        # Safety clipping to [0, 1]
        field_scores   = np.clip(field_scores,   0, 1)
        semantic_scores = np.clip(semantic_scores, 0, 1)
        ngram_scores   = np.clip(ngram_scores,   0, 1)
        bm25_scores    = np.clip(bm25_scores,    0, 1)

        scores = {
            'semantic':    semantic_scores,
            'field_score': field_scores,   # replaced 'lexical'
            'bm25':        bm25_scores,
            'ngram':       ngram_scores
        }

        final_scores = np.zeros(len(self.corpus_metadata))

        # Use semantic-only as fallback if no weights provided
        if not weight_distribution or len(weight_distribution) <= 1:
            final_scores = scores['semantic']
        else:
            # Weighted fusion: remap 'lexical' → 'field_score' for backward compat
            for method, weight in weight_distribution.items():
                key = 'field_score' if method == 'lexical' else method
                if key in scores:
                    final_scores += scores[key] * weight

        print(f"[OK] Calculated {len(final_scores)} final scores (normalized & fused)")
        return final_scores, scores
    
    def _compute_threshold_sensitivity(self, all_final_scores):
        """Compute threshold sensitivity analysis"""
        print("📈 Computing threshold sensitivity...")
        
        thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
        sensitivity_table = []
        
        for t in thresholds:
            count = sum(score >= t for score in all_final_scores)
            sensitivity_table.append({
                "threshold": round(t, 3),
                "matches_found": int(count)
            })
        
        print(f"[OK] Sensitivity computed for {len(thresholds)} thresholds")
        return sensitivity_table
    
    def enhanced_find_similar_cases(self, user_folder=None, top_k=55, similarity_threshold=0.5, skip_rerank=False):
        """Enhanced similarity matching with per-case optimization and threshold sensitivity
        
        Args:
            user_folder: User embedding folder
            top_k: Number of results
            similarity_threshold: Minimum similarity score
            skip_rerank: If True, skip cross-encoder reranking (for enhanced version)
        """
        print(f"\n🎯 ENHANCED SIMILARITY MATCHING")
        print(f"   Top K: {top_k}, Threshold: {similarity_threshold}, Skip Rerank: {skip_rerank}")
        
        try:
            # If no specific folder provided, find the latest one
            if user_folder is None:
                user_folder = self.get_latest_user_folder()
            
            if not user_folder:
                print("[ERROR] No user folder found")
                return self._create_error_result("No user folder found")
            
            print(f"[Searching] Using user folder: {os.path.basename(user_folder)}")
            
            # Load user embeddings
            user_embeddings, user_metadata, user_info = self._load_user_embeddings(user_folder)
            user_embedding = user_embeddings[0]
            
            # Extract similarity text for optimization
            similarity_text = self._extract_similarity_text(user_metadata)
            
            # Run per-case optimization
            weight_distribution = {}
            case_analysis = {}
            
            try:
                # Import and use the optimizer
                from per_case_optimizer_v2 import PerCaseOptimizerV2
                
                # Create optimizer instance
                optimizer = PerCaseOptimizerV2(
                    self.corpus_embeddings,
                    self.corpus_metadata,
                    sample_size=min(100, len(self.corpus_embeddings))
                )
                
                # Prepare case metadata for optimizer
                case_meta_for_optimizer = {
                    'similarity_text': similarity_text,
                    'Idea Name': user_metadata.get('Idea Name', ''),
                    'Idea Description': user_metadata.get('Idea Description', ''),
                    'Domain': user_metadata.get('Domain', '')
                }
                
                # Run optimization
                adjusted_scores, weights, analysis = optimizer.optimize_for_case(
                    user_embedding,
                    case_meta_for_optimizer
                )
                
                weight_distribution = weights
                case_analysis = analysis
                
                print("✅ Per-case optimization completed")
                for method, weight in weight_distribution.items():
                    print(f"   - {method}: {weight:.3f}")
                    
            except Exception as e:
                print(f"[WARNING] Per-case optimization failed: {e}")
                traceback.print_exc()
                # Fallback to basic semantic weights
                weight_distribution = {'semantic': 1.0}
                case_analysis = {
                    'error': str(e),
                    'explanation': ['Using fallback semantic-only weights']
                }
            
            # Calculate ALL final scores for threshold sensitivity
            # Pass user_metadata so field_score uses per-field data
            all_final_scores, all_scores = self._calculate_all_final_scores(
                user_embedding, similarity_text, weight_distribution,
                user_metadata=user_metadata
            )
            
            # Compute threshold sensitivity
            threshold_sensitivity = self._compute_threshold_sensitivity(all_final_scores)

            # Get top_k most similar cases based on the final combined score
            top_indices = np.argsort(all_final_scores)[::-1][:top_k*2]
            
            similar_cases = []
            
            for idx in top_indices:
                final_score = all_final_scores[idx]
                
                # Filter by similarity threshold
                if final_score < similarity_threshold:
                    continue

                if len(similar_cases) >= top_k:
                    break
                
                # Get case metadata
                case_metadata = self.corpus_metadata.iloc[idx]
                
                case_name = case_metadata.get('Idea Name', 'Unnamed Case')
                case_id = case_metadata.get('Idea Id', f'Case_{idx}')
                
                domain = 'Not Specified'
                if 'Domain' in case_metadata:
                    domain_value = case_metadata['Domain']
                    if pd.notna(domain_value) and str(domain_value).strip() != '':
                        domain = str(domain_value).strip()
                
                description = 'No description available'
                if 'Idea Description' in case_metadata and pd.notna(case_metadata['Idea Description']) and str(case_metadata['Idea Description']).strip() != '':
                    desc_text = str(case_metadata['Idea Description']).strip()
                    if len(desc_text) > 10:
                        description = desc_text[:300] + '...' if len(desc_text) > 300 else desc_text
                
                # Create similarity scores breakdown from the pre-calculated scores
                similarity_scores = {method: scores[idx] for method, scores in all_scores.items()}

                case_result = {
                    'rank': len(similar_cases) + 1,
                    'case_id': case_id,
                    'case_name': case_name,
                    'similarity_score': round(all_scores['semantic'][idx], 4), # Keep semantic for reference
                    'final_score': round(final_score, 4),
                    'domain': domain,
                    'description': description,
                    'embedding_idx': int(idx),
                    'similarity_text': f"{case_name} {domain} {description}",
                    'similarity_scores': {k: round(v, 4) for k, v in similarity_scores.items()},
                    'case_specific_weights': weight_distribution,
                    'full_metadata': case_metadata.to_dict()
                }
                similar_cases.append(case_result)
            
            print(f"[OK] Found {len(similar_cases)} similar cases (after threshold filtering)")
            
            # DEBUG PRINT - Add this to see what's happening
            print(f"🔍 DEBUG - Base class reranking: skip_rerank={skip_rerank}, cross_encoder={self.cross_encoder is not None}")
            
            # Apply Cross-Encoder re-ranking if available - ONLY IF NOT SKIPPED
            if not skip_rerank and self.cross_encoder and similar_cases:
                print("🔄 Base class APPLYING reranking")
                similar_cases = self._rerankwith_cross_encoder(similarity_text, similar_cases)
            elif skip_rerank:
                print("⏭️ Base class SKIPPING reranking as requested")
            
            # Prepare results
            results = {
                'user_input': {
                    'user_id': user_info.get('user_id', 'Unknown'),
                    'folder': user_folder,
                    'idea_name': user_metadata.get('Idea Name', 'Unknown'),
                    'domain': user_metadata.get('Domain', 'Unknown'),
                    'funding_source': user_metadata.get('fundingSource', 'Unknown'),
                    'expected_benefits': user_metadata.get('Expected benefits', 'Unknown'),
                    'idea_description': user_metadata.get('Idea Description', 'Unknown'),
                    'potential_challenges': user_metadata.get('potential Challenges', 'Unknown'),
                    'similarity_text_preview': (similarity_text[:100] + '...') if similarity_text and len(similarity_text) > 100 else similarity_text,
                    'similarity_text': similarity_text,  # Add full text for reranking
                    'embedding_dimension': user_info.get('embedding_dim', 'Unknown'),
                    'created_at': user_info.get('created_at', 'Unknown')
                },
                'similar_cases': similar_cases,
                '_all_final_scores': all_final_scores,  # Store for enhanced version
                '_all_scores': all_scores,  # Store for enhanced version
                'match_statistics': {
                    'total_cases_searched': len(self.corpus_embeddings),
                    'total_matches_found': len(similar_cases),
                    'top_similarity_score': similar_cases[0]['final_score'] if similar_cases else 0,
                    'similarity_threshold': similarity_threshold,
                    'top_k_requested': top_k,
                    'methods_used': list(weight_distribution.keys()) if weight_distribution else ['semantic'],
                    'optimization_mode': 'true_per_case' if weight_distribution and len(weight_distribution) > 1 else 'basic',
                    'weight_distribution': weight_distribution,
                    'case_analysis': case_analysis,
                    'weight_computation_explanation': case_analysis.get('explanation', ['Weights computed per case based on content characteristics']),
                    'threshold_sensitivity': threshold_sensitivity,
                    'all_final_scores': all_final_scores,
                    'filtered_final_scores': [case['final_score'] for case in similar_cases],
                    'cross_encoder_enabled': self.cross_encoder is not None,
                    'cross_encoder_model': 'cross-encoder/ms-marco-MiniLM-L-6-v2' if self.cross_encoder else None,
                    'cross_encoder_scores': [case.get('cross_encoder_score', None) for case in similar_cases] if self.cross_encoder else []
                }
            }
            
            return results
            
        except Exception as e:
            print(f"[ERROR] Error in enhanced_find_similar_cases: {e}")
            traceback.print_exc()
            return self._create_error_result(str(e))
    def _create_error_result(self, error_message):
        """Create error result"""
        return {
            'user_input': {
                'user_id': 'ERROR',
                'error': True,
                'error_message': error_message
            },
            'similar_cases': [],
            'match_statistics': {
                'total_cases_searched': 0,
                'total_matches_found': 0,
                'error': True,
                'error_message': error_message
            }
        }