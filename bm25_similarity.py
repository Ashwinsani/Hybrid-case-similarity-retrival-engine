import numpy as np
import re

# Try to import BM25Okapi; if not available, we'll provide graceful fallback
try:
    from rank_bm25 import BM25Plus
    _BM25_AVAILABLE = True
except ImportError as ie:
    print(f"⚠️ Warning: rank_bm25 library not installed: {ie}")
    print("   Install with: pip install rank-bm25")
    _BM25_AVAILABLE = False
except Exception as e:
    print(f"⚠️ Warning: Could not import BM25Plus: {e}")
    _BM25_AVAILABLE = False

def tokenize(text):
    return [w for w in re.findall(r'\w+', text.lower()) if len(w) > 2]

class BM25Similarity:
    def __init__(self, corpus_texts):
        """
        Initialize BM25 similarity with corpus texts.
        Gracefully handles unavailable rank_bm25 library.
        """
        self.corpus_texts = corpus_texts
        self.bm25 = None
        
        # Check if BM25 library is available
        if not _BM25_AVAILABLE:
            print("⚠️ BM25Okapi not available. BM25 similarity will return zeros.")
            return
        
        # Initialize BM25 only if corpus is not empty
        if not self.corpus_texts:
            print("⚠️ Empty corpus provided to BM25Similarity")
            self.bm25 = None
        else:
            try:
                tokenized_corpus = [tokenize(doc) for doc in corpus_texts]
                # BM25Plus: adds delta floor so identical docs always score highest
                # delta=1.0 guarantees every matching token contributes positively
                self.bm25 = BM25Plus(tokenized_corpus, k1=1.5, b=0.75, delta=1.0)
                print(f"✅ BM25Plus initialized with {len(corpus_texts)} documents (delta=1.0)")
            except Exception as e:
                print(f"⚠️ Failed to initialize BM25Okapi: {e}")
                self.bm25 = None

    def get_scores(self, query):
        """
        Get BM25 scores for a query against the corpus.
        Returns normalized scores in [0, 1] or zeros if BM25 unavailable.
        """
        # Return zeros if BM25 is not available
        if not _BM25_AVAILABLE or self.bm25 is None:
            if self.corpus_texts:
                return np.zeros(len(self.corpus_texts))
            return np.array([])
        
        # Empty query handling
        if not query or not isinstance(query, str):
            if self.corpus_texts:
                return np.zeros(len(self.corpus_texts))
            return np.array([])
            
        tokenized_query = tokenize(query)
        
        # Empty tokenized query handling
        if not tokenized_query:
            if self.corpus_texts:
                return np.zeros(len(self.corpus_texts))
            return np.array([])
        
        try:
            doc_scores = self.bm25.get_scores(tokenized_query)
        except Exception as e:
            print(f"⚠️ BM25 scoring failed: {e}")
            return np.zeros(len(self.corpus_texts)) if self.corpus_texts else np.array([])
        
        if len(doc_scores) == 0:
            return np.array([])
        
        # Normalize scores to [0, 1]
        min_score = np.min(doc_scores)
        max_score = np.max(doc_scores)
        
        if max_score == min_score:
            return np.full_like(doc_scores, 0.5, dtype=float)
            
        normalized_scores = (doc_scores - min_score) / (max_score - min_score + 1e-12)
        return normalized_scores