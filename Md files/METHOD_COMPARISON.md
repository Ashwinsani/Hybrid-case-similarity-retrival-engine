# 4 Similarity Methods Visual Comparison

## Method 1: Semantic Similarity

```
INPUT: 
  User text: "AI fraud detection system"
  Case text: "Machine learning to catch money laundering"

PROCESS:
  1. Encode both with all-MiniLM-L6-v2
     user_emb = [0.12, -0.34, 0.67, ..., -0.18]  (384-d vector)
     case_emb = [0.14, -0.32, 0.65, ..., -0.19]  (384-d vector)
     
  2. Compute cosine similarity
     cos(u, c) = (u · c) / (||u|| × ||c||)
     = dot_product / (norm_u × norm_c)
     = 0.82
     
OUTPUT: Score 0.82
  ✅ HIGH: Different words but same meaning (fraud/money laundering = both illicit)
  
STRENGTHS:
  ✓ Catches synonyms
  ✓ Robust to paraphrasing
  ✓ Captures semantic meaning
  
WEAKNESSES:
  ✗ Not sensitive to specific technical details
  ✗ Can miss keyword-important distinctions
  ✗ More expensive (embedding neural network forward pass)
```

## Method 2: Lexical Similarity (TF-IDF)

```
INPUT:
  User text: "AI fraud detection system"
  Case text: "Machine learning to catch money laundering"
  Corpus: [5000 documents for IDF calculation]

PROCESS:
  1. Tokenize both (lowercase, remove stopwords):
     user_tokens = ["ai", "fraud", "detection", "system"]
     case_tokens = ["machine", "learning", "catch", "money", "laundering"]
     
  2. Compute TF (term frequency):
     user: {ai: 1/4, fraud: 1/4, detection: 1/4, system: 1/4}
     case: {machine: 1/5, learning: 1/5, catch: 1/5, money: 1/5, laundering: 1/5}
     
  3. Compute IDF (inverse document frequency):
     IDF[fraud] = log(5000 / 120) = 6.02  (120 docs contain "fraud", rare)
     IDF[system] = log(5000 / 890) = 1.74  (890 docs contain "system", common)
     IDF[money] = log(5000 / 340) = 4.09  (340 docs contain "money", medium)
     
  4. TF-IDF vectors:
     user = [1/4 × 0, 1/4 × 6.02, 1/4 × 4.51, 1/4 × 1.74] = [0, 1.50, 1.13, 0.44]
     case = [1/5 × 3.40, 1/5 × 2.81, 1/5 × 4.12, 1/5 × 4.09, 1/5 × 3.89] = [0.68, 0.56, 0.82, 0.82, 0.78]
     (assuming fraud not in case text)
     
  5. Cosine similarity:
     cos(TF-IDF_user, TF-IDF_case) = 0.35  (minimal overlap)

OUTPUT: Score 0.35
  ❌ LOW: No keyword overlap between two texts
  
STRENGTHS:
  ✓ Sensitive to exact terminology
  ✓ Weights rare terms higher (specific tech terms get boosted)
  ✓ Fast computation
  
WEAKNESSES:
  ✗ Fails for synonyms (money laundering ≠ fraud detection)
  ✗ Penalizes paraphrased content heavily
  ✗ Can overweight unusual but unimportant words
```

## Method 3: BM25 Ranking

```
INPUT:
  User query: "AI fraud detection system"
  Case: "Machine learning to catch money laundering"
  All tokenized documents in corpus

PROCESS:
  1. Tokenize query & document:
     query = ["ai", "fraud", "detection", "system"]
     doc = ["machine", "learning", "catch", "money", "laundering"]
     
  2. For each term in query, compute BM25 score:
     
     BM25("ai"):
     ├─ f(ai) = 0  (not in document)
     └─ score = 0
     
     BM25("fraud"):
     ├─ f(fraud) = 0  (not in document)
     └─ score = 0
     
     BM25("detection"):
     ├─ f(detection) = 0  (not in document)
     └─ score = 0
     
     BM25("system"):
     ├─ f(system) = 0  (not in document)
     └─ score = 0
     
  3. Total BM25 = 0 + 0 + 0 + 0 = 0
     
  4. Normalize to [0, 1]:
     (but if ALL docs score 0, normalization gives 0.5)
     final_score = 0.5 (normalized neutral)

OUTPUT: Score 0.50
  ⚠️ NEUTRAL: No keyword overlap, but parameterized scoring handles length
  
STRENGTHS:
  ✓ Industry standard (Elasticsearch, Lucene)
  ✓ Handles document length well (doesn't favor long docs)
  ✓ More nuanced than TF-IDF
  
WEAKNESSES:
  ✗ Also fails for synonyms
  ✗ No semantic understanding
  ✗ Needs careful parameter tuning (k1=1.5, b=0.75)
```

## Method 4: Keyword Matching (Jaccard)

```
INPUT:
  User text: "AI fraud detection system"
  Case text: "Machine learning to catch money laundering"

PROCESS:
  1. Extract keywords (length > 2, not stopwords, lowercase):
     user_keywords = {"ai", "fraud", "detection", "system"}
     case_keywords = {"machine", "learning", "catch", "money", "laundering"}
     
  2. Compute set overlap:
     intersection = {"ai"} ∩ {"machine"} = {}  (EMPTY!)
     
  3. Jaccard similarity:
     J = |A ∩ B| / |A ∪ B|
       = 0 / 9
       = 0.0
       
  4. Optional: boost by overlap_factor:
     overlap_factor = min(0 / max(4, 5), 1.0) = 0.0
     final = (0.0 × 0.6) + (0.0 × 0.4) = 0.0

OUTPUT: Score 0.00
  ❌ ZERO: No keyword overlap at all
  
STRENGTHS:
  ✓ Simple & transparent
  ✓ Handles exact keyword matching well
  ✓ No preprocessing complexity
  
WEAKNESSES:
  ✗ No synonym handling
  ✗ Binary (overlap or not)
  ✗ Noisy with very short texts
```

---

## Real Example: Why These Matter

### Case A: "AI Tax Fraud Detection"
```
Query: "Machine learning for detecting fraudulent tax filings"

SCORES:
├─ Semantic: 0.88 ✅ HIGH
│  └─ Meaning captured: AI + fraud detection + tax = tax fraud detection
│
├─ Lexical: 0.72 ✅ GOOD
│  └─ Keyword overlap: "fraud", "detection", "tax" present
│
├─ BM25: 0.65 ✅ GOOD
│  └─ Keywords well-distributed, IDF boost for "fraud" & "detection"
│
└─ Keyword: 0.58 ⚠️ MEDIUM
   └─ 3/5 query tokens in doc (fraud, detection, but not AI, ML, filings)

VERDICT: ✅ STRONG MATCH (all methods high)
WEIGHTED SCORE (with query-specific weights):
= 0.88 × 0.28 + 0.72 × 0.24 + 0.65 × 0.25 + 0.58 × 0.23
= 0.246 + 0.173 + 0.163 + 0.133
= 0.715 ✅ VERY HIGH

CONFIDENCE BOOST: No (lexical is high, not paraphrased)
FINAL: 0.715
```

### Case B: "Unlawful Tax Practice Identification System"
```
Query: "Machine learning for detecting fraudulent tax filings"

SCORES:
├─ Semantic: 0.84 ✅ HIGH
│  └─ Meaning similar: both about identifying problematic tax behavior
│
├─ Lexical: 0.18 ❌ VERY LOW
│  └─ Minimal word overlap (just "tax", very different wording)
│
├─ BM25: 0.55 ⚠️ MEDIUM
│  └─ "tax" + some contextual match, but few query keywords
│
└─ Keyword: 0.32 ❌ LOW
   └─ Only 1/5 query tokens present ("tax" only)

PARAPHRASE PATTERN DETECTED:
├─ semantic = 0.84 >= 0.60 ✓
├─ lexical = 0.18 < 0.30 ✓
├─ bm25 = 0.55 >= 0.50 ✓
└─ → PATTERN MATCH! Confidence = 0.75 (high)

WEIGHTED SCORE (before boosting):
= 0.84 × 0.28 + 0.18 × 0.24 + 0.55 × 0.25 + 0.32 × 0.23
= 0.235 + 0.043 + 0.138 + 0.074
= 0.490

CONFIDENCE BOOST APPLIED:
├─ Rule 1: semantic >= 0.65 AND bm25 >= 0.50 ✓
├─ boost_factor = 1.15 + (0.84 - 0.65) × 0.5
│                = 1.15 + 0.095
│                = 1.245 (24.5% boost)
└─ boosted_score = 0.490 × 1.245 = 0.610

FINAL: 0.610 (significantly boosted!)
RANKING: Lower than Case A, but still respectable
REASON: Paraphrase detected & handled correctly!
```

### Case C: "Agricultural Crop Optimization"
```
Query: "Machine learning for detecting fraudulent tax filings"

SCORES:
├─ Semantic: 0.32 ❌ VERY LOW
│  └─ Completely different domain (agriculture vs tax)
│
├─ Lexical: 0.08 ❌ NEAR ZERO
│  └─ Almost no keyword overlap
│
├─ BM25: 0.12 ❌ VERY LOW
│  └─ No content word overlap
│
└─ Keyword: 0.05 ❌ NEAR ZERO
   └─ No token overlap

PARAPHRASE PATTERN: ❌ NO
└─ semantic = 0.32 < 0.60, fails first check

NO BOOST APPLIED
└─ Correctly identified as unrelated case

WEIGHTED SCORE:
= 0.32 × 0.28 + 0.08 × 0.24 + 0.12 × 0.25 + 0.05 × 0.23
= 0.090 + 0.019 + 0.030 + 0.012
= 0.151

ADAPTIVE THRESHOLD: 0.24
FILTERING: ❌ FILTERED OUT (0.151 < 0.24)

FINAL: Doesn't appear in results (as it should)
```

---

## Summary Table

| Method | Captures | Handles | Speed | Use Case |
|--------|----------|---------|-------|----------|
| **Semantic** | Meaning, context | Synonyms, paraphrases | Medium (neural) | Abstract queries, meaning matching |
| **Lexical** | Exact keywords | Terminology matching | Fast | Technical domain matching |
| **BM25** | Important keywords | Document length variation | Fast | Information retrieval baseline |
| **Keyword** | Token overlap | Simple similarity | Very fast | Quick filter, debugging |

**Parting note:** No single method is perfect. That's why:
1. We use all 4 in parallel
2. Weights are dynamic (adapt to query)
3. Confidence scoring bridges gaps (detects & boosts paraphrases)
