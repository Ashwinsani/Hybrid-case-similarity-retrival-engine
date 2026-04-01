# Visual Diagrams: 4-Method Hybrid Similarity Engine

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER QUERY INPUT                            │
│          "Build ML model for fraud detection"                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
    ┌─────▼──────┐            ┌────────▼────────┐
    │ Extract    │            │ Corpus Data     │
    │ Features   │            │ (10,000 cases)  │
    └─────┬──────┘            └────────┬────────┘
          │                            │
          └──────────────┬─────────────┘
                         │
        ┌────────────────▼───────────────────┐
        │    4 INDEPENDENT SCORERS           │
        │  (Score all cases simultaneously)  │
        └────┬───────┬───────┬──────────┬────┘
             │       │       │          │
        ┌────▼─┐ ┌───▼──┐ ┌──▼───┐ ┌───▼────┐
        │      │ │      │ │      │ │        │
        │Seman-│ │Lexi- │ │ BM25 │ │N-Gram  │
        │ tic  │ │ cal  │ │      │ │(Bigram)│
        │      │ │      │ │      │ │        │
        └────┬─┘ └───┬──┘ └──┬───┘ └───┬────┘
             │       │       │         │
    (0-1) 0.82    0.45    0.92     0.30
      │       │       │         │
      └───────┼───────┼─────────┘
              │
        ┌─────▼───────────────────────────┐
        │  MIN-MAX NORMALIZATION          │
        │  (Scale all to [0,1])           │
        └─────┬───────────────────────────┘
              │
    (0-1) 0.92    0.46    1.00     0.30
      │       │       │         │
      └───────┼───────┼─────────┘
              │
        ┌─────▼──────────────────────────────┐
        │  WEIGHT COMPUTATION                │
        │  (Per-case auto-tuning)            │
        │  Query: Short, keyword-heavy       │
        └─────┬──────────────────────────────┘
              │
    w  = 0.30   0.20   0.35    0.15
      │       │       │       │
      └───────┼───────┼───────┘
              │
        ┌─────▼─────────────────────────────┐
        │  WEIGHTED FUSION                  │
        │  final_score = Σ w * score        │
        │  = 0.92×0.30 + 0.46×0.20          │
        │    + 1.00×0.35 + 0.30×0.15        │
        │  = 0.68                           │
        └─────┬─────────────────────────────┘
              │
        ┌─────▼──────────────────┐
        │  RANKING & THRESHOLD   │
        │  Top 55 OR score ≥0.35 │
        └─────┬──────────────────┘
              │
        ┌─────▼──────────────────────────┐
        │  OUTPUT RESULTS                │
        │  • All 4 method scores         │
        │  • Case-specific weights       │
        │  • Top-K matching cases        │
        │  • Weight explanation          │
        └──────────────────────────────┘
```

---

## 2. Normalization Process

```
RAW SCORES (Different Ranges):

Method      Min   Value  Max    Problem
────────────────────────────────────────
Semantic    0.0   0.82   1.0    ✓ Standard range
Lexical     0.0   0.45   1.0    ✓ Standard range
BM25        0.0   15.3   100+   ✗ HUGE variation!
N-Gram      0.0   0.30   1.0    ✓ Standard range

Without normalization:
  If w_bm25 = 0.25, but BM25 ∈ [0-100]
  BM25 contributes 0-25 to final score
  Other methods contribute 0-0.25 each
  → BM25 completely dominates!


NORMALIZED SCORES (All [0, 1]):

Min-Max Formula: score_norm = (score - min) / (max - min)

BM25:    (15.3 - 0) / (100+ - 0) = 0.15  ← Scaled down
Semantic: (0.82 - 0) / (1.0 - 0) = 0.82  ← Already ok
Lexical:  (0.45 - 0) / (1.0 - 0) = 0.45  ← Already ok
N-Gram:   (0.30 - 0) / (1.0 - 0) = 0.30  ← Already ok

Now all methods on equal footing!
Weight distribution actually matters.
```

---

## 3. Weight Distribution by Input Type

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT: "Build ML model for fraud detection"                │
│  Features: Short (7 words), keyword-heavy, technical        │
└─────────────────┬───────────────────────────────────────────┘
                  │
              ANALYSIS
                  │
    ┌─────────────┼─────────────┐
    │             │             │
Length   Tech   Keywords
 7 words  HIGH    HIGH
              │
    ┌─────────┴──────────┐
    │                    │
  DECISION: Boost keyword methods (BM25, N-Gram)
            Lower semantic (less context)
            Lower lexical (too short)
            │
         WEIGHTS
            │
   ┌────────┴────────┐
   │                 │
Semantic  BM25  Lexical  N-Gram
  0.30    0.35   0.20    0.15
   │       │      │       │
   └───────┼──────┼───────┘
           │
    ┌──────▼────────────────────────────┐
    │  FINAL SCORE = 0.82×0.30          │
    │              + 0.92×0.35          │
    │              + 0.45×0.20          │
    │              + 0.30×0.15          │
    │  = 0.6795                         │
    │                                   │
    │  ✓ BM25 gets extra boost (0.35)   │
    └───────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────┐
│ INPUT: "Comprehensive healthcare system managing patient records │
│         and appointments with real-time sync and notifications"  │
│ Features: Long (25+ words), descriptive, context-rich           │
└────────────────────┬────────────────────────────────────────────┘
                     │
                 ANALYSIS
                     │
   ┌─────────────────┼─────────────────┐
   │                 │                 │
Length       Context    Keywords
25+ words    RICH       SCATTERED
             │
   ┌──────────┴──────────┐
   │                     │
DECISION: Boost semantic (understands context)
          Lower N-gram (phrases diluted)
          Lower BM25 (keywords scattered)
          Keep lexical moderate
          │
       WEIGHTS
          │
   ┌──────┴──────┐
   │             │
Semantic  BM25  Lexical  N-Gram
  0.50    0.15   0.20    0.15
   │      │      │       │
   └──────┼──────┼───────┘
          │
   ┌──────▼────────────────────────────┐
   │  FINAL SCORE = 0.88×0.50          │
   │              + 0.65×0.15          │
   │              + 0.52×0.20          │
   │              + 0.25×0.15          │
   │  = 0.653                          │
   │                                   │
   │  ✓ Semantic gets boost (0.50)     │
   └───────────────────────────────────┘
```

---

## 4. N-Gram (Bigram) Extraction

```
TEXT: "machine learning model development"

STEP 1: Tokenize
  Input:  "machine learning model development"
  Tokens: ['machine', 'learning', 'model', 'development']

STEP 2: Create consecutive pairs (bigrams)
  Pair 1: ('machine', 'learning')
  Pair 2: ('learning', 'model')
  Pair 3: ('model', 'development')
  
  Result: {('machine', 'learning'), ('learning', 'model'), 
           ('model', 'development')}

TEXT 1: "machine learning model development"
  Bigrams: {
    ('machine', 'learning'),
    ('learning', 'model'),
    ('model', 'development')
  }

TEXT 2: "model development and machine learning"
  Bigrams: {
    ('model', 'development'),
    ('development', 'and'),
    ('and', 'machine'),
    ('machine', 'learning')
  }

COMPARISON:
  Intersection: {('machine', 'learning'), ('model', 'development')}
               = 2 matching bigrams
  
  Union: {('machine', 'learning'),
          ('learning', 'model'),
          ('model', 'development'),
          ('development', 'and'),
          ('and', 'machine')}
        = 5 total unique bigrams
  
  Jaccard = 2 / 5 = 0.40


USE CASE: Captures phrase rearrangement!
TEXT 1: "neural network architecture design"
TEXT 2: "design neural network architecture"

Bigrams differ (order matters), so N-gram score is low
→ Correctly identifies that meaning is preserved but structure differs
→ Semantic method will catch the meaning similarity
→ N-Gram focuses on structural/phrase patterns
```

---

## 5. Weight Constraints Visualization

```
WEIGHT BOUNDS ENFORCEMENT:

Goal: No single method dominates
      Balanced portfolio approach

Constraint: 0.05 ≤ w ≤ 0.50 for each method
            w_sum = 1.0

Example Scenarios:

  Case 1: Initial weights before constraint
  ┌──────────────────────────────────────┐
  │ Semantic:  0.90  ← EXCEEDS MAX 0.50! │
  │ BM25:      0.05                      │
  │ Lexical:   0.03  ← BELOW MIN 0.05!   │
  │ N-Gram:    0.02  ← BELOW MIN 0.05!   │
  │ Sum:       1.00                      │
  └──────────────────────────────────────┘
              ↓ (Apply bounds)
  ┌──────────────────────────────────────┐
  │ Semantic:  0.50  ← Capped at MAX     │
  │ BM25:      0.05                      │
  │ Lexical:   0.05  ← Raised to MIN     │
  │ N-Gram:    0.05  ← Raised to MIN     │
  │ Sum:       0.65  ← No longer 1.0!    │
  └──────────────────────────────────────┘
              ↓ (Re-normalize)
  ┌──────────────────────────────────────┐
  │ Semantic:  0.50/0.65 = 0.769         │
  │ BM25:      0.05/0.65 = 0.077         │
  │ Lexical:   0.05/0.65 = 0.077         │
  │ N-Gram:    0.05/0.65 = 0.077         │
  │ Sum:       1.0000  ✓                 │
  └──────────────────────────────────────┘

Result: Semantic gets 77% (not 90%)
        All others get 7.7% (not dropped to 0%)
        → Balanced portfolio maintained


  Case 2: Reasonable weights
  ┌────────────────────────────────────────┐
  │ Semantic:  0.30  ← Within bounds      │
  │ BM25:      0.35  ← Within bounds      │
  │ Lexical:   0.20  ← Within bounds      │
  │ N-Gram:    0.15  ← Within bounds      │
  │ Sum:       1.00  ✓                    │
  └────────────────────────────────────────┘
              ↓ (No constraint violation)
           PASS AS-IS
```

---

## 6. Final Score Computation

```
FORMULA:
  final_score = w_semantic × score_semantic +
                w_lexical × score_lexical +
                w_bm25 × score_bm25 +
                w_ngram × score_ngram

INPUT EXAMPLE:
  Scores (normalized): [0.82, 0.45, 0.92, 0.30]
  Weights:            [0.30, 0.20, 0.35, 0.15]

COMPUTATION:
  final_score = 0.82 × 0.30 +
                0.45 × 0.20 +
                0.92 × 0.35 +
                0.30 × 0.15
  
              = 0.246 +
                0.090 +
                0.322 +
                0.045
  
              = 0.703

OUTPUT:
  ✓ final_score = 0.703 (valid: 0.0-1.0)
  ✓ Weights sum to 1.0
  ✓ BM25 contributes most (0.322)
  ✓ All methods contribute something
  ✓ Single method doesn't dominate


THRESHOLD & RANKING:
  Case A: final_score = 0.703  ✓ Accepted (> 0.35)
  Case B: final_score = 0.28   ✗ Rejected (< 0.35)
  Case C: final_score = 0.12   ✗ Rejected BUT in top-55
  
  → Accept Case C anyway (top-K fallback)
  → Return up to 55 cases minimum
```

---

## 7. Threshold Sensitivity Analysis

```
ANALYSIS:
  Query: "fraud detection system"
  Results: 100 cases total

┌───────────────────────────────────────────┐
│ Threshold  │  Matches  │  Cumulative      │
├───────────────────────────────────────────┤
│   0.30     │    55     │ 55 cases found   │
│   0.35     │    50     │ 50 cases found   │
│   0.40     │    42     │ 42 cases found   │
│   0.45     │    35     │ 35 cases found   │
│   0.50     │    28     │ 28 cases found   │
│   0.55     │    18     │ 18 cases found   │
│   0.60     │    12     │ 12 cases found   │
│   0.65     │     8     │ 8 cases found    │
│   0.70     │     5     │ 5 cases found    │
│   0.75     │     2     │ 2 cases found    │
└───────────────────────────────────────────┘

VISUALIZATION:
  
  0.75  ##
  0.70  #####
  0.65  ########
  0.60  ############
  0.55  ##################
  0.50  ############################
  0.45  ###################################
  0.40  ##########################################
  0.35  ##################################################
  0.30  #######################################################
         0   10   20   30   40   50   60

DECISION:
  - Use threshold 0.35-0.40 for balanced results
  - Lower (0.30) gives more results (55+)
  - Higher (0.60+) very strict (≤12)
  - Current default: 0.35
```

---

## 8. Complete Data Flow

```
                    USER QUERY
                        ↓
              ┌─────────┴──────────┐
              │                    │
          Feature           Corpus
         Extraction       Sampling
              │                │
              └────────┬───────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
    ┌───▼────┐  ┌────▼────┐  ┌───▼────┐  ┌────▼────┐
    │Semantic│  │ Lexical │  │  BM25  │  │ N-Gram  │
    │ Score  │  │  Score  │  │ Score  │  │ Score   │
    │  0.82  │  │  0.45   │  │ 0.92   │  │  0.30   │
    └───┬────┘  └────┬────┘  └───┬────┘  └────┬────┘
        │            │           │            │
        └────────────┼───────────┴────────────┘
                     │
            NORMALIZE TO [0,1]
                     │
        ┌────────────┼───────────┬────────────┐
        │            │           │            │
       0.92         0.46        1.00         0.30
        │            │           │            │
        └────────────┼───────────┴────────────┘
                     │
           WEIGHT OPTIMIZATION
                     │
        ┌────────────┼───────────┬────────────┐
        │            │           │            │
       w=0.30      w=0.20      w=0.35      w=0.15
        │            │           │            │
        └────────────┼───────────┴────────────┘
                     │
           WEIGHTED FUSION
      final = 0.92×0.30 + 0.46×0.20 +
              1.00×0.35 + 0.30×0.15 = 0.703
                     │
              ┌──────┴───────┐
              │              │
        Apply Threshold   Top-K
        (≥ 0.35)         (Top 55)
              │              │
              └──────┬───────┘
                     │
            RANK & OUTPUT
        28 matching cases
        All scores & weights shown
```

---

**Visual Summary Created**  
**Total Diagrams: 8**  
**Status: ✅ Complete**

These diagrams illustrate:
1. System architecture
2. Normalization importance
3. Weight distribution logic
4. N-gram extraction process
5. Constraint enforcement
6. Final score computation
7. Threshold sensitivity
8. Complete data flow
