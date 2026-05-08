# Tagore Songs Auto-Research — Agent Instructions

You are an autonomous research agent. Your goal is to discover interesting, statistically grounded, and narratively meaningful insights from a corpus of ~2,300 songs by Rabindranath Tagore.

Run experiments in a loop, indefinitely, until stopped. Build on what you learn.

## The Dataset

Located at `Pradipta/tagore_songs` on Hugging Face. Load it using `dataset.py`:

```python
from dataset import load_tagore, describe_dataset
df = load_tagore()
```

**Columns** (all originally in Bengali, renamed to English):

| Column | Description |
|--------|-------------|
| `lyrics` | Full Bengali song lyrics (Unicode) |
| `category` | Primary theme — পূজা (Devotion/Worship), প্রেম (Love), প্রকৃতি (Nature), স্বদেশ (Patriotism), নাট্যগীতি/গীতিনাট্য (Dramatic/Theater songs), বিচিত্র (Miscellaneous), আনুষ্ঠানিক (Ceremonial) and others. 16 classes. |
| `subcategory` | Finer thematic grouping. 92 classes. |
| `serial_number` | Catalog number in the Gitabitan (Tagore's song anthology) |
| `bengali_year` | Year of composition (Bengali calendar) |
| `gregorian_year` | Year of composition (Gregorian calendar) |
| `raga` | Melodic mode from Hindustani classical music. 394 distinct values. |
| `taal` | Rhythmic cycle pattern. 79 distinct values. |

### Raga Field Caveats

The `raga` field is noisy and requires interpretation. Experiments working with ragas should decide how to handle the following:

1. **Paired ragas**: Some entries list two ragas joined together (e.g., "যমন-ভৈরবী" / Yaman-Bhairavi), meaning the song uses both. Possible treatments: count as both ragas, treat as a distinct combination, or pick the first/primary.
2. **Modifier prefixes**: Names like "মিশ্র ভৈরবী" (Misra Bhairavi) use a modifier ("Misra" = mixed). One can argue Misra Bhairavi should be counted as Bhairavi for grouping purposes, or kept separate if the analysis cares about modal nuance.
3. **Non-raga entries**: Some values are not ragas at all — e.g., "বাউল" (Baul) is a folk genre, "ইটালিয়ান সুর" (Italian Tune) is a foreign melody reference, "কীর্তন" (Kirtan) is a devotional genre. Experiments should decide whether to include, exclude, or flag these as a separate "genre-based" category distinct from classical ragas.

Each experiment should document its raga-handling strategy in its result block.

## Who Was Tagore

Rabindranath Tagore (1861–1941) was a Bengali polymath — poet, musician, playwright, painter, Nobel laureate in Literature (1913). His ~2,200 songs, collectively called **Rabindra Sangeet**, form a genre of their own in Bengali music. Key life context:

- Early songs (1880s–1890s): influenced by Brahmo Samaj devotional tradition
- Middle period (1900s–1910s): Swadeshi movement, patriotic songs, personal loss (wife and children died 1902–1907)
- Nobel period (1913+): international fame, philosophical deepening
- Late period (1930s–1941): experimental, sometimes darker themes, illness

His songs are organized in the **Gitabitan** into thematic sections (পর্যায়/category), which he himself curated.

## Your Research Loop

For each iteration:

### 1. HYPOTHESIZE

Read `research_log.md` to see what's been tried. Then propose a **specific, testable hypothesis** or prediction task. Examples:

- "Songs written after personal tragedy (1902–1907) use a measurably different vocabulary than surrounding years"
- "Raga can be predicted from lyrics with >X% accuracy (where random baseline is Y%)"  
- "Tagore's use of taal shifted over his career from complex meters to simpler ones"
- "Devotional and love songs are distinguishable by character n-gram features alone"

Be specific. "There are patterns in the data" is not a hypothesis.

### 2. RESEARCH CONTEXT

Before writing code, use web search to gather domain knowledge relevant to your hypothesis. Understand:

- Why might this hypothesis be true? What's the causal mechanism?
- What does musicology/literary criticism say about this aspect of Tagore?
- Are there known facts that make this interesting or trivial?

This step is critical. A finding is only interesting if there's a plausible story behind it. Skip this step only if you already gathered sufficient context in a prior iteration.

### 3. WRITE EXPERIMENT

Write a self-contained Python script at `experiments/experiment_NNN.py` where NNN is zero-padded (001, 002, ...). The script must:

- Import from `dataset.py` (use `sys.path` to find it)
- Perform the analysis
- Print a structured result block at the end:

```
=== RESULT ===
hypothesis: <one-line hypothesis>
method: <what you did>
key_finding: <the main number or fact>
statistical_significance: <p-value, accuracy vs baseline, effect size, etc.>
conclusion: <1-2 sentence interpretation>
=== END RESULT ===
```

### 4. EXECUTE

Run the script: `uv run experiments/experiment_NNN.py` with a 2-minute timeout.

If it fails, debug and fix (up to 2 retries). If it still fails, log the failure and move on.

### 5. EVALUATE (Two Axes)

Score the result on two dimensions:

**Statistical Strength (0–5)**:
- 0: Script failed or produced no result
- 1: Result exists but not statistically tested
- 2: Tested but not significant (p>0.05) or accuracy near baseline
- 3: Significant (p<0.05) or accuracy meaningfully above baseline
- 4: Strongly significant (p<0.01) with decent effect size
- 5: Overwhelming evidence, large effect, robust to methodology choices

**Narrative Strength (0–5)**:
- 0: No interpretable meaning
- 1: Technically true but trivially obvious (e.g., "longer songs have more words")
- 2: Mildly interesting but no clear mechanism
- 3: Interesting finding with a plausible causal story
- 4: Surprising and well-explained — would interest a musicologist
- 5: Genuinely revelatory — changes understanding of the corpus

**Verdict:**
- INTERESTING: Both scores ≥ 3
- WEAK: One or both scores < 3 but not a failure
- FAILED: Script error or nonsensical result

### 6. LOG

Append to `research_log.md`:

```markdown
## Experiment NNN: <Title>
- **Hypothesis**: ...
- **Method**: ...
- **Key finding**: ...
- **Statistical evidence**: ...
- **Context/narrative**: ...
- **Statistical score**: X/5
- **Narrative score**: X/5
- **Verdict**: INTERESTING / WEAK / FAILED
- **Script**: experiments/experiment_NNN.py
- **Build-on ideas**: <what future experiments could extend this>
```

Then loop back to step 1.

## Analytical Toolbox

You may use any of these approaches:

**Machine Learning & Prediction:**
- Decision trees, random forests, gradient boosting (XGBoost)
- SVM, k-NN, logistic regression, naive Bayes
- Multi-class and binary classification
- Regression (predicting year, serial number, etc.)
- Always report accuracy vs a sensible baseline (majority class, random, etc.)
- Use cross-validation (k-fold), never evaluate on training data

**Clustering & Dimensionality Reduction:**
- k-means, hierarchical clustering, DBSCAN
- PCA, t-SNE, UMAP
- Silhouette scores, cluster purity

**Statistical Tests:**
- Chi-squared, Fisher exact, ANOVA, Kruskal-Wallis
- Mutual information, correlation analysis
- Permutation tests, bootstrap confidence intervals

**NLP Features (from lyrics):**
- TF-IDF (character and word level)
- N-gram distributions (character n-grams work well for Bengali)
- Text statistics: length, unique word count, vocabulary richness
- Character frequency distributions
- Punctuation and structural features (verse count, line length)

**FORBIDDEN:**
- Do NOT use large language models (LLMs) for classification, embedding, or feature extraction. The dataset is likely in their training data, making any LLM-based result circular and meaningless.
- Do NOT use any pre-trained language model embeddings (BERT, sentence-transformers, etc.) for the same reason.
- Stick to classical ML and hand-crafted features.

## Research Directions

Explore these and invent your own:

1. **Musical-textual relationships**: Can lyrics predict raga? Taal? Do certain words/sounds associate with specific ragas?
2. **Temporal evolution**: How did Tagore's style change over 60 years? Vocabulary, complexity, raga choices, themes.
3. **Thematic classification**: Predict category/subcategory from lyrics using classical ML. What features matter most?
4. **Cross-feature structure**: Do raga and taal co-occur in non-random patterns? Does category predict musical choices?
5. **Life events as features**: Do songs cluster differently around known biographical events (Nobel Prize 1913, wife's death 1902, Swadeshi movement 1905)?
6. **Phonetic/prosodic patterns**: Character-level analysis — do devotional songs favor certain Bengali phonemes?
7. **Compositional productivity**: How did output volume and diversity change over time?
8. **Raga ecology**: Distribution of ragas — are most songs in a few ragas? Long tail? Changes over time?

## Anti-Patterns (Avoid These)

- **Counting without testing**: "Category X has the most songs" is not an insight unless you test whether the distribution is different from expected.
- **Obvious findings**: "Longer lyrics have more unique words" — don't waste an experiment on this.
- **No baseline comparisons**: Reporting 60% accuracy means nothing without knowing what random/majority baseline is.
- **Repeating failures**: If an approach failed, don't retry the same thing. Build on it or move on.
- **Overfitting**: Always cross-validate. Never report training accuracy as a result.
- **Ignoring nulls**: Many songs have missing raga/taal/year. Handle or exclude them explicitly.
- **LLM leakage**: No pre-trained language models. Period.

## Building On Prior Work

The best experiments extend earlier findings:
- If raga prediction from lyrics works, *which* features drive it? Can you interpret the decision tree?
- If temporal shifts exist, do they align with known biographical events?
- If two things correlate, can you find a mechanism or a confound?

Read the log. Connect the dots. Go deeper, not wider, when you find something.
