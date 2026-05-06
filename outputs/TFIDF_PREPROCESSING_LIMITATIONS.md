# TF-IDF Preprocessing — Limitations and Implications

This document records the preprocessing decisions made for the TF-IDF model pipeline, the steps that were *not* performed, and how this affects the interpretation of the results. It is intended as a transparent companion to the modelling report and as a basis for the "Limitations" section of any thesis or paper using these results.

---

## 1. What the TF-IDF preprocessing pipeline did

The TF-IDF v2 model (`pipeline_tfidf_v2.py`) used the following preprocessing steps:

| Step | Implementation | Effect |
|---|---|---|
| **Lowercasing** | `lowercase=True` in `TfidfVectorizer` | "Erdoğan" and "erdoğan" treated as the same token |
| **Word-level stopword removal** | Custom 150-word Turkish stopword list (`TURKISH_STOPWORDS`) | Removes common function words (ve, için, ile, bir, bu, ...) from word n-grams |
| **Word n-grams (1–2)** | `ngram_range=(1, 2)` | Captures bigrams like "kentsel dönüşüm", "ak parti" |
| **Character n-grams (3–5)** | Separate `TfidfVectorizer` with `analyzer="char_wb"` | Captures Turkish morphological variation (sürdürülebilirlik / sürdürülebilir / sürdürülebilirliği share substrings) |
| **Minimum document frequency filter** | `min_df=3` | Removes terms appearing in fewer than 3 documents (eliminates one-off names, typos, very rare words) |
| **Sublinear TF scaling** | `sublinear_tf=True` | `1 + log(tf)` instead of raw counts; reduces dominance of frequently-repeated words |
| **Class weighting (sustainability, technical)** | Inverse-frequency `sample_weight` per frame | Addresses imbalance in sparse frames |
| **Combined feature matrix** | `hstack([word_vec, char_vec])` | Word and character n-grams concatenated |

These steps are sufficient for a meaningful baseline, and they yielded the highest macro QWK in our experiments (0.603 in 5-fold CV).

---

## 2. What the TF-IDF preprocessing pipeline did *not* do

The following standard preprocessing operations were intentionally or unintentionally omitted:

### 2.1 Named Entity Removal / Normalization

We did **not** apply Named Entity Recognition (NER) to identify and either remove or normalize references to specific people, parties, and institutions.

The top features for the `political` frame include:
- Person names: `erdoğan`, `kılıçdaroğlu`, `özhaseki`, `türkoğlu`
- Party names: `ak parti`
- Institutional names: `tbmm`, `cumhurbaşkanı`, `bakan kurum`
- Honorifics: `sayın`, `inşallah`

In a fully NER-cleaned pipeline, these surface forms would be replaced with abstract tokens such as `<PERSON>`, `<ORG>`, `<HONORIFIC>`, or removed entirely. The model would then have to learn the political frame from contextual and structural cues rather than from the literal names of period-specific actors.

### 2.2 Numeric Token Replacement

We did **not** replace numeric tokens with a placeholder like `<NUM>`. The vocabulary contains:
- Years: `2023`
- Quantities: `bin` (thousand), `doksan` (ninety), `dokuz` (nine)
- Other numerics

In a more rigorous pipeline, all digit sequences would be normalized (`re.sub(r"\d+", " NUM ", text)`) and Turkish written-out numbers would optionally be removed via an extended stopword list.

### 2.3 Lemmatization / Stemming

We did **not** apply morphological lemmatization. Turkish is an agglutinative language with rich suffixation; surface forms of the same lemma can multiply substantially:
- `konut` (housing), `konutta` (in housing), `konutlar` (housings), `konutlarımız` (our housings), `konutlarda` (in housings), ...

Character n-grams (3–5) **partially mitigate** this issue by capturing shared substrings, which is one reason `tfidf_v2` significantly outperformed the word-only `tfidf_v1` baseline. But character n-grams do not produce a true unified representation per lemma; they only smooth over surface variation.

### 2.4 Comprehensive Stopword List

Our custom stopword list contains ~150 high-frequency Turkish function words. Established Turkish NLP resources have larger lists:
- **NLTK Turkish stopwords:** ~360 words
- **spaCy Turkish stopwords:** ~480 words
- **Zeyrek/Hugging Face TR stopword sets:** 200–500 words

A wider stopword list would remove additional low-information words (e.g., `şimdi`, `daha`, `belki`, `ayrıca`) from the word n-gram features.

### 2.5 Stopword Removal in Character N-grams

The `stop_words` parameter in `TfidfVectorizer` only operates on whole-token boundaries; **it does not affect character n-grams**. So while the word "için" is removed from word features, the substring `için` (and its character 3–5 grams) remains present in the character feature space.

### 2.6 Text Cleaning

We did **not** remove or normalize:
- URLs and hyperlinks
- HTML tags or entities
- Emoji
- Excessive punctuation or whitespace
- Mentions and hashtags (if any)

In our specific corpus (institutional press releases and statements), these are likely rare, but the absence of explicit cleaning is worth noting.

---

## 3. Why this matters

The omitted preprocessing steps have three distinct implications.

### 3.1 Temporal portability is reduced

The TF-IDF model has memorized period-specific names and references. A model that scores `political` highly when it sees `erdoğan` will not transfer well to:
- A future Turkish corpus where political leadership has changed
- A corpus from a different country
- A corpus from a different historical period

For example, a 2030 corpus discussing Turkish politics may contain different leader names, in which case the `political` predictions of the current TF-IDF model would degrade. The model is fit to the political surface of February 2023 – December 2024, not to "political discourse" as an abstract concept.

### 3.2 Feature interpretability is conflated with content

When we report top features per frame, names and dates appear alongside genuinely semantic terms. This can mislead a reader into thinking the political frame is about a particular person, when it is more accurately about a *kind of speech act* (campaigning, partisan reference, leadership address). The interpretability of the `political` model is partially an artifact of preprocessing rather than purely a property of the discourse.

### 3.3 Cross-model agreement is partially structural

DistilBERTurk does not use surface tokens directly; it uses contextual embeddings from a pretrained transformer that encode names and concepts in a shared semantic space. The high cross-model Pearson correlation (r = 0.69 – 0.93) on the deployment corpus shows that, *within this corpus*, both models agree. But part of this agreement comes from the fact that political documents in this corpus *do* mention specific names that the TF-IDF model has memorized. If the corpus changed, the agreement would not necessarily hold.

### 3.4 What this *does not* invalidate

The findings reported in the modelling report and the visualization analysis remain valid for the following reasons:

1. **The corpus and the models are matched in time.** The 300 labelled training documents and the 2139 deployment documents come from the same period, the same political environment, and the same set of source institutions. Any concept drift between training and deployment is minimal.

2. **The convergent validity result is robust.** Two architecturally distinct models (sparse linear model vs. fine-tuned transformer) produce highly correlated predictions across all four frames. Even granting that part of this agreement is "structural" (because the same names cue both models), the agreement on `sustainability` (r = 0.69) — where almost no person names appear in the top features — supports the substantive conclusion that the frames capture genuine signal.

3. **The substantive findings are about within-period dynamics.** The visualizations (period bars, monthly time series, province × frame heatmaps, source-type violins) describe how discourse shifts *within* the studied window. These shifts are not artifacts of name memorization, because the names are present throughout the window; the variation comes from how often and in what combinations they are used.

---

## 4. How to write this up in a thesis or paper

Below are two suggested phrasings, depending on the depth of treatment desired.

### 4.1 Brief mention (single paragraph in Limitations)

> Our TF-IDF pipeline applied lowercasing, a 150-word Turkish stopword list, and combined word (1–2 gram) and character (3–5 gram) features with inverse-frequency class weighting. We did not apply named entity normalization, numeric token replacement, or lemmatization. Consequently, the TF-IDF model's feature importances for the political frame are heavily influenced by specific actors and institutional names from the studied period (e.g., "erdoğan", "kılıçdaroğlu", "ak parti", "tbmm"). This limits the temporal portability of the trained TF-IDF model: applied to a corpus from a different political moment, the political frame predictions would degrade. The DistilBERTurk model is less affected by this issue because contextual embeddings encode semantic role rather than surface form. The strong cross-model agreement (Pearson r = 0.69 – 0.93) on the deployment corpus, particularly for the sustainability frame where person names are largely absent from the feature space, indicates that this preprocessing limitation does not bias the substantive findings reported within the studied period.

### 4.2 Extended mention (subsection)

**4.2.1 Preprocessing decisions and their consequences**

> The TF-IDF pipeline applied four preprocessing operations: lowercasing, removal of approximately 150 high-frequency Turkish stopwords from word n-grams, a minimum document frequency filter of 3, and sublinear term frequency scaling. Word features (1–2 grams) and character features (3–5 grams) were extracted in parallel and concatenated. Character n-grams were chosen specifically to address Turkish morphology — the language is agglutinative, and a single lemma may surface as ten or more variants in a corpus of this size. Substring features partially compensate for the absence of true lemmatization, which is a meaningful improvement in low-resource settings where reliable Turkish lemmatizers are not consistently available.

> Several standard preprocessing operations were not applied. We did not apply Named Entity Recognition to identify and normalize references to specific people, parties, or institutions. As a result, the most predictive features for the political frame include the names of national political figures active during the studied window (e.g., "erdoğan", "kılıçdaroğlu", "özhaseki"), the dominant parliamentary party ("ak parti"), and key institutions ("tbmm", "cumhurbaşkanı"). Numeric tokens, including the year "2023", were retained rather than replaced with a placeholder. The stopword list is also smaller than the established NLTK or spaCy lists for Turkish; a more aggressive list would have removed additional low-information word features, though the impact on character n-grams would be unchanged since `TfidfVectorizer`'s stopword mechanism does not operate at the substring level.

> The principal consequence of these choices is reduced temporal portability. The TF-IDF model is, in effect, fit to the political surface of Turkey between February 2023 and December 2024, not to a temporally invariant notion of "political discourse." A future application of this trained model to a corpus from a different political moment, or to a different country, would produce degraded predictions on the political frame in particular. The technical, development, and sustainability frames are less affected, because their top features are dominated by domain vocabulary ("kentsel dönüşüm", "geçici barınma", "sıfır atık") rather than person names.

> The DistilBERTurk model is structurally less vulnerable to this limitation. Contextual transformer embeddings encode the semantic and syntactic role of a name within a sentence, rather than treating the name as an opaque token. The strong cross-model agreement on the deployment corpus (Pearson r = 0.69 – 0.93 across frames) is therefore informative: it indicates that on this corpus, the two models converge despite their very different relationships to surface tokens. The agreement is particularly meaningful on the sustainability frame, where person names are largely absent from the feature space, suggesting that both models capture genuine discourse signal rather than purely lexical artifacts.

> A version of the analysis with stricter TF-IDF preprocessing — wider stopword list, NER-based name removal, numeric token replacement, and optionally lemmatization via a Turkish morphological analyzer such as Zeyrek — was not pursued in this work due to time constraints, but is a natural extension. Based on the cross-model agreement observed here, we expect such a pipeline to yield modest improvements in macro QWK (perhaps +0.02 – 0.05) and substantially improved temporal portability, without altering the substantive findings on within-period discourse dynamics.

---

## 5. If a stricter pipeline becomes necessary

Should a reviewer or examiner request a more rigorously preprocessed TF-IDF model, the following steps would be needed:

1. **Replace numeric tokens.** Apply `re.sub(r"\d+", " NUM ", text)` to all input texts before vectorization.

2. **Wider stopword list.** Replace the 150-word custom list with NLTK's 360-word Turkish stopword list, optionally augmented with corpus-specific function words observed during inspection.

3. **NER-based name removal.** Use a Turkish NER model — for example, the `savasy/bert-base-turkish-ner-cased` model on Hugging Face, or the `stanza` package's Turkish NER pipeline — to identify spans labelled `PER`, `ORG`, and `LOC`. Replace each span with the corresponding placeholder (`<PER>`, `<ORG>`, `<LOC>`) in the input text. This adds a preprocessing dependency and adds 1–2 minutes per 1000 documents to the preprocessing step.

4. **Optional lemmatization.** Apply Turkish morphological analysis with `zeyrek` or `Zemberek-Python`. Lemmatization quality is variable for Turkish (estimated 80–90% token-level accuracy on news text), and lemmatization may interact poorly with the existing character n-gram features, so this step should be evaluated carefully.

5. **Re-fit, re-evaluate, re-score.** Re-run `pipeline_tfidf_v2.py` with the new preprocessing on the same 300-document training set, re-evaluate under 5-fold CV, and re-score the deployment corpus. Compare the new top-features list to the original and verify that the political frame's top features now contain more semantic markers and fewer surface-form names.

The DistilBERTurk pipeline does not require any change, since the transformer tokenizer and the model are independent of these surface-level decisions.

---

## 6. Summary

The TF-IDF preprocessing in this project is **deliberately lightweight**: lowercasing, a small stopword list, character n-grams for morphology, and class weighting for imbalance. This is sufficient for a competitive baseline within the studied period (macro QWK = 0.603, the highest among all tested models), but it does not produce a temporally portable model. The DistilBERTurk model, which uses contextual embeddings, is more robust to this limitation, and the strong agreement between the two models within the studied corpus indicates that the substantive findings are not artifacts of preprocessing choices.

For the present analysis — describing how earthquake-related discourse evolved across Turkey between February 2023 and December 2024 — the preprocessing pipeline is fit for purpose. For any extension that requires applying these models to a different time period, region, or political environment, a stricter preprocessing pipeline (Section 5) is recommended.
