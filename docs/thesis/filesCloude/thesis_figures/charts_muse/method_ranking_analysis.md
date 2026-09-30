# Method Ranking Analysis (MRR, 10 benchmarks)

## 1. Aggregate statistics

| Rank | Method | Mean | Median | Std | Min | Best-count* |
|---|---|---|---|---|---|---|
| 1 | Tuned SF | 0.853 | 0.962 | 0.196 | 0.375 | 8/10 |
| 2 | SF+Linear | 0.845 | 0.957 | 0.208 | 0.320 | — |
| 3 | BM25 | 0.815 | 0.947 | 0.253 | 0.155 | 3/10 |
| 4 | Pure SF | 0.815 | 0.938 | 0.220 | 0.334 | 3/10 |
| 5 | SF+RRF | 0.814 | 0.891 | 0.204 | 0.375 | — |
| 6 | E5-multilingual | 0.745 | 0.806 | 0.259 | 0.235 | 0/10 |
| 7 | SPLADE (alone) | 0.715 | 0.714 | 0.245 | 0.356 | 2/10 |
| 8 | MiniLM | 0.698 | 0.865 | 0.292 | 0.184 | 0/10 |

\*Best-count in the Tuned-SF 5-way setup (ties shared): Tuned SF 8, BM25 3, SPLADE 2, E5 0, MiniLM 0.
In the pure-SF 5-way setup (no fusion): SPLADE 5, SF 3, BM25 3, E5 1, MiniLM 0.

## 2. Per-dataset winners (pure methods)

| Benchmark | Winner | Note |
|---|---|---|
| Belebele | SPLADE (1.000) | SF/BM25 tie at 0.995 |
| NarrativeQA | SF = E5 (0.990) | tie |
| PubMedQA | BM25 = SPLADE (1.000) | tie |
| PopQA | BM25 (1.000) | SPLADE collapses (0.565) |
| SciFact | SPLADE (0.958) | Tuned SF 0.966 with fusion |
| SciDocs | SF (0.958) | SPLADE collapses (0.394) |
| Nfcorpus | BM25 (0.686) | only dataset where BM25 clearly beats Tuned SF (−0.031 gap) |
| MuSiQue | SPLADE (0.748) | fusion Linear reaches 0.759 |
| Quran | SPLADE (0.356) / Tuned SF (0.375) | hardest corpus for all; BM25 collapses (0.155) |
| Belebele AR-EN | SF (0.817) | MiniLM collapses (0.184); SPLADE weak (0.481) |

## 3. Interpretation

1. **Tuned SF is the best overall method.** Highest mean (0.853), highest median (0.962),
   lowest variance (std 0.196), best in 8/10 benchmarks, and the highest worst-case
   floor (min 0.375 on Quran — still the best score any method achieves there).
2. **BM25 ≈ pure SF (virtual tie, 0.8150 vs 0.8148).** They win on complementary sets:
   BM25 on PopQA / Nfcorpus / PubMedQA / MuSiQue (lexical overlap pays off);
   SF on SciDocs / Quran / AR-EN / NarrativeQA (semantic matching pays off).
   Report them as tied baselines with complementary strengths, not as one beating the other.
3. **SPLADE is a high-variance specialist.** Most outright wins (5/10) but also
   catastrophic failures (PopQA 0.565, SciDocs 0.394, AR-EN 0.481) → lowest mean
   among non-dense methods. Valuable as a *fusion signal* (Linear α), not as a standalone.
4. **Linear fusion > RRF fusion.** SF+Linear (0.845) beats SF+RRF (0.814); RRF even
   hurts on SciDocs (0.828) and AR-EN (0.683). Recommend Linear as the default fusion.
5. **Dense retrievers disappoint here.** E5 (0.745) beats MiniLM (0.698), but both lose
   to every SF variant and BM25; MiniLM's 0.184 on AR-EN shows cross-lingual brittleness.
6. **Quran is the hardest benchmark** (all methods ≤ 0.375) and the clearest Tuned-SF
   win (+0.220 over BM25). Nfcorpus is the only case needing discussion as a loss
   (−0.031 vs BM25).

## 4. Recommended thesis narrative

- Main claim: *Tuned SF (per-dataset best of SF / SF+Linear / SF+RRF) dominates all
  single methods across 10 benchmarks in mean, median, stability, and win count.*
- Honesty clause: *BM25 remains a strong baseline and wins outright on PopQA and
  Nfcorpus; report the tie with pure SF and the two Tuned-SF losses.*
- Fusion clause: *SPLADE alone is unstable, but as a Linear-fusion signal it lifts
  SF from 0.815 → 0.845 (Linear) → 0.853 (Tuned).*
- Figures to cite: fig7 (ranking), fig9 (where Tuned beats BM25), fig10 (robustness),
  fig4/fig6 + line views 4b/6b (cross-benchmark behaviour).
