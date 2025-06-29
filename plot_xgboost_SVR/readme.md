| **Feature Name**                               | **What it Measures**                    | **Hypothesis (Why It May Affect CTR)**                                                     |
| ---------------------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------ |
| `title_len` / `subtitle_len`                   | Total number of characters in the field | Longer or shorter text may correlate with user curiosity, clarity, or click fatigue.       |
| `title_word_count` / `subtitle_word_count`     | Total number of words in the field      | Wordiness may impact how quickly a user understands the content or decides to click.       |
| `title_avg_word_len` / `subtitle_avg_word_len` | Average number of characters per word   | Indicates lexical complexity; longer words may signal sophistication or lower readability. |


|                       |                                                                                                                                                                                                                                                                         |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **What it Measures**  | The total number of characters in the title or subtitle.                                                                                                                                                                                                                |
| **Why It Matters**    | This captures how long the text is — which influences readability, visual length on mobile, and the perceived complexity.                                                                                                                                               |
| **CTR Hypothesis**    | Titles that are *too short* may lack context or specificity. Titles that are *too long* may get truncated or be harder to skim. There tends to be a “sweet spot” in character length (e.g. 40–70 chars) that balances clarity and brevity — potentially increasing CTR. |
| **What You Observed** | Clear bell-shaped distributions or correlations across categories. Some categories (like Business or News) may perform better with longer headlines, while others (like Sport or Culture) may favor shorter ones.                                                       |



|                       |                                                                                                                                                                                                                                            |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **What it Measures**  | The total number of *words* in the title or subtitle.                                                                                                                                                                                      |
| **Why It Matters**    | Word count is a close proxy to how digestible or skimmable the text is.                                                                                                                                                                    |
| **CTR Hypothesis**    | Higher CTR might occur in titles with 5–10 words: enough to convey value, but not overwhelming. Very short (1–3 words) or very long (15+ words) titles may underperform. This also interacts with formatting (e.g., listicles vs. quotes). |
| **What You Observed** | Some clear visual separations: mid-length titles had tighter CTR distributions, while long or very short ones showed more variance and occasional drop-offs.                                                                               |


|                       |                                                                                                                                                                                                                                             |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **What it Measures**  | The average number of characters per word.                                                                                                                                                                                                  |
| **Why It Matters**    | A measure of lexical complexity — longer average word lengths often mean more formal, technical, or nuanced language.                                                                                                                       |
| **CTR Hypothesis**    | Titles with *very short words* may feel generic or low-effort. Titles with *very long words* may feel academic or intimidating. Moderate word length may indicate natural, engaging language.                                               |
| **What You Observed** | Strong and smooth CTR patterns: extremely low or high averages had dips in CTR, while moderate values (\~4.5–6.5 characters/word) tended to cluster around higher CTRs. This pattern held across categories like Tech, Food, and Lifestyle. |
