# Voice of Customer Feedback Analyzer — v2

## Why this isn't just "paste your CSV into ChatGPT"

A fair critique of an early AI wrapper is: glue all the feedback into
one prompt, ask the AI to summarize it, show the paragraph. That's a
real limitation, so this version is built specifically to answer it.

The difference is architectural, not cosmetic:

- **Every row keeps its identity.** Instead of one summary paragraph,
  every single feedback row gets individually tagged with a theme,
  sentiment, and feature request, and that tag stays attached to the
  original text. That's what makes it possible to click "Performance
  Issues (12)" and see the exact 12 comments behind it, instead of
  just trusting a sentence the AI wrote.
- **Themes are decided once, applied consistently.** A separate pass
  first proposes a clean set of theme names for the whole dataset,
  then every row gets classified against that *same* list. A plain
  summarization prompt has no mechanism to keep theme labels
  consistent row to row.
- **The numbers are real, not described.** Because every row has a
  structured tag, theme counts, sentiment breakdowns, and a
  prioritization score are computed directly from the data with
  Pandas, not generated as prose by the AI and hoped to be accurate.
- **The output is reusable.** The fully tagged dataset can be
  exported as a CSV, something a chat response can't give you without
  manual copy-paste.

In short: pasting feedback into a chat tool gives you an opinion about
your data. This gives you a structured dataset derived from your data,
which a Product Manager can sort, weight, and act on.

## What changed from first version, in plain terms

Your original app did this:
1. Glue all the feedback comments together into one big block of text.
2. Ask the AI to write a paragraph summarizing it.
3. Show that paragraph on the screen.

The problem: once the AI writes a paragraph, there's no way for the app
to *count* anything, *sort* anything, or show you *which comments*
backed up a given theme. You'd have to trust the paragraph and re-read
the original CSV yourself to check it.

This version does it differently:
1. **Pass 1 (theme discovery):** show the AI a sample of the feedback and
   ask it to name the 5-8 main themes (e.g. "Performance Issues").
   This happens once, so every comment gets compared against the
   *same* list of themes instead of each comment getting a slightly
   different made-up label.
2. **Pass 2 (classification):** go through every row of feedback (in
   batches of 20, so it's fast) and ask the AI to tag each one with:
   which theme it belongs to, its sentiment, and whether it's a
   feature request — and get that answer back as structured data
   (JSON), not a sentence.
3. Once every row is tagged, all the counting, sorting, charting, and
   "show me the evidence" drill-down is done by Python directly,
   no AI needed. This is also why it's fast and predictable.

This is the architectural difference between "an AI wrote a summary"
and "a tool that uses AI to structure data, which a PM can then
analyze." That distinction is the core of your portfolio answer to
"why not just paste this into ChatGPT."

## What's new in the app itself

- **Theme Frequency & Priority table** — actual counts per theme, plus
  a transparent 0-100 priority score (currently based on frequency;
  see the comment in `calculate_priority()` in app.py for how to add
  more signals like customer tier or severity if your data has them).
- **Evidence by Theme** — expandable sections, one per theme, showing
  the exact original feedback rows that were tagged with that theme.
- **Sentiment chart** — a simple bar chart instead of just percentages
  in text.
- **Feature Requests table** — deduplicated and counted, instead of a
  flat bullet list.
- **CSV export** — download the full dataset with theme/sentiment/
  feature-request tags attached to every row, so it can be reused
  in Excel or elsewhere.

## How to run it locally

1. Make sure you have a `secrets.toml` file (Streamlit looks for this
   automatically) containing your API key:
   ```
   OPENAI_API_KEY = "your-key-here"
   ```
   This usually goes in a `.streamlit/secrets.toml` file in the same
   folder as `app.py`.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
   streamlit run app.py
   ```
4. Upload `sample_feedback.csv` (included alongside this README) to
   try it out, or use your own CSV as long as it has a column named
   exactly `Feedback`.

## Things worth knowing about how this behaves

- **Batching:** feedback is classified in chunks of 20 rows per AI
  call (see `BATCH_SIZE` near the top of the main app flow). Larger
  files just mean more batches, shown via the progress bar — it won't
  break, just take a bit longer.
- **Error handling:** if the AI's response for a batch doesn't come
  back as valid JSON (rare, but it happens), that batch gets marked
  "Unclassified" instead of crashing the whole app. You'll still get
  results for everything else.
- **Cost:** Pass 1 is one API call total. Pass 2 is roughly
  (number of rows ÷ 20) calls. A 100-row CSV is about 6 calls total,
  not 100 — keeps cost and speed reasonable.

## Natural next steps 

- If real feedback data has columns like customer tier, MRR, or
  ticket severity, wire those into `calculate_priority()` so the
  priority score reflects business impact, not just raw frequency.
  This is flagged with a comment in the code already.
- Persisting themes across multiple uploads (so re-uploading next
  month's feedback compares against the same theme list rather than
  generating a fresh one) is a good "phase 2" to mention even if you
  don't build it — it's the difference between a one-off analysis
  tool and something that builds institutional memory over time.
