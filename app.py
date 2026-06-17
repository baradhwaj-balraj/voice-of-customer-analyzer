"""
Voice of Customer Feedback Analyzer
====================================

WHAT CHANGED FROM THE FIRST VERSION (read this if you're comparing):

Old approach: dump all feedback into ONE prompt, ask the AI to write a
summary paragraph. Problem: you can't compute anything from a paragraph,
and you lose track of which comment said what.

New approach: two AI calls instead of one.
  PASS 1 (theme discovery): show the AI a sample of feedback, ask it to
    propose a short list of theme names (e.g. "Performance Issues",
    "Integration Requests"). This happens ONCE for the whole dataset so
    every row gets judged against the SAME list of themes, instead of
    each row inventing its own slightly different label.
  PASS 2 (row-by-row classification): for every single feedback row,
    ask the AI "which theme does this belong to? what's the sentiment?
    is it a feature request?" and force the answer into a strict JSON
    format (a structured data format, not a sentence) so Python can
    read it reliably.

Once every row has a theme tag attached to it, everything else -
counting how many rows are in each theme, showing the original
comments behind a theme, calculating a priority score - is just
normal Pandas data work. No more AI calls needed for that part.
This is also why "evidence mapping" (showing the receipts behind
each theme) becomes possible: we never lose the connection between
a row's original text and the theme it got tagged with.
"""

import streamlit as st
import pandas as pd
from openai import OpenAI
import json

# ---------------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------------
# These just control what shows up in the browser tab and at the
# top of the page. Purely cosmetic, no logic here.
st.set_page_config(page_title="Voice of Customer Analyzer", layout="wide")

st.title("Voice of Customer Feedback Analyzer")
st.write(
    "Upload customer feedback to identify themes, sentiment, feature requests, "
    "and which themes deserve attention first."
)

# ---------------------------------------------------------------
# CONNECT TO THE AI
# ---------------------------------------------------------------
# st.secrets reads your API key from a secure config file (secrets.toml)
# rather than writing the key directly in the code. This is standard
# practice so you never accidentally publish your API key on GitHub.
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Which OpenAI model we're using. Keeping it as one variable at the top
# makes it easy to swap models later without hunting through the file.
MODEL_NAME = "gpt-4.1-mini"


# ---------------------------------------------------------------
# FUNCTION: discover_themes
# ---------------------------------------------------------------
# This is PASS 1. We send the AI a SAMPLE of the feedback (not
# necessarily all of it - if you have 5,000 rows, sampling 40-60 is
# usually enough to spot the recurring topics) and ask it to propose
# a short, clean list of theme names.
#
# Why sample instead of sending everything here? Two reasons:
# 1. Cost/speed - this call doesn't need every row, just enough
#    variety to see the patterns.
# 2. We only need a LIST OF LABELS out of this step, not analysis
#    of every comment - that detailed work happens in Pass 2.
def discover_themes(feedback_list, max_themes=8, sample_size=60):
    # If we have more rows than sample_size, randomly sample to keep
    # this call fast and cheap. If we have fewer rows than that, just
    # use all of them.
    if len(feedback_list) > sample_size:
        sample = pd.Series(feedback_list).sample(sample_size, random_state=42).tolist()
    else:
        sample = feedback_list

    # Number each line so the AI's response is easy to read in logs
    # if you ever need to debug it - not required, just a nice touch.
    numbered_feedback = "\n".join(f"{i+1}. {text}" for i, text in enumerate(sample))

    prompt = f"""You are reviewing a sample of customer feedback for a software product.

Identify the {max_themes} or fewer most useful recurring THEMES that would
help a Product Manager understand what's going on. Themes should be broad
categories (e.g. "Performance Issues", "Integration Requests"), not
restatements of individual comments.

Return ONLY a JSON array of theme names, nothing else. Example format:
["Performance Issues", "Integration Requests", "Usability Problems"]

Customer feedback sample:
{numbered_feedback}
"""

    # The API call can fail for reasons outside our control - an invalid
    # or expired API key, hitting OpenAI's rate limit, or a network
    # timeout. Catching this here means the app shows a clear message
    # instead of a raw error traceback if that happens.
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You identify recurring themes in customer feedback. You respond only with valid JSON, no other text."},
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as e:
        st.error(
            "Could not reach the AI service. This is usually a temporary "
            "issue (rate limit or network hiccup) or an API key problem. "
            f"Details: {e}"
        )
        st.stop()

    raw_text = response.choices[0].message.content

    # The AI is asked to return only JSON, but models occasionally wrap
    # it in markdown code fences (```json ... ```). This strips that off
    # if present, so json.loads() doesn't choke on it.
    cleaned = raw_text.strip().strip("`").replace("json\n", "", 1)

    try:
        themes = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: if parsing fails for any reason, give the user a
        # generic theme list rather than crashing the whole app.
        st.warning("Could not parse theme list from AI response. Using a default fallback.")
        themes = ["General Feedback"]

    return themes


# ---------------------------------------------------------------
# FUNCTION: classify_feedback_batch
# ---------------------------------------------------------------
# This is PASS 2. Unlike Pass 1 (which looked at a sample to find
# patterns), this pass needs to look at EVERY row, because every row
# needs its own theme/sentiment/feature-request tag attached to it.
#
# To keep this efficient, we send rows in BATCHES (e.g. 20 at a time)
# instead of one API call per row. One call per row would be slow and
# expensive for large files; one call for ALL rows at once risks the
# AI losing track partway through a very long list. Batching is the
# middle ground.
#
# The key design choice: we ask for a JSON ARRAY back, with one object
# per input row, IN THE SAME ORDER we sent them. That's what lets us
# safely match each AI answer back to the original row afterward.
def classify_feedback_batch(feedback_batch, theme_list):
    numbered_feedback = "\n".join(f"{i+1}. {text}" for i, text in enumerate(feedback_batch))
    theme_options = ", ".join(theme_list)

    prompt = f"""Classify each piece of customer feedback below.

Available themes (choose the single best fit for each item): {theme_options}

For each numbered feedback item, return a JSON object with these fields:
- "theme": one of the available themes listed above
- "sentiment": one of "Positive", "Neutral", "Negative"
- "feature_request": a short feature name if this feedback requests a
  specific feature, otherwise null

Return a JSON array with exactly one object per feedback item, in the
same order as the input. Return ONLY the JSON array, no other text.

Feedback items:
{numbered_feedback}
"""

    # Same reasoning as the theme discovery call above: don't let a
    # transient API issue crash the whole app mid-analysis.
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You classify customer feedback. You respond only with a valid JSON array, no other text."},
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as e:
        st.error(
            "Could not reach the AI service while classifying feedback. "
            f"Details: {e}"
        )
        st.stop()

    raw_text = response.choices[0].message.content
    cleaned = raw_text.strip().strip("`").replace("json\n", "", 1)

    try:
        results = json.loads(cleaned)
    except json.JSONDecodeError:
        # If this batch fails to parse, we fill it with placeholder
        # "Unclassified" entries rather than crashing the whole analysis.
        # This means one bad batch doesn't take down the entire upload.
        results = [
            {"theme": "Unclassified", "sentiment": "Neutral", "feature_request": None}
            for _ in feedback_batch
        ]

    # Safety check: if the AI returned a different number of results
    # than we sent in, something went wrong with matching. Pad or trim
    # so the rest of the app doesn't break trying to line up mismatched
    # lists.
    if len(results) != len(feedback_batch):
        results = results[: len(feedback_batch)] + [
            {"theme": "Unclassified", "sentiment": "Neutral", "feature_request": None}
            for _ in range(len(feedback_batch) - len(results))
        ]

    return results


# ---------------------------------------------------------------
# FUNCTION: calculate_priority
# ---------------------------------------------------------------
# This turns theme data into a transparent priority score. It's
# intentionally simple and SHOWN to the user (not hidden inside an AI
# call) because the point is for a Product Manager to be able to see
# and challenge the formula, not just trust a black box.
#
# This version blends THREE signals instead of just counting mentions:
#
#   1. FREQUENCY - how many times this theme came up. More mentions
#      generally means a more widespread issue.
#   2. CUSTOMER TIER - if the CSV has a "plan_tier" column (Free/Pro/
#      Enterprise), mentions from higher-paying tiers count for more.
#      Reasoning: losing an Enterprise account usually costs the
#      business more than losing a Free user, so a complaint from an
#      Enterprise customer deserves more weight, not just an equal vote.
#   3. DISSATISFACTION - if the CSV has an "nps_score" column (0-10,
#      where lower means more frustrated), mentions tied to a low score
#      count for more. Reasoning: a theme mentioned by someone who is
#      actively unhappy is a stronger signal than the same theme
#      mentioned in passing by someone who's otherwise satisfied.
#
# If the uploaded CSV doesn't have plan_tier or nps_score columns, this
# automatically falls back to frequency-only scoring (exactly like the
# original version) - it never breaks just because optional columns
# are missing.
def calculate_priority(results_df):
    # PLAN_TIER_WEIGHTS: this is the part you'd point to in an
    # interview as a real product decision. We're saying "an Enterprise
    # mention is worth 3x a Free mention, Pro sits in between." These
    # numbers are a starting judgment call, not a universal truth - a
    # different company might weight this differently depending on
    # their actual revenue mix.
    PLAN_TIER_WEIGHTS = {"Enterprise": 3, "Pro": 2, "Free": 1}

    has_tier = "plan_tier" in results_df.columns
    has_nps = "nps_score" in results_df.columns

    # IMPORTANT DESIGN DECISION, read this before changing anything below:
    #
    # An earlier version of this function tried to decide whether a
    # WHOLE THEME counted as "mostly positive" and, if so, excluded it
    # from the priority ranking entirely. That was a mistake. The
    # problem: if 8 people loved a feature and 2 people hit a real bug
    # in it, that's 80% positive - which silently buried the 2 negative
    # comments, even though they didn't stop being true or important
    # just because more people happened to say something nice nearby.
    #
    # Positive and negative feedback are not opposites that cancel out.
    # They're two separate things that can both be true about the same
    # theme at the same time. So this version NEVER lets positive volume
    # suppress negative signal. Instead:
    #   - Priority score is calculated using ONLY the negative mentions
    #     for a theme (their count, customer tier, and dissatisfaction).
    #     Positive and neutral mentions contribute ZERO to the priority
    #     score - they literally cannot dilute or outweigh a problem.
    #   - Positive and negative counts are shown side by side for every
    #     theme, so you can see "8 positive / 2 negative" plainly,
    #     instead of the two being collapsed into one misleading average.
    #   - A theme with ZERO negative mentions naturally scores 0 and
    #     sorts to the bottom - not because we decided to hide it, but
    #     because there's genuinely nothing to fix. That's a real
    #     difference from before: nothing is excluded, it just honestly
    #     scores low when there's no problem to act on.
    priority_rows = []

    for theme, group in results_df.groupby("Theme"):
        total_mentions = len(group)
        sentiment_counts = group["Sentiment"].value_counts()
        positive_count = int(sentiment_counts.get("Positive", 0))
        negative_count = int(sentiment_counts.get("Negative", 0))
        neutral_count = int(sentiment_counts.get("Neutral", 0))

        # Only the negative-sentiment rows feed into the priority
        # calculation below. This is the key fix: praise elsewhere in
        # the same theme has no ability to water this down.
        negative_rows = group[group["Sentiment"] == "Negative"]

        # --- Tier weight: based ONLY on who issued the negative mentions ---
        if has_tier and negative_count > 0:
            tier_weights = negative_rows["plan_tier"].map(PLAN_TIER_WEIGHTS).fillna(1)
            avg_tier_weight = tier_weights.mean()
        else:
            avg_tier_weight = 0  # no negative mentions = no tier signal to weight

        # --- Dissatisfaction weight: based ONLY on NPS scores tied to negative mentions ---
        if has_nps and negative_count > 0:
            nps_values = negative_rows["nps_score"].dropna()
            avg_frustration = (10 - nps_values).mean() if len(nps_values) > 0 else 5
        else:
            avg_frustration = 0 if negative_count == 0 else 5

        priority_rows.append({
            "Theme": theme,
            "Total Mentions": total_mentions,
            "Positive": positive_count,
            "Negative": negative_count,
            "Neutral": neutral_count,
            "_negative_count": negative_count,
            "_avg_tier_weight": avg_tier_weight,
            "_avg_frustration": avg_frustration,
        })

    scored_df = pd.DataFrame(priority_rows)

    # Normalize each raw signal to a 0-100 scale so they're comparable
    # before blending. Note this is now based on NEGATIVE count, not
    # total mentions - a theme with lots of praise but few complaints
    # should score based on how many/how serious the complaints are,
    # not get extra credit for unrelated positive volume.
    max_neg = scored_df["_negative_count"].max()
    scored_df["_freq_score"] = (scored_df["_negative_count"] / max_neg) * 100 if max_neg > 0 else 0

    max_tier = scored_df["_avg_tier_weight"].max()
    scored_df["_tier_score"] = (scored_df["_avg_tier_weight"] / max_tier) * 100 if max_tier > 0 else 0

    max_frustration = scored_df["_avg_frustration"].max()
    scored_df["_frustration_score"] = (
        (scored_df["_avg_frustration"] / max_frustration) * 100 if max_frustration > 0 else 0
    )

    # THE BLEND: same 50/30/20 split as before, but every input now
    # comes exclusively from negative mentions - positive feedback
    # plays no role in this number at all, by design.
    scored_df["Priority Score"] = round(
        (scored_df["_freq_score"] * 0.5)
        + (scored_df["_tier_score"] * 0.3)
        + (scored_df["_frustration_score"] * 0.2)
    )

    # CONFIDENCE CAP: a theme with only 1-2 NEGATIVE mentions can still
    # hit a high score through tier/NPS weighting alone (e.g. one
    # Enterprise complaint = max tier score). Cap low-negative-count
    # themes so they can't reach "High" purely on weighting - they need
    # enough complaints to back it up, not just one outspoken voice.
    LOW_SAMPLE_THRESHOLD = 3  # fewer than this many NEGATIVE mentions = capped
    LOW_SAMPLE_CAP = 65       # cap score so it lands at "Medium" at most

    scored_df["Priority Score"] = scored_df.apply(
        lambda row: min(row["Priority Score"], LOW_SAMPLE_CAP)
        if (0 < row["_negative_count"] < LOW_SAMPLE_THRESHOLD)
        else row["Priority Score"],
        axis=1,
    )

    def label_priority(score):
        if score >= 66:
            return "High"
        elif score >= 33:
            return "Medium"
        elif score > 0:
            return "Low"
        else:
            return "No issues reported"

    scored_df["Priority"] = scored_df["Priority Score"].apply(label_priority)

    scored_df["Confidence"] = scored_df["_negative_count"].apply(
        lambda c: "Low (few complaints)" if 0 < c < LOW_SAMPLE_THRESHOLD
        else ("N/A" if c == 0 else "Adequate")
    )

    display_df = scored_df[
        ["Theme", "Total Mentions", "Positive", "Negative", "Neutral", "Priority Score", "Priority", "Confidence"]
    ]

    return display_df.sort_values("Priority Score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------
# MAIN APP FLOW
# ---------------------------------------------------------------
uploaded_file = st.file_uploader("Upload Feedback CSV", type=["csv"])

if uploaded_file is not None:
    # Reading the CSV can fail if the file is malformed - for example,
    # if a feedback comment contains a comma but isn't wrapped in quotes,
    # Pandas thinks that row has an extra column it wasn't expecting and
    # raises a ParserError instead of guessing what you meant.
    #
    # on_bad_lines="warn" tells Pandas: instead of crashing the whole
    # app over one broken row, skip that row, print a warning in the
    # background, and keep loading everything else. This is what makes
    # the app resilient to messy real-world CSV exports.
    #
    # engine="python" is required for on_bad_lines="warn" to work - the
    # faster default C engine doesn't support this option.
    try:
        df = pd.read_csv(uploaded_file, on_bad_lines="warn", engine="python")
    except Exception as e:
        st.error(
            "Could not read this CSV file. This usually means a feedback "
            "comment contains a comma that isn't wrapped in quotation marks, "
            "confusing the column structure. Try opening the file in a plain "
            "text editor, finding the problem row, and wrapping that cell's "
            "text in double quotes, like: \"Need integration, especially with Salesforce\""
        )
        st.stop()

    st.subheader("Feedback Preview")
    st.dataframe(df)

    if "Feedback" not in df.columns:
        st.error("CSV must contain a column named 'Feedback'.")
    else:
        st.success(f"CSV uploaded successfully. {len(df)} rows found.")

        # Drop empty rows so we don't waste API calls classifying blanks.
        # .dropna() removes rows where Feedback is missing entirely;
        # the str.strip() != "" check also removes rows that are just
        # whitespace, which dropna() alone wouldn't catch.
        df = df[df["Feedback"].notna()]
        df = df[df["Feedback"].astype(str).str.strip() != ""]

        # If every row was empty or blank, there's nothing to analyze.
        # Without this check, the app would try to call the AI with zero
        # feedback and produce a confusing downstream error instead of a
        # clear message.
        if len(df) == 0:
            st.warning("No usable feedback found after removing empty rows. Please check your CSV.")
            st.stop()

        # SAFETY CAP: this app is running on a public demo link using a
        # personal API key. Without a limit, anyone could upload a huge
        # file and run up real costs on that key with no rate limiting.
        # This caps how many rows a single upload will process - large
        # files still preview fine above, they just can't be analyzed
        # past this limit. Raise this number later if you're comfortable
        # with the cost, or remove it entirely once you have proper
        # usage limits set on your OpenAI account.
        MAX_ROWS = 200
        if len(df) > MAX_ROWS:
            st.warning(
                f"This demo is capped at {MAX_ROWS} rows per upload to keep costs "
                f"predictable. Your file has {len(df)} rows after cleaning - only "
                f"the first {MAX_ROWS} will be analyzed."
            )
            df = df.head(MAX_ROWS)

        if st.button("Analyze Customer Feedback"):
            feedback_list = df["Feedback"].astype(str).tolist()

            # --- PASS 1: discover themes ---
            with st.spinner("Step 1 of 2: Identifying themes..."):
                themes = discover_themes(feedback_list)

            st.write("**Themes identified:**", ", ".join(themes))

            # --- PASS 2: classify every row, in batches ---
            BATCH_SIZE = 20
            all_results = []

            progress_bar = st.progress(0, text="Step 2 of 2: Classifying feedback...")
            total_batches = (len(feedback_list) // BATCH_SIZE) + 1

            for batch_num, start in enumerate(range(0, len(feedback_list), BATCH_SIZE)):
                batch = feedback_list[start:start + BATCH_SIZE]
                batch_results = classify_feedback_batch(batch, themes)
                all_results.extend(batch_results)

                progress_bar.progress(
                    (batch_num + 1) / total_batches,
                    text=f"Step 2 of 2: Classifying feedback... ({start + len(batch)}/{len(feedback_list)} rows)"
                )

            progress_bar.empty()

            # --- Attach the AI's per-row answers back onto the original dataframe ---
            # This is the step that makes evidence mapping possible: every
            # row in results_df still has its original "Feedback" text
            # sitting right next to its theme/sentiment/feature_request tag.
            results_df = df.reset_index(drop=True).copy()
            results_df["Theme"] = [r.get("theme", "Unclassified") for r in all_results]
            results_df["Sentiment"] = [r.get("sentiment", "Neutral") for r in all_results]
            results_df["Feature Request"] = [r.get("feature_request") for r in all_results]

            # ---------------------------------------------------------------
            # DISPLAY: Theme frequency + priority
            # ---------------------------------------------------------------
            st.subheader("Theme Frequency & Priority")

            # value_counts() does the actual "quantification" - it counts
            # how many rows landed in each theme. This is plain Pandas,
            # no AI call needed, because the AI already did the hard part
            # (tagging each row) in Pass 2. We still compute this for the
            # bar chart below, even though calculate_priority recalculates
            # counts internally as part of its scoring.
            theme_counts = results_df["Theme"].value_counts()

            # calculate_priority returns one table covering every theme.
            # Priority score is driven ENTIRELY by negative mentions - a
            # theme with lots of praise but no complaints will correctly
            # score 0 and sort to the bottom, but it's never excluded or
            # hidden. Positive, Negative, and Neutral counts are shown as
            # separate columns so praise can never visually cancel out or
            # bury a real complaint sitting in the same theme.
            priority_df = calculate_priority(results_df)

            # Let the user know exactly which signals fed into the score,
            # since the whole point of this formula is to be transparent
            # rather than a black box. If the uploaded CSV doesn't have
            # plan_tier or nps_score columns, this honestly says so instead
            # of silently pretending it used data it didn't have.
            has_tier = "plan_tier" in results_df.columns
            has_nps = "nps_score" in results_df.columns
            if has_tier or has_nps:
                signals_used = ["negative mention count (50%)"]
                if has_tier:
                    signals_used.append("customer tier of complainants (30%)")
                if has_nps:
                    signals_used.append("dissatisfaction / NPS of complainants (20%)")
                st.caption(
                    f"Priority score is based on: {', '.join(signals_used)}. "
                    "Positive feedback is shown for context but never reduces a theme's priority score."
                )
            else:
                st.caption(
                    "Priority score is based on negative mention count only - upload a CSV with "
                    "'plan_tier' and/or 'nps_score' columns to weight by customer impact. "
                    "Positive feedback is shown for context but never reduces a theme's priority score."
                )

            col1, col2 = st.columns([1, 1])
            with col1:
                st.dataframe(priority_df, use_container_width=True, hide_index=True)
            with col2:
                st.bar_chart(theme_counts)

            # ---------------------------------------------------------------
            # DISPLAY: Evidence mapping (drill-down per theme)
            # ---------------------------------------------------------------
            st.subheader("Evidence by Theme")
            st.write("Click a theme below to see the original feedback behind it.")

            # st.expander creates a collapsible section. We make one per
            # theme so the page isn't overwhelming, and the user can open
            # only the themes they care about.
            for theme in theme_counts.index:
                theme_rows = results_df[results_df["Theme"] == theme]
                with st.expander(f"{theme} ({len(theme_rows)})"):
                    for _, row in theme_rows.iterrows():
                        st.markdown(f"- {row['Feedback']}  \n  *Sentiment: {row['Sentiment']}*")

            # ---------------------------------------------------------------
            # DISPLAY: Sentiment breakdown
            # ---------------------------------------------------------------
            st.subheader("Overall Sentiment")
            sentiment_counts = results_df["Sentiment"].value_counts()
            st.bar_chart(sentiment_counts)

            # ---------------------------------------------------------------
            # DISPLAY: Feature requests
            # ---------------------------------------------------------------
            st.subheader("Feature Requests")
            # Pull out only rows where the AI found an actual feature request
            # (not null/None), and show each unique one with how often it
            # came up.
            feature_requests = results_df["Feature Request"].dropna()
            feature_requests = feature_requests[feature_requests.astype(str).str.strip() != ""]

            if len(feature_requests) > 0:
                st.dataframe(
                    feature_requests.value_counts().reset_index().rename(
                        columns={"index": "Feature Request", "Feature Request": "Mentions"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.write("No specific feature requests identified.")

            # ---------------------------------------------------------------
            # DOWNLOAD: full tagged dataset
            # ---------------------------------------------------------------
            # This lets the user take the row-by-row tagged data (theme,
            # sentiment, feature request all attached) into Excel or
            # elsewhere. Useful both as a real feature and as something
            # to point to in a portfolio demo.
            st.subheader("Export")
            csv_export = results_df.to_csv(index=False)
            st.download_button(
                "Download tagged feedback as CSV",
                data=csv_export,
                file_name="tagged_feedback.csv",
                mime="text/csv",
            )

else:
    st.info("Please upload a CSV file to begin.")