# GST AI Roles and Usage Review

**Date:** January 2026  
**Purpose:** Confirm each AI is used for the right task and leverage its strengths. Recommend prompt changes only where needed.

---

## 1. Grok (xAI) – Narrative Momentum Analyst

**Intended role:** Detect emerging/accelerating narratives from X (Twitter), social momentum, and attention shifts that can move markets.

**Current usage:**
- **Scanners (agent):** `scan_sentiment(tickers)` and `scan_market_overview()` run once per cycle. Prompts were generic (“Analyze market sentiment”, “Analyze current US stock market outlook”) with no explicit X/Twitter or narrative framing.
- **MACA thesis:** The Grok “thesis” is produced by an adapter that calls `scan_market_overview()` **again** and turns the best signal into a proposal. So Grok’s API is hit 3× per cycle (sentiment, overview in agent, overview again in MACA). When the third call returns [], we use a fallback from technical context.

**Gaps:**
- Outlook and sentiment prompts did not explicitly ask for X/Twitter or narrative momentum, so Grok was not steered toward its differentiator.
- Duplicate `scan_market_overview()` in MACA is redundant and fragile (see Architecture Redesign).

**Changes made:**
- **Prompts:** `_build_simple_sentiment_prompt` and `_build_simple_outlook_prompt` now explicitly ask Grok to use X/Twitter and public discourse, and to surface narrative/attention/momentum. JSON schema for outlook extended with optional `narrative_themes` and `attention_shift`.
- **Architecture (future):** Per ARCHITECTURE_REDESIGN_PROPOSAL.md, Grok should run once in Collect; the “Grok analyst” in Council should only consume the shared signal digest and produce a narrative-momentum thesis without calling the scanner again.

**Verdict:** Grok is the right model for narrative/social momentum. Use it explicitly for X and narrative; remove duplicate scanner call when we implement the redesign.

---

## 2. Perplexity (Sonar Pro) – External Reality Analyst

**Intended role:** Web-backed, citation-rich facts: news, filings, earnings, macro data. Strong recency (e.g. last 6 hours).

**Current usage:**
- `generate_thesis()` receives `additional_context` = the committee’s SIGNAL INVENTORY (FRED, Polymarket, Event signals). Prompt already requires “only information published within the LAST 6 HOURS” and “cite sources (URLs)”.
- Perplexity’s API uses live search; we do leverage its web-browsing ability.

**Gaps:**
- The prompt did not explicitly state that the signal inventory is **input from other sources** and that Perplexity’s job is to **use web search to validate, contradict, or extend** those signals and to find additional catalysts. Making that explicit improves focus and citation quality.

**Changes made:**
- Prompt updated to state clearly: “The SIGNAL INVENTORY below is from other committee sources (FRED, Polymarket, Events). Use your web search to (1) validate or contradict these, (2) find additional catalysts from the last 6 hours. Cite URLs for key claims.”

**Swap to Gemini?**  
Perplexity is well-suited for this role (web search + citations + recency). Gemini with Google Search grounding could substitute if we preferred one fewer vendor or Google’s index; the same role and prompt structure would apply. No change recommended unless we want to consolidate providers.

**Verdict:** Perplexity is used correctly. Prompt tightened so its web search is explicitly tied to the committee’s signal inventory.

---

## 3. ChatGPT (GPT-4o) – Sentiment & Cognitive Bias Analyst

**Intended role:** Market psychology, sentiment regime, bias detection (herding, overconfidence, narrative exhaustion). Does **not** browse the web or read charts.

**Current usage:**
- `generate_thesis()` receives `market_context` (combined) and `additional_context` (signal inventory). Prompt clearly restricts: “You do NOT browse the web. You do NOT analyze charts.” It asks for sentiment/bias view and outputs like `bias_flags` and `what_you_might_be_missing`.

**Gaps:**
- Minor: Emphasize that `bias_flags` and a contrarian/“what you might be missing” view are **required** outputs so the committee consistently gets a bias check.

**Changes made:**
- One sentence added: “You MUST list bias_flags (e.g. herding, overconfidence, recency) and provide what_you_might_be_missing as a contrarian consideration.”

**Verdict:** ChatGPT is used correctly. Small prompt tweak to stress bias and contrarian output.

---

## 4. Claude (Chair + Technical Validator)

**Chair:** Synthesizes proposals and debate into a single decision. Does not browse or propose trades independently.  
**Technical validator:** Chart/structure only (supports/weakens/invalidates).  
No prompt changes made; roles are clear and consistent with the design.

---

## 5. Summary

| AI        | Role                  | Leveraged correctly?     | Change |
|----------|------------------------|--------------------------|--------|
| Grok     | Narrative momentum (X) | Partially; prompts generic | Yes – prompts now X/narrative; redesign will remove duplicate call |
| Perplexity | Web facts, citations, 6hr | Yes                      | Yes – prompt ties web search to signal inventory |
| ChatGPT  | Sentiment, bias        | Yes                      | Yes – stress bias_flags and contrarian view |
| Claude   | Chair + technical      | Yes                      | No     |

No need to swap Perplexity for Gemini unless we want to consolidate on Google; Perplexity’s browsing is leveraged and now explicitly framed.
