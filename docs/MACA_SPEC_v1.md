# Multi-Agent Consensus Architecture (MACA) Specification

**Version:** 1.0.0
**Status:** Draft
**Last Updated:** January 2026
**Companion Documents:** `FRAMEWORK.md`, `IMPLEMENTATION_SPEC_v1.md`, `GROK_SIGNAL_AGENT_SPEC_v1_1.md`

---

## 1. Executive Summary

The Multi-Agent Consensus Architecture (MACA) evolves Gann Sentinel Trader from a single-source signal system to a diversified multi-AI analyst model. Three AI systems (Grok, Perplexity, ChatGPT) independently generate investment theses, which Claude synthesizes and validates before peer review.

### Core Innovation

Instead of relying on one AI for signal generation (Grok) and another for decision-making (Claude), MACA creates a "Council of AI Analysts" where:

1. **Multiple perspectives** reduce echo chamber risk
2. **Independent theses** surface non-obvious opportunities
3. **Peer review** catches errors and blind spots
4. **Claude as CIO** has final authority on all decisions

### Problem Solved

The original architecture surfaced the same stocks repeatedly (TSLA, NVDA, PLTR, MSTR) because Grok was scanning X/Twitter, which has concentrated retail sentiment around popular tech stocks. MACA diversifies signal sources to capture:

- **Grok**: Social sentiment, trending narratives, retail momentum
- **Perplexity**: Fundamental research, citations, financial data
- **ChatGPT**: Pattern recognition, market structure, risk scenarios

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SCHEDULED SCAN TRIGGER                            │
│                        (Every 60 min during market hours)                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              PHASE 1: PARALLEL THESIS GENERATION (Async)                    │
│                                                                             │
│   Each AI receives identical context:                                       │
│   • Current portfolio positions and P&L                                     │
│   • Available cash                                                          │
│   • Recent signals from FRED, Polymarket                                    │
│   • Request: "Propose your highest conviction opportunity"                  │
│                                                                             │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │
│   │    GROK     │     │ PERPLEXITY  │     │   CHATGPT   │                   │
│   │   (xAI)     │     │ (Sonar Pro) │     │  (GPT-4o)   │                   │
│   │             │     │             │     │             │                   │
│   │ Strengths:  │     │ Strengths:  │     │ Strengths:  │                   │
│   │ • X/Twitter │     │ • Citations │     │ • Patterns  │                   │
│   │ • Sentiment │     │ • Financial │     │ • Risk      │                   │
│   │ • Momentum  │     │ • Research  │     │ • Scenarios │                   │
│   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘                   │
│          │                   │                   │                          │
│          ▼                   ▼                   ▼                          │
│   ThesisProposal      ThesisProposal      ThesisProposal                    │
└──────────┼───────────────────┼───────────────────┼──────────────────────────┘
           │                   │                   │
           └───────────────────┼───────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 2: SYNTHESIS (CLAUDE)                          │
│                                                                             │
│   Claude as "Chief Investment Officer":                                     │
│                                                                             │
│   1. Receives all 3 thesis proposals with raw data                          │
│   2. Cross-references against:                                              │
│      • FRED macro indicators                                                │
│      • Polymarket probability shifts                                        │
│      • Technical chart analysis (5-year weekly)                             │
│      • Current portfolio state                                              │
│   3. Evaluates fund repositioning opportunities                             │
│   4. Selects best proposal OR synthesizes new recommendation                │
│   5. Assigns final conviction score                                         │
│                                                                             │
│   Output: SynthesisDecision (conviction >= 80 → proceed to review)          │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            │                                                   │
            ▼ (conviction >= 80)                                ▼ (conviction < 80)
┌───────────────────────────────────┐           ┌───────────────────────────────┐
│      PHASE 3: PEER REVIEW         │           │     WAIT FOR NEXT SCAN        │
│                                   │           │                               │
│   Claude's recommendation sent    │           │   Log: "No opportunities      │
│   to all 3 AIs for validation:    │           │   above 80 threshold"         │
│                                   │           │                               │
│   Each AI provides:               │           │   Exception: If Claude        │
│   • APPROVE or REJECT             │◄──────────│   identifies TIME-SENSITIVE   │
│   • Specific concerns             │           │   catalyst, may restart       │
│   • Confidence adjustment (-10    │           │   immediately                 │
│     to +10)                       │           │                               │
│                                   │           │                               │
└───────────────┬───────────────────┘           └───────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   PHASE 4: FINAL DECISION (CLAUDE)                          │
│                                                                             │
│   Claude consolidates peer reviews:                                         │
│                                                                             │
│   • Reviews are ADVISORY, not binding                                       │
│   • Claude has FINAL AUTHORITY                                              │
│   • Adjusts conviction based on concerns raised                             │
│   • Documents dissenting opinions in trade record                           │
│                                                                             │
│   Decision Matrix:                                                          │
│   ┌─────────────────┬───────────────────────────────────────────┐           │
│   │ Review Outcome  │ Claude's Typical Response                 │           │
│   ├─────────────────┼───────────────────────────────────────────┤           │
│   │ 3/3 APPROVE     │ Proceed with confidence                   │           │
│   │ 2/3 APPROVE     │ Proceed, note minority concern            │           │
│   │ 1/3 APPROVE     │ Reduce conviction, likely NO_TRADE        │           │
│   │ 0/3 APPROVE     │ NO_TRADE unless Claude overrides          │           │
│   └─────────────────┴───────────────────────────────────────────┘           │
│                                                                             │
│   Claude may override consensus if:                                         │
│   • Concerns are based on outdated information                              │
│   • Dissenting AI misunderstood the thesis                                  │
│   • Risk/reward still favorable after adjustment                            │
│                                                                             │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              EXISTING: Risk Engine → Human Approval → Execute               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Schemas

### 3.1 Thesis Proposal Schema

Each AI must return a structured thesis proposal:

```json
{
  "schema_version": "1.0.0",
  "proposal_id": "<uuid-v4>",
  "ai_source": "<grok | perplexity | chatgpt>",
  "timestamp_utc": "<ISO-8601>",
  "scan_cycle_id": "<uuid of current scan>",

  "proposal_type": "<NEW_BUY | SELL | ROTATE | NO_OPPORTUNITY>",

  "recommendation": {
    "ticker": "<symbol or null>",
    "side": "<BUY | SELL | null>",
    "conviction_score": "<integer 0-100>",
    "thesis": "<1-3 sentence investment thesis>",
    "time_horizon": "<intraday | days | weeks | months>",
    "catalyst": "<what event or trend drives this>",
    "catalyst_deadline": "<ISO-8601 or null if no deadline>"
  },

  "rotation_details": {
    "sell_ticker": "<ticker to sell, if ROTATE>",
    "sell_rationale": "<why selling this position>",
    "expected_net_gain": "<estimated $ improvement>"
  },

  "supporting_evidence": {
    "key_signals": [
      {
        "signal_type": "<sentiment | fundamental | technical | prediction>",
        "summary": "<brief description>",
        "source": "<where this came from>",
        "confidence": "<high | medium | low>"
      }
    ],
    "bull_case": "<key bullish factors>",
    "bear_case": "<key bearish factors>",
    "risks": ["<risk 1>", "<risk 2>"]
  },

  "raw_data": {
    "search_queries_used": ["<query 1>", "<query 2>"],
    "sources_consulted": ["<source 1>", "<source 2>"],
    "data_points": {}
  },

  "time_sensitive": "<boolean - true if catalyst has deadline>",

  "metadata": {
    "model": "<model version used>",
    "latency_ms": "<integer>",
    "tokens_used": "<integer>"
  }
}
```

### 3.2 Synthesis Decision Schema

Claude's output after evaluating all proposals:

```json
{
  "schema_version": "1.0.0",
  "synthesis_id": "<uuid-v4>",
  "scan_cycle_id": "<uuid>",
  "timestamp_utc": "<ISO-8601>",

  "decision_type": "<TRADE | NO_TRADE | WATCH>",

  "selected_proposal": {
    "proposal_id": "<uuid of chosen proposal or null>",
    "ai_source": "<which AI's proposal was selected>",
    "modifications": "<what Claude changed, if any>"
  },

  "recommendation": {
    "ticker": "<symbol>",
    "side": "<BUY | SELL>",
    "conviction_score": "<integer 0-100>",
    "thesis": "<Claude's refined thesis>",
    "position_size_pct": "<float 0-25>",
    "stop_loss_pct": "<float>",
    "time_horizon": "<intraday | days | weeks | months>"
  },

  "cross_validation": {
    "fred_alignment": "<supports | neutral | conflicts>",
    "polymarket_alignment": "<supports | neutral | conflicts>",
    "technical_alignment": "<supports | neutral | conflicts>",
    "notes": "<specific observations>"
  },

  "proposal_evaluation": [
    {
      "ai_source": "grok",
      "proposal_id": "<uuid>",
      "evaluation": "<selected | rejected | partially_used>",
      "reason": "<why this decision>"
    },
    {
      "ai_source": "perplexity",
      "proposal_id": "<uuid>",
      "evaluation": "<selected | rejected | partially_used>",
      "reason": "<why this decision>"
    },
    {
      "ai_source": "chatgpt",
      "proposal_id": "<uuid>",
      "evaluation": "<selected | rejected | partially_used>",
      "reason": "<why this decision>"
    }
  ],

  "proceed_to_review": "<boolean>",
  "time_sensitive_override": "<boolean - true if restarting due to catalyst>"
}
```

### 3.3 Peer Review Schema

Each AI's review of Claude's recommendation:

```json
{
  "schema_version": "1.0.0",
  "review_id": "<uuid-v4>",
  "proposal_id": "<uuid of Claude's synthesis>",
  "scan_cycle_id": "<uuid>",
  "reviewer_ai": "<grok | perplexity | chatgpt>",
  "timestamp_utc": "<ISO-8601>",

  "verdict": "<APPROVE | REJECT>",

  "confidence_adjustment": "<integer -10 to +10>",

  "review_details": {
    "agrees_with_thesis": "<boolean>",
    "concerns": ["<concern 1>", "<concern 2>"],
    "additional_risks": ["<risk not mentioned>"],
    "missing_information": ["<what should have been considered>"],
    "alternative_view": "<different interpretation, if any>"
  },

  "validation_checks": {
    "facts_verified": "<boolean>",
    "timing_appropriate": "<boolean>",
    "risk_reward_acceptable": "<boolean>"
  },

  "raw_response": "<full AI response for audit>"
}
```

---

## 4. Fund Repositioning Logic

### 4.1 When to Consider Repositioning

Repositioning (ROTATE) should be evaluated when:

1. **New opportunity has higher conviction** than existing position
2. **Existing position is underperforming** relative to thesis
3. **Catalyst has passed** for existing position
4. **Better risk/reward** available elsewhere

### 4.2 Repositioning Calculation

```python
def evaluate_rotation(current_position, new_opportunity):
    """
    Evaluate whether rotating from current position to new opportunity
    creates positive expected value.
    """

    # Current position metrics
    current_unrealized_pnl = current_position.unrealized_pnl
    current_unrealized_pct = current_position.unrealized_pnl_pct
    current_conviction = current_position.thesis_conviction  # May have decayed

    # New opportunity metrics
    new_conviction = new_opportunity.conviction_score
    new_expected_return = estimate_return(new_opportunity)

    # Transaction costs
    sell_cost = estimate_sell_cost(current_position)  # Slippage, spread
    buy_cost = estimate_buy_cost(new_opportunity)
    total_transaction_cost = sell_cost + buy_cost

    # Calculate net expected value
    # If current position is profitable, we're "giving up" some gains
    # If current position is losing, we're "cutting losses"

    if current_unrealized_pnl > 0:
        # Profitable position - need higher conviction to rotate
        opportunity_cost = current_unrealized_pnl * (1 - current_conviction/100)
    else:
        # Losing position - more willing to rotate
        opportunity_cost = current_unrealized_pnl * 0.5  # Treat as sunk cost

    net_expected_value = (
        new_expected_return
        - total_transaction_cost
        - opportunity_cost
    )

    return {
        "should_rotate": net_expected_value > 0 and new_conviction >= 80,
        "net_expected_value": net_expected_value,
        "conviction_improvement": new_conviction - current_conviction,
        "rationale": generate_rationale(...)
    }
```

### 4.3 Example: BTC → GOOG Rotation

```
Current Position:
  Asset: BTC
  Entry: $95,000
  Current: $94,930
  Unrealized P&L: -$70 (-0.07%)
  Original Conviction: 75 (below threshold)

New Opportunity:
  Asset: GOOG
  Conviction: 87
  Expected Return: 8% (30-day)

Transaction Costs:
  Sell BTC: ~$15 (slippage)
  Buy GOOG: ~$10 (spread)
  Total: $25

Calculation:
  Expected GOOG Return: $1,000 * 0.08 = $80
  Transaction Cost: -$25
  Realized Loss: -$70

  Net Expected: $80 - $25 - $70 = -$15 (Day 0)

  But: GOOG moved +$120 on Day 1
  Actual Net: +$120 - $25 - $70 = +$25

Decision: ROTATE was correct
```

---

## 5. AI-Specific Prompting

### 5.1 Grok Prompt Template

```
You are a Signal Intelligence Analyst specializing in social sentiment and market momentum.

CURRENT CONTEXT:
- Date: {current_date}
- Portfolio: {portfolio_summary}
- Available Cash: ${cash}

YOUR TASK:
Scan X/Twitter and web sources for the single highest-conviction investment opportunity right now.

CONSIDER:
1. Trending narratives and sentiment shifts
2. Retail momentum and positioning
3. Breaking news with market implications
4. Unusual volume or attention patterns

OUTPUT:
Return a JSON object matching the ThesisProposal schema.
Focus on opportunities others might be missing.
If proposing a ROTATE, explain why the new opportunity beats our current positions.

CONSTRAINTS:
- Conviction must be 0-100 (honest assessment)
- Include specific sources and signals
- Note if this is time-sensitive
```

### 5.2 Perplexity Prompt Template

```
You are a Fundamental Research Analyst with access to comprehensive financial data.

CURRENT CONTEXT:
- Date: {current_date}
- Portfolio: {portfolio_summary}
- Available Cash: ${cash}

YOUR TASK:
Research and identify the single highest-conviction investment opportunity based on fundamental analysis.

CONSIDER:
1. Recent earnings, guidance, or financial developments
2. Valuation relative to peers and historical norms
3. Industry trends and competitive positioning
4. Institutional activity and analyst sentiment

OUTPUT:
Return a JSON object matching the ThesisProposal schema.
CITE YOUR SOURCES - include URLs or specific data points.
If proposing a ROTATE, provide fundamental rationale.

CONSTRAINTS:
- Conviction must be 0-100 (honest assessment)
- Every claim needs supporting evidence
- Highlight any data gaps or uncertainties
```

### 5.3 ChatGPT Prompt Template

```
You are a Quantitative Strategist specializing in market patterns and risk analysis.

CURRENT CONTEXT:
- Date: {current_date}
- Portfolio: {portfolio_summary}
- Available Cash: ${cash}
- Recent Market Conditions: {market_summary}

YOUR TASK:
Identify the single highest-conviction investment opportunity based on pattern recognition and risk/reward analysis.

CONSIDER:
1. Technical patterns and market structure
2. Historical analogues and what they suggest
3. Risk scenarios (best case, base case, worst case)
4. Portfolio correlation and diversification

OUTPUT:
Return a JSON object matching the ThesisProposal schema.
Quantify risk/reward where possible (expected value, Sharpe-like metrics).
If proposing a ROTATE, explain risk reduction or return enhancement.

CONSTRAINTS:
- Conviction must be 0-100 (honest assessment)
- Include probability estimates where meaningful
- Flag any model uncertainty or assumption sensitivity
```

### 5.4 Peer Review Prompt (All AIs)

```
You are reviewing a proposed trade recommendation. Your role is to validate or challenge it.

PROPOSED TRADE:
{synthesis_recommendation}

ORIGINAL THESIS:
{thesis}

SUPPORTING DATA:
{supporting_evidence}

YOUR TASK:
1. Verify the factual claims made
2. Assess whether the thesis is sound
3. Identify risks that may have been overlooked
4. Determine if the timing is appropriate

OUTPUT:
Return a JSON object matching the PeerReview schema.

VERDICT OPTIONS:
- APPROVE: You agree the trade should proceed
- REJECT: You believe the trade should not proceed

CONFIDENCE ADJUSTMENT:
- Provide a number from -10 to +10
- Negative if you think conviction should be lower
- Positive if you think conviction should be higher
- Zero if you agree with the stated conviction

Be specific about concerns. Vague objections are not helpful.
```

---

## 6. Integration Points

### 6.1 New Environment Variables

```bash
# Perplexity API
PERPLEXITY_API_KEY=pplx-xxx
PERPLEXITY_MODEL=sonar-pro

# OpenAI (ChatGPT)
OPENAI_API_KEY=sk-xxx
OPENAI_MODEL=gpt-4o

# Existing
XAI_API_KEY=xxx  # Grok
ANTHROPIC_API_KEY=xxx  # Claude
```

### 6.2 New Database Tables

See `database.py` for full schemas:

- `ai_proposals` - Stores each AI's thesis proposal
- `ai_reviews` - Stores peer review responses
- `scan_cycles` - Tracks multi-agent scan rounds

### 6.3 Modified Scan Cycle

```python
async def run_maca_scan_cycle():
    """Execute a full MACA scan cycle."""

    cycle_id = create_scan_cycle()

    # Phase 1: Parallel thesis generation
    proposals = await asyncio.gather(
        grok_analyst.generate_thesis(context),
        perplexity_analyst.generate_thesis(context),
        chatgpt_analyst.generate_thesis(context)
    )

    save_proposals(cycle_id, proposals)

    # Phase 2: Claude synthesis
    synthesis = await claude_analyst.synthesize(
        proposals=proposals,
        fred_data=get_fred_signals(),
        polymarket_data=get_polymarket_signals(),
        technical_data=get_technical_analysis(),
        portfolio=get_portfolio_state()
    )

    if synthesis.conviction < 80 and not synthesis.time_sensitive_override:
        return log_no_trade(cycle_id, synthesis)

    # Phase 3: Peer review
    reviews = await asyncio.gather(
        grok_analyst.review_proposal(synthesis),
        perplexity_analyst.review_proposal(synthesis),
        chatgpt_analyst.review_proposal(synthesis)
    )

    save_reviews(cycle_id, reviews)

    # Phase 4: Final decision
    final_decision = await claude_analyst.finalize(
        synthesis=synthesis,
        reviews=reviews
    )

    if final_decision.proceed:
        return submit_to_risk_engine(final_decision)
    else:
        return log_no_trade(cycle_id, final_decision)
```

---

## 7. Phase 2 Implementation Checklist

### Week 1: Telegram Logging + Perplexity Integration

- [x] Add telegram_messages table to database
- [x] Update TelegramBot to log all activity
- [x] Add /logs command
- [x] Create PerplexityAnalyst class
- [ ] Test Perplexity API integration
- [ ] Run dual-source (Grok + Perplexity) test cycles

### Week 2: ChatGPT Integration + Synthesis

- [x] Create ChatGPTAnalyst class
- [x] Update Claude prompts for multi-proposal synthesis
- [x] Implement proposal evaluation logic
- [ ] Test three-source thesis generation

### Week 3: Peer Review + Repositioning

- [x] Implement peer review prompts
- [x] Add review logging and storage
- [ ] Build repositioning calculation
- [ ] Test full MACA cycle end-to-end

### Week 4: Validation + Go-Live

- [ ] Run parallel paper trading (old vs MACA)
- [ ] Compare signal diversity metrics
- [ ] Tune conviction thresholds
- [ ] Deploy MACA to production

---

## 8. Success Metrics

### Signal Diversity

| Metric | Phase 1 (Current) | MACA Target |
|--------|-------------------|-------------|
| Unique tickers per week | 3-5 | 10-15 |
| Source agreement rate | N/A | 30-50% |
| Non-tech stock ratio | <20% | >40% |

### Decision Quality

| Metric | Measurement |
|--------|-------------|
| Consensus accuracy | % of 3/3 APPROVE trades that are profitable |
| Dissent value | % of 1/3 APPROVE rejections that would have lost |
| Override success | % of Claude overrides that outperform |

### System Health

| Metric | Target |
|--------|--------|
| Cycle completion rate | >95% |
| Average cycle time | <90 seconds |
| API error rate | <5% |

---

## 9. Risk Considerations

### API Costs

| Service | Estimated Cost/Cycle | Monthly (30 cycles/day) |
|---------|---------------------|------------------------|
| Grok | ~$0.05 | ~$45 |
| Perplexity | ~$0.10 | ~$90 |
| ChatGPT | ~$0.08 | ~$72 |
| Claude | ~$0.15 | ~$135 |
| **Total** | ~$0.38 | ~$342 |

### Latency Budget

| Phase | Target | Timeout |
|-------|--------|---------|
| Thesis generation (parallel) | 15s | 30s |
| Claude synthesis | 20s | 45s |
| Peer review (parallel) | 10s | 20s |
| Final decision | 5s | 15s |
| **Total** | 50s | 110s |

### Failure Modes

| Failure | Mitigation |
|---------|------------|
| One AI times out | Proceed with 2/3 proposals |
| All AIs timeout | Fall back to Phase 1 (Grok-only) |
| Claude synthesis fails | Log error, wait for next cycle |
| Conflicting reviews | Claude makes final call |

---

## Appendix A: Migration Path

### Gradual Rollout

1. **Week 1**: Add Perplexity alongside Grok (no changes to decision flow)
2. **Week 2**: Add ChatGPT, begin logging all three proposals
3. **Week 3**: Enable Claude synthesis (still comparing to old system)
4. **Week 4**: Enable peer review
5. **Week 5**: Full MACA with repositioning

### Rollback Plan

If MACA underperforms, we can instantly revert by:

```python
# In config.py
MACA_ENABLED = False  # Falls back to Phase 1 Grok-only mode
```

All MACA data is logged separately, so rollback is clean.

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **MACA** | Multi-Agent Consensus Architecture |
| **CIO** | Chief Investment Officer (Claude's role) |
| **Thesis Proposal** | Structured investment recommendation from an AI |
| **Synthesis** | Claude's consolidated recommendation |
| **Peer Review** | Validation step where AIs critique Claude's synthesis |
| **Conviction** | 0-100 score indicating trade confidence |
| **ROTATE** | Sell existing position to buy new opportunity |
| **Time-Sensitive** | Opportunity with a deadline (earnings, FDA, etc.) |
