# GST Architecture Redesign Proposal

**Goal:** Fewer moving parts, one clear data flow, and nothing in the critical path that can "always break." Analysts and Chair should never depend on which scanners succeeded today.

---

## 1. What’s Wrong Today

| Problem | Why it hurts |
|--------|----------------|
| **Grok called 3 times per cycle** | Agent calls `scan_sentiment()` + `scan_market_overview()`; MACA then calls `scan_market_overview()` again for the "Grok thesis." When that third call returns [], we had to add a fallback and still get syntax/edge-case bugs. |
| **Signals split by magic keys** | Agent filters with `s.get("source") == "fred"` / `"polymarket"`. Event signals live in a separate list. Any scanner that uses `source_type` or a different value breaks MACA or the Telegram inventory. |
| **Four separate lists into MACA** | `fred_signals`, `polymarket_signals`, `event_signals`, plus `technical_analysis` and `market_context`. Many code paths assume one of these exists; when FRED/Polymarket/Events all return 0, we patch fallbacks everywhere. |
| **Different signal types** | FREDSignal, PolymarketSignal, GrokSignal, event dicts, technical dict. Normalization happens late and is inconsistent; downstream code still branches on source. |
| **Scanners can throw** | Each scanner is in its own try/except; one failure only zeros that source. But then MACA and Telegram have to handle "only technical" or "only Grok" and we get special cases (e.g. Grok fallback from technical context). |

Net effect: **data flow is fragmented and brittle.** A small change (e.g. a new scanner, a key rename, or an API returning empty) forces fixes in agent, MACA, and sometimes Telegram.

---

## 2. Design Principles for the Redesign

1. **One pipeline, one shape**  
   Everything downstream sees a single, normalized structure. No splitting by `source` in the agent or MACA.

2. **Scanners can’t break the cycle**  
   Each scanner runs in a guarded runner. Timeout + catch all exceptions → return `[]`. The cycle always gets a "scan result" object; it may have zero signals from some sources.

3. **Analysts don’t call APIs for "signals"**  
   Analysts only produce theses. They get one **signal digest** (text + optional structured summary). No analyst re-calls Grok or any scanner. That removes duplicate calls and adapter/fallback complexity.

4. **Explicit phases**  
   Collect → Normalize → Council → Decide → Notify. Each phase has one input type and one output type. Easy to log, test, and debug.

5. **Graceful degradation**  
   Council and Chair are written so that "zero FRED, zero Polymarket, zero events" is a normal input. No special "only technical" or "Grok empty" branches; one code path that handles N signals for any N ≥ 0.

---

## 3. Proposed Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: COLLECT (agent or a dedicated ScanRunner)             │
│  • Run each scanner in a guard: timeout + try/except → []       │
│  • Output: list of raw results per source (no throwing)         │
└─────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: NORMALIZE                                              │
│  • Single schema: { source, summary, confidence, timestamp, ... } │
│  • One list: all_signals (FRED, Polymarket, Event, Grok, etc.)  │
│  • One optional: technical_summary (single dict or null)        │
│  • Output: CycleInput = { signals, technical_summary, portfolio } │
└─────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3: COUNCIL (MACA)                                         │
│  • Input: CycleInput only (no separate fred/poly/event/tech)    │
│  • Build one signal_digest string from CycleInput.signals        │
│  • Each analyst: generate_thesis(portfolio, signal_digest,      │
│                 technical_summary) → proposal                    │
│  • Grok is an analyst: same interface, no scan_market_overview  │
│    in the middle (use signals already in CycleInput from agent)  │
└─────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 4: DECIDE                                                  │
│  • Chair: synthesize(proposals, CycleInput) → decision          │
│  • Risk engine → approve/reject                                  │
└─────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 5: NOTIFY                                                  │
│  • Telegram formats from CycleInput + proposals + decision       │
│  • Single place that maps CycleInput.signals to "FRED (N)", etc. │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Key Contract: CycleInput

One object for the whole council and notify path:

```python
@dataclass
class CycleInput:
    cycle_id: str
    signals: List[NormalizedSignal]   # All sources, one schema
    technical_summary: Optional[Dict[str, Any]]  # One dict or None
    portfolio: Dict[str, Any]
    timestamp_utc: str
```

```python
@dataclass
class NormalizedSignal:
    source: str       # "fred" | "polymarket" | "event" | "grok" | "technical"
    summary: str
    confidence: float
    timestamp_utc: str
    # optional: tickers, raw_value, etc.
```

- **Normalize** converts every scanner output into `NormalizedSignal` (and one technical blob).
- **Agent** (or ScanRunner) produces `CycleInput` once. No more `fred_signals_dict`, `polymarket_signals_dict`, `event_signals_dict`, or "first technical signal" passed separately.
- **MACA** and **Telegram** only accept `CycleInput` (plus proposals/decision). They derive counts and labels from `CycleInput.signals` by grouping on `source`.

---

## 5. Grok: One Role, No Duplicate Calls

- **In Collect:** Agent runs Grok once (or twice if we keep sentiment + overview as two steps). Those results are turned into `NormalizedSignal`s with `source="grok"` and go into `CycleInput.signals`.
- **In Council:** The "Grok analyst" is a thesis generator only. It receives the same `signal_digest` (and optional technical_summary) as Perplexity and ChatGPT. It does **not** call `scan_market_overview()` again. Its thesis is "given these signals (which already include Grok’s own scan), what’s the narrative thesis?"
- That removes the duplicate Grok call, the Grok-only fallback, and the adapter that re-calls the scanner. If there are zero Grok signals in CycleInput (because the scan failed or returned []), the Grok analyst still runs and produces a HOLD thesis from the other signals.

---

## 6. Scanners: Never Crash the Cycle

- Wrap every scanner in a runner, e.g. `async def run_scanner(name, coro) -> List[Any]: ... try: return await asyncio.wait_for(coro(), timeout=30) except: return []`.
- Agent (or ScanRunner) builds the raw "per-source" lists from these results, then runs **Normalize** to produce `CycleInput`. So even if FRED, Polymarket, and Events all return [], we get a valid `CycleInput` with `signals=[]` (plus whatever Technical and Grok returned). No special-case "no signals" early exit unless we explicitly want to skip the council when total signals are below a threshold.

---

## 7. Implementation Order (Suggested)

1. **Define `NormalizedSignal` and `CycleInput`** in a small `models` or `core` module and use them in one place (e.g. a new `normalize.py` that takes raw scanner results and returns `CycleInput`).
2. **Introduce a ScanRunner** (or refactor agent) so that every scanner runs through the guarded runner and returns a list; agent builds the single `CycleInput` from that.
3. **Change MACA to take only `CycleInput`** (and maybe `market_context` string if we keep it). Inside MACA, build `signal_digest` from `CycleInput.signals` once. Remove `fred_signals`, `polymarket_signals`, `event_signals`, `technical_analysis` as separate arguments.
4. **Make Grok an analyst only:** remove the second `scan_market_overview()` call from the Grok thesis path. Grok analyst receives `signal_digest` (which already includes Grok signals from the collect phase) and returns a proposal like the others.
5. **Update Telegram** to build the "Signals Collected" section from `CycleInput.signals` (group by `source`, count, show summaries). Remove reliance on separate lists or `signal_inventory` keyed by FRED/Polymarket/Events/Technical.
6. **Deprecate** the old agent path that splits by `s.get("source") == "fred"` and the Grok fallback from technical context once the new path is default.

---

## 8. What Stays the Same

- Philosophy: committee, debate, HOLD as success, human approval, explainability.
- Scanners themselves (FRED, Polymarket, Event, Technical, Grok) can stay; only the way we run them and pass results changes.
- Chair and risk engine and Telegram UX (3-part message, buttons) stay; they just get data in one shape.
- Alpaca, approval gate, and paper trading unchanged.

---

## 9. Summary

| Current | Proposed |
|--------|----------|
| 4+ separate signal lists + technical + market_context | One `CycleInput`: signals (normalized) + technical_summary + portfolio |
| Grok called 3×; adapter re-calls scanner; fallback when empty | Grok called 1× in Collect; Grok analyst only generates thesis from shared digest |
| Filter by `source` in agent; event_signals separate | Single list of `NormalizedSignal`; group by `source` only for display/counts |
| Scanners can throw; many branches for "0 signals" | Guarded runner; 0 signals is normal input; one code path |
| MACA and Telegram know about FRED/Poly/Event/Technical by name | They only know `CycleInput` and `NormalizedSignal.source` |

This keeps the GST philosophy and product behavior while making the data flow simple, predictable, and resilient so that "something is always breaking" becomes rare.

---

*Next step: decide whether to adopt this direction; then we can implement in small steps (e.g. add CycleInput + normalize first, then switch MACA, then Grok, then Telegram).*
