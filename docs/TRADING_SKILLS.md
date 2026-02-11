# Trading Skills Reference

Reference for the committee when recommending **long**, **short**, and **options**. Use this to form and defend theses.

---

## 1. Trading Long (Buy Equity)

**What it is:** Buy shares of a stock or ETF. Profit when price rises; loss when it falls. Position is "long" the underlying.

**When to recommend LONG:**
- Catalysts support upside (earnings beat, product launch, sector tailwind).
- Technical structure supports (support hold, breakout, trend alignment).
- Narrative and sentiment align with upside (e.g. Fed pivot, sector rotation into the name).
- Valuation or fundamentals are favorable vs. peers or history.

**Skills:**
- Define **entry**: price or zone (e.g. "on pullback to support", "on breakout above X").
- Define **invalidation**: level or condition that kills the thesis (e.g. "below support Y", "if earnings miss").
- **Position sizing**: size relative to portfolio and conviction; avoid overconcentration.
- **Time horizon**: intraday, swing (days–weeks), or multiweek; state it so risk is clear.

**Execution:** System submits BUY (equity) via Alpaca. Can buy and sell the same name daily if needed.

---

## 2. Trading Short (Sell Short)

**What it is:** Borrow shares, sell them, and later buy them back to close. Profit when price falls; loss when it rises. Position is "short" the underlying.

**When to recommend SHORT:**
- Catalysts support downside (earnings miss, regulatory risk, sector headwind).
- Technical structure supports (resistance hold, breakdown, downtrend).
- Narrative and sentiment align with downside (e.g. overvaluation, crowded long).
- Event or fundamental overhang (e.g. lockup expiry, debt maturity).

**Skills:**
- **Borrow and margin:** Shorting requires margin; hard-to-borrow names can be expensive or unavailable. Prefer liquid, easy-to-borrow names when recommending short.
- **Squeeze risk:** Shorts can be squeezed on sharp rallies; state invalidation (e.g. "cover if price closes above X").
- **Asymmetric risk:** Max gain is 100% (stock to zero); max loss is unbounded. Prefer high-conviction, well-defined shorts.
- Define **entry** and **invalidation** clearly (e.g. "short on break of support", "cover if earnings beat and gap up").

**Execution:** System submits SELL to open short (or close long). Margin account required; execution is equity order.

---

## 3. Buying Options

**What it is:** Buy a **call** (right to buy at strike by expiry) or **put** (right to sell at strike by expiry). Pay **premium**; no obligation to exercise. Max loss = premium paid.

**When to recommend OPTIONS:**
- **Calls:** Bullish on name or index; want leverage or defined risk; event (earnings, FDA) with clear date.
- **Puts:** Bearish on name or index; want leverage or defined risk; hedging a long; or event with clear date.
- Prefer when **time horizon is defined** (earnings, expiry) and **volatility** is a consideration (IV high = expensive; IV low = cheaper premium).

**Key terms:**
- **Strike:** Price at which you can buy (call) or sell (put) the underlying.
- **Expiry:** Date the option expires. Further out = more time value, usually more premium.
- **Premium:** Price paid for the option. Total cost = premium × contract multiplier (usually 100).
- **In the money (ITM) / At the money (ATM) / Out of the money (OTM):** Call is ITM when underlying > strike; put is ITM when underlying < strike. ATM = strike near current price; OTM = cheaper, more leverage, less probability.

**Skills:**
- **Theta (time decay):** Options lose value as expiry approaches. Favor enough time to thesis (e.g. 2–4+ weeks for swing, or match to event).
- **Implied volatility (IV):** High IV = expensive premium; low IV = cheaper. Avoid buying expensive options right after a vol spike unless you expect another move.
- **Define thesis and invalidation:** e.g. "Buy call if we expect move above X by expiry; exit or let expire if price stays below Y."
- **Position sizing:** Options are leveraged; size small (e.g. premium as % of portfolio) so max loss (premium) is acceptable.

**Execution:** System can recommend options; execution may require options-enabled account and options symbol format (e.g. OCC). When in doubt, recommend equity long/short and note "options alternative" in thesis.

---

## Summary Table

| Type   | Action | When to use           | Main risk              |
|--------|--------|------------------------|-------------------------|
| Long   | BUY    | Upside thesis          | Price decline            |
| Short  | SELL   | Downside thesis        | Squeeze, unlimited loss   |
| Option | Call   | Bullish, defined date  | Theta, IV, premium loss  |
| Option | Put    | Bearish, hedge, event   | Theta, IV, premium loss  |

Use this reference when forming recommendations and when debating so the committee consistently applies long, short, and options skills.
