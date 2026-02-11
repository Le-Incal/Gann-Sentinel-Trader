# Telegram message feed (for Cursor / Claude)

Every Telegram message the bot sends or receives is appended to a file in the **data/** folder (same location as the database) so it persists on Railway and Cursor/Claude can reference it.

## Location

- **Path:** `data/telegram_feed.md`
- **Why data/:** Same directory as `sentinel.db`; on Railway the data directory is typically the persistent volume, so the feed survives restarts and is writable.
- **Format:** Each message is a block: `---` then a header line `**timestamp | direction | message_type**`, then the message content.

## How to use it

- **Local runs:** When you run GST locally, open `data/telegram_feed.md` in Cursor. Ask Claude to "read the latest Telegram output" or "check data/telegram_feed.md."
- **Deployed (Railway):** The file is written to the persistent data volume. To have Claude reference it: use **GET /api/telegram-feed?token=YOUR_LOGS_API_TOKEN** to fetch the full feed, or **GET /api/logs?token=...&limit=50** for DB-backed message logs. Paste the response into Cursor when needed.

## Note

The `data/` directory is in `.gitignore`, so `telegram_feed.md` is not committed. Message content stays out of the repo. If the feed still doesn't appear, check Railway logs for: `Telegram feed write failed` (warnings are logged on write failure).
