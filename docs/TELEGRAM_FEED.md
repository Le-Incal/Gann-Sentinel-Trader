# Telegram message feed (for Cursor / Claude)

Every Telegram message the bot sends or receives is appended to a file in the project so Cursor or Claude Code can reference it.

## Location

- **Path:** `logs/telegram_feed.md`
- **Format:** Each message is a block: `---` then a header line `**timestamp | direction | message_type**`, then the message content.

## How to use it

- **Local runs:** When you run GST locally, open `logs/telegram_feed.md` in Cursor. You can ask Claude to "read the latest Telegram output" or "check logs/telegram_feed.md and fix the error in the last scan."
- **Deployed (Railway):** The file is written on the server; it is not in your Git repo. To have Claude reference it, either run the app locally sometimes so the file exists, or use the API: `GET /api/logs?token=YOUR_LOGS_API_TOKEN&limit=50` and paste the response when you need it.

## Note

The `logs/` directory is in `.gitignore`, so `telegram_feed.md` is not committed. That keeps message content out of the repo. The feed is still in the project directory whenever the app has been run.
