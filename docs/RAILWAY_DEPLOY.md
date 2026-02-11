# Railway – Why Deployments Might Not Be Happening

If pushes to GitHub aren’t showing up as new deployments on Railway, work through this list.

---

## 1. Confirm the repo Railway is using

- **Railway dashboard** → your project → **Settings** (or the service’s **Settings**).
- Under **Source** / **Connect Repo** / **GitHub**, check that the connected repo is:
  - **Le-Incal/Gann-Sentinel-Trader** (or your fork).
  - The correct **branch** (usually `main`).

If it says “Not connected” or shows a different repo, connect or reconnect the correct GitHub repo.

---

## 2. Turn on auto-deploy

- In the same **Settings** (project or service), find **Deploy** / **Build**.
- Ensure **Auto-deploy** (or “Deploy on push”) is **enabled** for the branch you push to (e.g. `main`).

If this is off, Railway will not deploy when you push.

---

## 3. Reconnect the GitHub repo (fix permissions)

Sometimes the GitHub ↔ Railway link breaks (e.g. after a password change or org changes):

1. **Railway** → Project → **Settings** → **Source** (or the service that runs this app).
2. **Disconnect** the current GitHub repo.
3. **Connect** again and choose **Le-Incal/Gann-Sentinel-Trader** (and the right branch).
4. Authorize Railway in GitHub if prompted (repo access, install GitHub app, etc.).

After reconnecting, push a small commit and see if a new deployment appears.

---

## 4. Trigger a deploy manually

To confirm the app *can* deploy (even if auto-deploy isn’t firing):

- **Railway** → your service → **Deployments**.
- Use **“Deploy”** / **“Redeploy”** / **“Deploy latest commit”** (wording varies).

If a new deployment is created and builds from the latest commit, the build/deploy path is fine and the issue is likely connection or auto-deploy settings (steps 1–3).

---

## 5. Confirm GitHub is receiving your pushes

- Open: **https://github.com/Le-Incal/Gann-Sentinel-Trader/commits/main**
- Check that your latest commits (e.g. “fix: lazy-import alpaca…”) are there.

If they’re not, push from your machine:

```bash
git status
git push origin main
```

---

## 6. Check Railway’s GitHub app and repo access

- **GitHub** → **Settings** → **Applications** → **Installed GitHub Apps** → **Railway**.
- Ensure Railway has access to the **Le-Incal/Gann-Sentinel-Trader** repo (or the org that owns it).

If the repo was added after linking Railway, you may need to grant access to that repo in the Railway GitHub app settings.

---

## Summary checklist

| Check | Where |
|-------|--------|
| Repo connected = **Le-Incal/Gann-Sentinel-Trader**, branch e.g. **main** | Railway → Project/Service → Settings → Source |
| **Auto-deploy** (Deploy on push) is **ON** | Railway → Settings → Deploy / Build |
| GitHub has latest commits | github.com/Le-Incal/Gann-Sentinel-Trader/commits/main |
| Manual **Redeploy** works | Railway → Deployments → Deploy / Redeploy |
| Railway app has repo access | GitHub → Settings → Applications → Railway |

After fixing the connection or auto-deploy, push a commit and you should see a new deployment on Railway. The repo also includes a **Procfile** (`worker: python main.py`) and **railway.toml** so Railway knows how to build and run the app.
