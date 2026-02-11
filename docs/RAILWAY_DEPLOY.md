# Railway – Why Deployments Might Not Be Happening

If pushes to GitHub aren’t showing up as new deployments on Railway, or **Railway no longer shows your repo** (integration used to work, now it’s broken), use the steps below to restore it.

---

## Fix broken integration (repo not showing in Railway)

When the Railway GitHub App is missing from GitHub or the repo no longer appears in Railway’s “Connect Repo” list, do a **full clean reconnect**. This is what Railway support recommends.

### Step A – Clean Railway’s side

1. Go to **https://railway.app/account** (or Railway dashboard → your profile → **Account Settings**).
2. Find **GitHub** / **Connected accounts** / **Integrations**.
3. **Disconnect** or **Remove** the GitHub connection so Railway has no stale link.

### Step B – Clean GitHub’s side

1. **Revoke OAuth:** GitHub → **Settings** → **Applications** → **Authorized OAuth Apps** → find **Railway** → **Revoke**.
2. **Re-install the GitHub App (so Railway appears again):**
   - Go to **https://github.com/apps/railway-app** (Railway’s GitHub App page).
   - Click **Configure** or **Install**.
   - **Where to install:** If the repo is under the **Le-Incal** organization, choose **Le-Incal** (the org), not your personal account. Otherwise Railway won’t see org repos.
   - **Repository access:** Either **All repositories** or **Only select repositories** and add **Gann-Sentinel-Trader**.
   - Complete the install.

**If you see “This action must be performed by an organization owner”:** The **organization owner** of Le-Incal must do the install (or approve the app for the org). Only org owners can grant Railway access to org repos.

### Step C – Reconnect from Railway

1. In **Railway** → your project (or New Project) → **Settings** → **Source** / **Connect Repo**.
2. Click **Connect GitHub** (or **Configure GitHub App**). Log in with GitHub if asked.
3. You should now see **Le-Incal/Gann-Sentinel-Trader** in the repo list. Select it and the branch (e.g. `main`).

After this, the integration is restored: repo shows in Railway, and pushes can trigger deploys again.

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

---

## Fallback: Deploy without GitHub (CLI-only)

Only if the steps above don’t fix the integration (e.g. org owner unavailable): you can deploy **from your machine** with the Railway CLI so the app still runs while the GitHub link is sorted out.

### 1. Create a project in Railway (no repo)

- Go to [railway.app](https://railway.app) and log in.
- **New Project** → choose **Empty Project** (or “Deploy from CLI” if you see it).
- Open the project. You’ll add the code in the next steps.

### 2. Install and log in with the CLI

```bash
# Install (macOS with Homebrew)
brew install railway

# Or with npm
npm i -g @railway/cli
```

Then log in (opens browser):

```bash
railway login
```

### 3. Link this repo to your Railway project

From the **Gann-Sentinel-Trader** folder:

```bash
cd /Users/kylemertensmeyer/Gann-Sentinel-Trader
railway link
```

When prompted, select the **workspace**, **project**, **environment**, and **service** (or create a new service if the project is empty).

### 4. Deploy

```bash
railway up
```

This uploads your code, builds it on Railway (using the Procfile/railway.toml), and deploys. Use `railway logs` to watch the app.

### 5. (Optional) Set env vars

In the Railway dashboard: Project → your service → **Variables**, add the same env vars you use locally (e.g. `XAI_API_KEY`, `TELEGRAM_BOT_TOKEN`, etc.).

---

**Summary:** With `railway link` + `railway up` you can deploy GST to Railway without ever connecting GitHub. For future updates, run `railway up` again from this folder.
