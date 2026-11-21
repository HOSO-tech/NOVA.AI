# NOVAAA AI V2

A larger deployable starter with a responsive silver/maroon web UI.

## Working in this build
- Backend AI chat gateway with Claude, Gemini and OpenAI provider selection (requires valid provider key and hosting).
- Story/script drafting through the connected AI provider.
- Creator prompt builder for video/image/voice prompts.
- Browser-local content planner.
- Browser-local project memory for story/character notes.
- Provider connection status.

## Not connected / not implemented
Actual media generation, social publishing, Telegram bot, OAuth login, owner/admin permissions, cloud database/storage, cross-device memory, and Android APK are not implemented. Their panels clearly indicate this. These require separate provider accounts, API approvals/credentials, and additional backend services.

## Run
Requires Node.js 18+:
1. `npm install`
2. Copy `.env.example` to `.env`
3. Add provider API keys to `.env` (never commit or share it)
4. `npm start`
5. Open `http://localhost:3000`

For deployment, use a Node.js hosting provider and set `npm start` as the start command. Add API keys through the host's secret/environment-variable settings, not through the website.


## YouTube and Instagram connection setup (OAuth)
This release adds official OAuth connection flows and account-status display. It does not yet publish content or manage DMs/comments.

Add these server-side environment variables:
- `APP_BASE_URL` — exact HTTPS deployed origin, e.g. your hosting URL (no trailing slash)
- `SESSION_SECRET` — long random secret
- `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`
- `INSTAGRAM_APP_ID`, `INSTAGRAM_APP_SECRET`

Register these exact redirect URIs in the provider developer consoles:
- `${APP_BASE_URL}/auth/youtube/callback`
- `${APP_BASE_URL}/auth/instagram/callback`

YouTube: enable YouTube Data API v3 and create a Web application OAuth client. The current flow requests read-only channel access. See Google's official authorization guide: https://developers.google.com/youtube/data/guides/authentication

Instagram: configure the Instagram API with Instagram Login in a Meta developer app and use an eligible account. This flow requests `instagram_business_basic` and only reads the connected profile. See Meta's official Instagram API documentation: https://developers.facebook.com/docs/instagram-platform/

Important production note: Express's default in-memory session store is for development only. Before public deployment, replace it with a persistent, secure session store and use HTTPS. Do not put OAuth secrets in frontend files, commit them, or send them in chat. Ask a parent/guardian to supervise account and developer-console setup.
