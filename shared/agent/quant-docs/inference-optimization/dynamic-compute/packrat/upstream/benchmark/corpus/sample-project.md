# Sample Project Memory

## Project Alpha
- **Stack:** TypeScript, React, Supabase, Vercel
- **Status:** LIVE at https://project-alpha.example.com
- **Deploy:** Vercel (auto-deploy from GitHub)
- TypeScript strict mode enabled
- React 18 with Suspense boundaries
- Supabase for auth + database
- Vercel edge functions for API

## Project Beta
- **Stack:** Python, FastAPI, Docker, Cloudflare
- **Status:** In development
- Python 3.12 with FastAPI
- Docker containerized
- Cloudflare Pages for frontend
- FastAPI handles all API routes

## Decisions
- Switched from JavaScript to TypeScript for type safety
- Moved from Firebase to Supabase (open source, cheaper)
- Deploy on Vercel instead of Netlify (better DX)
- React over Vue for ecosystem size
- Docker for reproducible builds

## Tools
- GitHub for source control
- Telegram bot for notifications
- Supabase dashboard for data management
- Vercel dashboard for deploys
- TypeScript compiler in strict mode
- React DevTools for debugging
- Docker Desktop for local development

## Notes
- TypeScript catches 90% of bugs at compile time
- Supabase row-level security is critical
- Vercel preview deploys save hours of QA
- React server components reduce bundle size
- Docker multi-stage builds cut image size 60%
- Cloudflare Workers are free for 100K requests/day
- GitHub Actions CI runs TypeScript type checks
- Telegram alerts on deploy success/failure
