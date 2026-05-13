# ArtPilot Daily — AI Art Print Generator

Automated pipeline that generates one high-resolution, print-ready AI painting every day across 5 best-selling Etsy art styles, organises all files, and produces ready-to-paste Etsy listing metadata.

**Daily cost:** ~$0.06 per painting | **Break-even:** 1 Etsy sale covers 166 days

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

Required keys:
- `ANTHROPIC_API_KEY` — [console.anthropic.com](https://console.anthropic.com)
- `FAL_API_KEY` — [fal.ai](https://fal.ai)
- `GITHUB_TOKEN` — GitHub PAT with `repo` scope
- `GITHUB_REPO` — `username/repo-name`

Optional fallback providers:
- `OPENAI_API_KEY` — DALL-E 3 fallback (~$0.12/image)
- `IDEOGRAM_API_KEY` — Ideogram v2 second fallback

### 3. Test without spending credits

```bash
python main.py --dry-run
```

Runs all 6 stages with a mock image. No API calls made. Completes in ~5 seconds.

### 4. Run for real

```bash
python main.py
```

---

## CLI Options

```
python main.py                             # Next style in rotation
python main.py --dry-run                   # Mock run, no API calls, no GitHub push
python main.py --style textured_abstract   # Override style
python main.py --variation "warm champagne and ivory"  # Override variation
python main.py --schedule                  # Run on schedule from config.json
```

Available style IDs: `textured_abstract`, `impressionist_landscape`, `nature_botanical`, `floral_pop`, `contemporary_portrait`

---

## Output Structure

```
output/
└── YYYY-MM-DD_style-id/
    ├── source_original.jpg
    ├── print_5x7_1500x2100px.jpg
    ├── print_8x10_2400x3000px.jpg
    ├── print_11x14_3300x4200px.jpg
    ├── print_16x20_4800x6000px.jpg
    ├── print_18x24_5400x7200px.jpg
    ├── print_24x36_7200x10800px.jpg
    ├── print_A4_2480x3508px.jpg
    ├── print_A3_3508x4961px.jpg
    ├── etsy_listing.json      ← Copy-paste ready Etsy listing
    ├── social_media.json      ← Instagram + Pinterest captions
    └── generation_log.json    ← Technical run details
```

---

## Scheduling

### Cron (Linux/macOS)

```bash
# Run at 7:00 AM daily
0 7 * * * cd /path/to/luminaryPrints && python main.py >> logs/cron.log 2>&1
```

### GitHub Actions (fully cloud, no local machine)

See `.github/workflows/daily_generation.yml`. Add your API keys as repository secrets.

---

## Style Rotation

The pipeline cycles through 5 styles × 7 colour variations = **35 unique paintings** before any repetition. After 35 days the cycle restarts with freshly generated variations.

| Style | Variations |
|---|---|
| Textured Abstract | 7 colour palettes |
| Impressionist Landscape | 7 scenes |
| Nature and Botanical | 7 botanical subjects |
| Floral Pop | 7 floral compositions |
| Contemporary Portrait / Figurative | 7 figure studies |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Configuration

Edit `config.json` to change schedule time, output sizes, JPEG quality, pricing, and more. No code changes needed.

Edit `styles/styles.json` to add new styles or variations without touching any Python.
