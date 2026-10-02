# HealthMax brand kit (v1)

<img src="logo/healthmax-mark.svg" width="72" alt="HealthMax mark">

**HealthMax · হেলথম্যাক্স · সঠিক সময়ে, সঠিক সেবা** (Right care, at the right time.)

The full guideline page is [`guidelines.html`](guidelines.html); open it in a browser. It supports light and dark mode.
Previews: [light](preview/guidelines-light.png) · [dark](preview/guidelines-dark.png).

## Why a new identity

The hackathon build used a crimson primary (`hsl(350 70% 55%)`) that is almost the same colour as its own emergency red (`hsl(0 84% 60%)`). In a triage app that's a real usability problem: the whole interface looks like an alarm, so a true emergency stops standing out. This kit:

1. moves the primary to a calm **petrol blue**;
2. **reserves red, amber and green for urgency only**;
3. keeps a softened **shapla pink** (Bangladesh's national flower, the water lily) as the one warm accent, carried over from the original build.

## Contents

| Path | What |
|---|---|
| `logo/healthmax-mark.svg` | Primary mark (petrol tile) |
| `logo/healthmax-mark-mono.svg` | One-colour mark |
| `logo/healthmax-glyph.svg` | Glyph without the tile, for small sizes and print |
| `logo/healthmax-mark-512.png` | Raster export |
| `tokens/tokens.css` · `tokens/tokens.json` | Colour, type, radius and tap-size tokens (light and dark) |
| `pattern/pulse-stitch.svg` | Tileable background pattern |
| `guidelines.html` | Brand guideline page |
| `banner.html` | Source of the README banner (`docs/images/banner.png`) |

## The mark

An **H whose crossbar is a heartbeat**. The two pillars are the person and the care they reach, and the pulse is the moment HealthMax connects them. The pulse inside the H also reads as an **M**, so the mark spells *HM*. A shapla-pink dot marks the peak of the pulse.

## Palette

All text pairings were checked against WCAG 2.1. The lowest is 4.54:1 (AA).

| Token | Light | Dark | Role |
|---|---|---|---|
| Petrol | `#0B4F6C` | `#6CC0DE` | Primary: trust, calm |
| Ink | `#10222B` | `#E8F0F2` | Text (15:1 on Rice) |
| Mist | `#E6F0F3` | `#15262E` | Surfaces |
| Rice | `#F7F5F0` | `#0D1A20` | Page ground: warm, not clinical white |
| Shapla | `#B8487A` | `#E58AB3` | Accent, for brand moments only |

**Urgency scale (state only, never decoration):**

| Level | Text | Background |
|---|---|---|
| Emergency | `#B42318` | `#FDECEA` |
| Urgent | `#7A4600` | `#FFF1D6` |
| Self-care | `#1F6F43` | `#E3F4EA` |

Colour never carries urgency on its own: each level always has a word and an icon too.

## Type

| Role | Font |
|---|---|
| Display (Bangla and Latin) | Anek Bangla 700–800 |
| Bangla text | Noto Sans Bengali 400/600, 17 px, line height 1.7 |
| Latin text | Inter 400/600 |

Bangla is set about 10% larger than Latin, because its glyphs carry more detail above and below the line. All fonts use the SIL Open Font License.

## Pattern

The pulse line is drawn as a **kantha running stitch**. Kantha is how worn cloth is mended in Bengali homes, so the pattern says "care that mends". Use it at low opacity behind heroes and empty states, never behind body text.

## UX principles (designing for a worried person)

1. Show the answer first and the explanation later. Urgency and the next step lead, and possible causes sit behind a tap.
2. One question per screen, with large yes/no buttons (48 px or more) and a microphone everywhere.
3. Say what to do, not just how bad it is.
4. Show where every answer came from.
5. Be honest about uncertainty.
6. Keep sensitive topics private by default.
7. Never say "diagnosis".

`docs/images/redesign-result-card.png` shows these rules applied to the triage result card, using sample data.

## Voice

Calm, plain, spoken Bangla, like a trusted elder sister who happens to be a nurse. Short sentences: no jargon, no fear, no false comfort.

> Status: the kit was designed after the project was archived (October 2026). The shipped app still uses the hackathon styling.
