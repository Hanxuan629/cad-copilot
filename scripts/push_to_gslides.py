"""Push a deck into a Google Slides presentation, one full-bleed image per slide.

Google Slides API can only insert images from an accessible URL (not local
files / base64), so the flow is:
  1. render each deck slide to a PNG (headless Chrome, from the #N hash view)
  2. upload the PNGs to a Drive folder, make them link-readable
  3. create one slide per image in the target presentation, image fills the page

Credentials (never committed):
  ~/.config/gcloud-oauth/client_secret.json   (OAuth desktop client you downloaded)
  ~/.config/gcloud-oauth/token.json           (auto-written after first consent)

First run opens a browser for consent; subsequent runs reuse the token.

Usage:
  # append image-slides to the END of an existing presentation (SAFE default):
  .venv/bin/python scripts/push_to_gslides.py <PRESENTATION_ID> \
      --deck results/<name>.html --n <N_SLIDES>

  # reuse already-rendered PNGs instead of re-rendering from HTML:
  .venv/bin/python scripts/push_to_gslides.py <PRESENTATION_ID> --png-dir results/slides_png/

  --clear   DESTRUCTIVE: delete every existing slide first (fresh rebuild).
            Only pass this when the user explicitly asks to replace the whole
            presentation. Default is append-to-end. Deleted slides can be
            recovered via Google Slides File -> Version history.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import time
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

ROOT = Path(__file__).resolve().parents[1]
CRED_DIR = Path.home() / ".config" / "gcloud-oauth"
CLIENT_SECRET = CRED_DIR / "client_secret.json"
TOKEN = CRED_DIR / "token.json"
SCOPES = [
    "https://www.googleapis.com/auth/presentations",
    "https://www.googleapis.com/auth/drive.file",
]

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
# US-letter 16:9 slide in EMU (Google's default page size): 10 x 5.625 in.
PAGE_W_EMU = 9144000
PAGE_H_EMU = 5143500


# --------------------------------------------------------------------------- #
# auth
# --------------------------------------------------------------------------- #
def get_creds() -> Credentials:
    if not CLIENT_SECRET.exists():
        raise SystemExit(
            f"Missing {CLIENT_SECRET}\n"
            "Put your downloaded OAuth desktop client JSON there first."
        )
    creds = None
    if TOKEN.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN.write_text(creds.to_json())
        os.chmod(TOKEN, 0o600)
    return creds


# --------------------------------------------------------------------------- #
# render slides to PNG
# --------------------------------------------------------------------------- #
def render_pngs(deck: Path, n: int, outdir: Path, width: int = 1600) -> list[Path]:
    height = round(width * 9 / 16)
    url = deck.resolve().as_uri()
    paths = []
    print(f"rendering {n} slides at {width}x{height} …")
    for i in range(1, n + 1):
        p = outdir / f"slide_{i:02d}.png"
        subprocess.run(
            [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
             f"--window-size={width},{height}",
             f"--screenshot={p}", f"{url}#{i}"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if not p.exists():
            raise SystemExit(f"failed to render slide {i}")
        paths.append(p)
    print(f"  rendered {len(paths)} PNGs")
    return paths


# --------------------------------------------------------------------------- #
# upload to Drive, make link-readable
# --------------------------------------------------------------------------- #
def upload_pngs(drive, pngs: list[Path], folder_name: str) -> list[str]:
    folder = drive.files().create(
        body={"name": folder_name, "mimeType": "application/vnd.google-apps.folder"},
        fields="id",
    ).execute()
    fid = folder["id"]
    print(f"drive folder {fid} ({folder_name})")
    urls = []
    for p in pngs:
        f = drive.files().create(
            body={"name": p.name, "parents": [fid]},
            media_body=MediaFileUpload(str(p), mimetype="image/png"),
            fields="id",
        ).execute()
        drive.permissions().create(
            fileId=f["id"], body={"type": "anyone", "role": "reader"},
        ).execute()
        urls.append(f"https://drive.google.com/uc?export=view&id={f['id']}")
        print(f"  uploaded {p.name}")
    return urls


# --------------------------------------------------------------------------- #
# build slides
# --------------------------------------------------------------------------- #
def clear_slides(slides, pres_id):
    pres = slides.presentations().get(presentationId=pres_id).execute()
    reqs = [{"deleteObject": {"objectId": s["objectId"]}}
            for s in pres.get("slides", [])]
    if reqs:
        slides.presentations().batchUpdate(
            presentationId=pres_id, body={"requests": reqs}).execute()
        print(f"cleared {len(reqs)} existing slides")


def add_image_slides(slides, pres_id, urls: list[str]):
    # unique suffix so re-runs never collide with previously-pushed deckslide_* IDs:
    # base it on how many pages already exist in the presentation
    existing = slides.presentations().get(presentationId=pres_id).execute()
    base = len(existing.get("slides", []))
    reqs = []
    for i, url in enumerate(urls):
        sid = f"deckslide_{base + i:03d}"
        iid = f"deckimg_{base + i:03d}"
        reqs.append({"createSlide": {"objectId": sid,
                                     "slideLayoutReference": {"predefinedLayout": "BLANK"}}})
        reqs.append({"createImage": {
            "objectId": iid, "url": url,
            "elementProperties": {
                "pageObjectId": sid,
                "size": {"width": {"magnitude": PAGE_W_EMU, "unit": "EMU"},
                         "height": {"magnitude": PAGE_H_EMU, "unit": "EMU"}},
                "transform": {"scaleX": 1, "scaleY": 1, "translateX": 0,
                              "translateY": 0, "unit": "EMU"}},
        }})
    # batch in chunks to stay well under request limits
    CH = 40
    for k in range(0, len(reqs), CH):
        slides.presentations().batchUpdate(
            presentationId=pres_id, body={"requests": reqs[k:k + CH]}).execute()
    print(f"created {len(urls)} image slides")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("presentation_id")
    ap.add_argument("--deck", default="results/sync_deck.html")
    ap.add_argument("--n", type=int, default=None,
                    help="number of slides to render; required unless --png-dir")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--png-dir", default=None,
                    help="reuse existing slide_NN.png here instead of re-rendering")
    args = ap.parse_args()

    creds = get_creds()
    slides = build("slides", "v1", credentials=creds)
    drive = build("drive", "v3", credentials=creds)

    if args.png_dir:
        d = (ROOT / args.png_dir).resolve()
        pngs = sorted(d.glob("slide_*.png"))
        if not pngs:
            raise SystemExit(f"no slide_*.png in {d}")
        print(f"reusing {len(pngs)} PNGs from {d}")
        urls = upload_pngs(drive, pngs, f"{d.name}")
    else:
        deck = (ROOT / args.deck).resolve()
        if not deck.exists():
            raise SystemExit(f"deck not found: {deck}")
        if not args.n:
            raise SystemExit("--n <N_SLIDES> is required when rendering from a deck")
        with tempfile.TemporaryDirectory() as td:
            pngs = render_pngs(deck, args.n, Path(td), args.width)
            urls = upload_pngs(drive, pngs, f"{deck.stem}_slides")

    if args.clear:
        clear_slides(slides, args.presentation_id)
    add_image_slides(slides, args.presentation_id, urls)
    # give Drive a moment so images resolve on first open
    time.sleep(2)
    print(f"\ndone → https://docs.google.com/presentation/d/{args.presentation_id}/edit")


if __name__ == "__main__":
    main()
