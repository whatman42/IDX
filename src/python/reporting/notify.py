"""
Telegram notifier – unidirectional push.
All charts are generated in-memory (io.BytesIO) and sent via multipart.
Never write image files to disk.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import httpx
import matplotlib.pyplot as plt
import typer
from dotenv import load_dotenv

app = typer.Typer()


def _create_equity_chart() -> bytes:
    """Generate a simple equity curve in memory."""
    fig, ax = plt.subplots(figsize=(8, 4))
    # Stub data
    x = range(30)
    y = [100 + i * 0.3 + (i % 5) * 0.1 for i in x]
    ax.plot(x, y, color="#00c853", linewidth=2)
    ax.set_title("Portfolio Equity (Paper)")
    ax.set_ylabel("Index")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def send_telegram_media_group(
    bot_token: str,
    chat_id: str,
    caption: str,
    images: list[bytes],
) -> None:
    """Send multiple images as a media group (avoids rate limits)."""
    url = f"https://api.telegram.org/bot{bot_token}/sendMediaGroup"

    media = []
    files = {}
    for i, img in enumerate(images):
        name = f"photo{i}"
        media.append(
            {
                "type": "photo",
                "media": f"attach://{name}",
                "caption": caption if i == 0 else "",
            }
        )
        files[name] = (f"{name}.png", img, "image/png")

    with httpx.Client(timeout=30) as client:
        resp = client.post(
            url,
            data={"chat_id": chat_id, "media": str(media).replace("'", '"')},
            files=files,
        )
        resp.raise_for_status()
        print("[Telegram] Media group sent")


@app.command()
def main(status: str = "success"):
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("[Notify] Telegram credentials missing – skip")
        return

    caption = f"IDX Quant Cycle finished with status: *{status}*"
    img = _create_equity_chart()

    try:
        send_telegram_media_group(token, chat_id, caption, [img])
    except Exception as e:
        # Fallback to simple text
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        httpx.post(url, json={"chat_id": chat_id, "text": caption + f"\n(Error: {e})"})


if __name__ == "__main__":
    app()
