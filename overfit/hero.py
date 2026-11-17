"""Rotating render of Figure 3 for the README: GIF (1200 x 675) and MP4."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from . import theme
from .plots import ROOT, fig3, load, plt


def render(frames: int = 72, out: Path | None = None) -> None:
    theme.use("dark")
    out = out or ROOT / "docs"
    out.mkdir(exist_ok=True)
    d = load()
    imgs = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(frames):
            azim = -55 + 35 * np.sin(2 * np.pi * i / frames)
            fig = fig3(d, None, elev=27, azim=azim, name=None, size=(12, 6.75))
            fig.axes[0].set_box_aspect((1, 1, 0.75), zoom=1.18)
            fig.suptitle("Joint density of in-sample and out-of-sample rank "
                         f"({d['summary']['n_trials']:,} trials, {d['summary']['cscv']['n_splits']:,} CSCV splits)", x=0.04, ha="left", fontsize=13)
            p = Path(tmp) / f"f{i:03d}.png"
            fig.savefig(p, dpi=100)
            plt.close(fig)
            imgs.append(Image.open(p).convert("RGB").resize((1200, 675)))
        pal = [im.quantize(colors=160, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for im in imgs]
        pal[0].save(out / "hero.gif", save_all=True, append_images=pal[1:], duration=60, loop=0, optimize=True)
        for i, im in enumerate(imgs):
            im.save(Path(tmp) / f"g{i:03d}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "16", "-i", str(Path(tmp) / "g%03d.png"), "-pix_fmt", "yuv420p",
                        "-vf", "scale=1200:674", str(out / "hero.mp4")], check=False)
