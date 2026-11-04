"""LinkedIn carousel for Breast Cancer Awareness Month: 7 portrait slides (1080x1350 px) + one PDF.

Needs data/processed/bcsc_clean.parquet (clean.py), model_menohrt_test.parquet (prepare.py)
and xgb_menohrt.json (notebooks/02_modeling.ipynb).
Run from the project root:  python src/linkedin_figures.py

Wording rules for slide text: "mammographic density" / "tissue density" (never the colloquial term),
the unit is screening mammograms, BMI only as "higher"/"lower BMI", no em dashes, no ML jargon.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_rgb
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Ellipse, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "output" / "linkedin"

PINK = "#c2185b"
PINK_TINT = "#f6dbe6"
GRAY = "#cfcdc7"
INK = "#1d1d1b"
INK_2 = "#5c5b57"
SURFACE = "#ffffff"

WIDTH_IN, HEIGHT_IN, DPI = 7.2, 9.0, 150  # 1080 x 1350 px
MARGIN = 0.07
N_SLIDES = 7
FOOTER = ("Data: Breast Cancer Surveillance Consortium, 6.8M screening mammograms (2005 to 2017)",
          "Exploratory analysis, not medical advice")

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "text.color": INK, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.edgecolor": GRAY,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


# ---------- layout helpers ----------

def wrap(fig, text: str, size: float, weight: str = "normal", width_frac: float = 1 - 2 * MARGIN) -> str:
    """Greedy word wrap using real rendered text widths."""
    renderer = fig.canvas.get_renderer()
    prop = FontProperties(family=plt.rcParams["font.family"], size=size, weight=weight)
    max_px = width_frac * fig.bbox.width
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            trial = f"{line} {word}".strip()
            if line and renderer.get_text_width_height_descent(trial, prop, ismath=False)[0] > max_px:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    return "\n".join(lines)


def put_text(fig, y: float, text: str, size: float, weight="normal", color=INK, x=MARGIN,
             width_frac=1 - 2 * MARGIN, linespacing=1.25) -> float:
    """Place wrapped text with its top at y (figure fraction); return y of its bottom."""
    t = fig.text(x, y, wrap(fig, text, size, weight, width_frac), fontsize=size, weight=weight,
                 color=color, va="top", ha="left", linespacing=linespacing)
    return y - t.get_window_extent(fig.canvas.get_renderer()).height / fig.bbox.height


def draw_ribbon(fig, x: float, y: float, h: float) -> None:
    """Pink awareness ribbon centred at (x, y), height h (figure fractions)."""
    aspect = fig.bbox.height / fig.bbox.width
    ax = fig.add_axes((x - h * aspect / 2, y - h / 2, h * aspect, h))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1.1, 1.1)
    ax.axis("off")
    lw = h * fig.bbox.height * (0.2 / 2.2) * 72 / DPI  # band width = 0.2 data units, in points
    loop = MplPath([(0.55, -1.0), (0.2, -0.5), (-0.4, 0.1), (-0.38, 0.5),
                    (-0.36, 0.98), (0.36, 0.98), (0.38, 0.5)], [MplPath.MOVETO] + [MplPath.CURVE4] * 6)
    tail = MplPath([(0.38, 0.5), (0.4, 0.1), (-0.2, -0.5), (-0.55, -1.0)],
                   [MplPath.MOVETO] + [MplPath.CURVE4] * 3)
    for path in (loop, tail):  # tail drawn last, with a white edge, to show the crossing
        ax.add_patch(PathPatch(path, fill=False, lw=lw * 1.35, edgecolor=SURFACE, capstyle="butt"))
        ax.add_patch(PathPatch(path, fill=False, lw=lw, edgecolor=PINK, capstyle="butt"))


def new_slide(number: int, title: str, subtitle: str | None, ribbon: bool = False):
    """Slide with title, subtitle and footer; returns (fig, y of the bottom of the header)."""
    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN), dpi=DPI)
    title_width = 1 - 2 * MARGIN - (0.12 if ribbon else 0)
    y = put_text(fig, 0.955, title, 25, "bold", width_frac=title_width, linespacing=1.15)
    if subtitle:
        y = put_text(fig, y - 0.02, subtitle, 16, color=INK_2, width_frac=title_width)
    if ribbon:
        draw_ribbon(fig, 1 - MARGIN - 0.04, 0.915, 0.09)
    fig.add_artist(plt.Line2D([MARGIN, 1 - MARGIN], [0.085, 0.085], color=GRAY, lw=1, transform=fig.transFigure))
    fig.text(MARGIN, 0.064, FOOTER[0], fontsize=9.5, color=INK_2, va="top")
    fig.text(MARGIN, 0.038, FOOTER[1], fontsize=9.5, color=INK_2, va="top", style="italic")
    fig.text(1 - MARGIN, 0.038, f"{number}/{N_SLIDES}", fontsize=10, color=INK_2, va="top", ha="right")
    return fig, y


def clean_axes(ax, keep=("bottom",)) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


# ---------- data ----------

def weighted_share(df: pd.DataFrame, by) -> pd.Series:
    dense = (df["high_density"] * df["count"]).groupby(by, observed=True).sum()
    return dense / df.groupby(by, observed=True)["count"].sum()


def overall_share(df: pd.DataFrame) -> float:
    return (df["high_density"] * df["count"]).sum() / df["count"].sum()


def load_numbers() -> dict:
    df = pd.read_parquet(PROCESSED_DIR / "bcsc_clean.parquet")
    known = df[df["high_density"].notna()]
    n = {"total_mammograms": int(df["count"].sum()), "known_mammograms": int(known["count"].sum()),
         "overall": overall_share(known)}

    age = known["age_group_5_years_label"].astype(str).replace({"18-29": "Under 35", "30-34": "Under 35"})
    age_order = ["Under 35"] + [a for a in known["age_group_5_years_label"].cat.categories if a not in ("18-29", "30-34")]
    n["by_age"] = weighted_share(known, age).loc[age_order]
    n["age_40_49"] = overall_share(known[known["age_group_5_years"].isin([4, 5])])

    bmi = known[known["bmi_group"].notna()]
    n["by_bmi"] = weighted_share(bmi, bmi["bmi_group_label"])

    fam = known[known["first_degree_hx"].notna()]
    n["fam_overall"] = weighted_share(fam, fam["first_degree_hx_label"])
    share = weighted_share(fam, [fam["age_group_5_years_label"], fam["first_degree_hx_label"]]).unstack()
    n["fam_diff_by_age"] = ((share["Yes"] - share["No"]) * 100).loc["40-44":]
    mix = fam.groupby([fam["first_degree_hx_label"], fam["age_group_5_years"] >= 8], observed=True)["count"].sum().unstack()
    n["fam_60plus"] = mix[True] / mix.sum(axis=1)

    test = pd.read_parquet(PROCESSED_DIR / "model_menohrt_test.parquet")
    booster = xgb.Booster()
    booster.load_model(PROCESSED_DIR / "xgb_menohrt.json")
    p = booster.predict(xgb.DMatrix(test[booster.feature_names]))
    w = test["sample_weight"].to_numpy()
    n.update(pred=p, pred_w=w, test_mammograms=int(w.sum()),
             uncertain=w[(p >= 0.35) & (p <= 0.65)].sum() / w.sum(),
             confident=w[(p < 0.2) | (p > 0.8)].sum() / w.sum())
    return n


# ---------- slides ----------

def slide_intro(n):
    fig, y = new_slide(1, "What 6.8 million mammograms can teach us about breast cancer screening", None, ribbon=True)
    blocks = [
        ("What is mammographic density?",
         "It describes how much glandular tissue the breast has relative to fat, as seen on the mammogram. "
         "Radiologists rate it in four levels. The top two count as high tissue density."),
        ("Why does it matter?",
         "High tissue density can hide tumors on the image, and it is linked to a higher risk of breast cancer."),
        ("Our question",
         "Can common risk factors such as age, BMI, menopause or family history predict a mammogram's tissue density?"),
    ]
    y -= 0.05
    x_text = MARGIN + 0.035
    for heading, body in blocks:
        y_h = put_text(fig, y, heading, 19.5, "bold", x=x_text, width_frac=1 - MARGIN - x_text)
        y_b = put_text(fig, y_h - 0.014, body, 17, color=INK_2, x=x_text, width_frac=1 - MARGIN - x_text)
        fig.add_artist(Rectangle((MARGIN, y_b), 0.009, y - y_b, color=PINK, transform=fig.transFigure))
        y = y_b - 0.065
    put_text(fig, max(y, 0.16), "Swipe to see what the data show  >", 17, "bold", color=PINK)
    return fig


def slide_overall(n):
    pct = round(n["overall"] * 100)
    fig, y = new_slide(2, "Almost half of screening mammograms show high tissue density",
                       "High density is common, so it is worth knowing whether it applies to you.")
    ax = fig.add_axes((MARGIN + 0.08, 0.215, 1 - 2 * MARGIN - 0.16, y - 0.26))
    ax.set_aspect("equal")
    for i in range(100):
        col, row = divmod(i, 10)  # fill column by column from the left
        ax.add_patch(FancyBboxPatch((col + 0.08, 9 - row + 0.08), 0.84, 0.84,
                                    boxstyle="round,pad=0,rounding_size=0.12",
                                    color=PINK if i < pct else GRAY, lw=0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.text(0.5, 0.165, f"{pct} out of every 100 mammograms", fontsize=21, weight="bold", color=PINK, ha="center")
    fig.text(0.5, 0.125, f"based on {n['known_mammograms'] / 1e6:.1f}M mammograms with a recorded density",
             fontsize=12.5, color=INK_2, ha="center")
    return fig


def slide_by_age(n):
    s = n["by_age"]
    fig, y = new_slide(3, "High tissue density is most common at younger ages",
                       "Routine screening often starts at 40, when high density is still common.")
    ax = fig.add_axes((MARGIN + 0.17, 0.13, 1 - 2 * MARGIN - 0.2, y - 0.165))
    ypos = np.arange(len(s))[::-1]
    highlight = [a in ("40-44", "45-49") for a in s.index]
    ax.barh(ypos, s * 100, color=[PINK if h else GRAY for h in highlight], height=0.72)
    for yy, v, h in zip(ypos, s, highlight):
        ax.text(v * 100 + 1.2, yy, f"{v:.0%}", va="center", fontsize=14, color=INK, weight="bold" if h else "normal")
    ax.set_yticks(ypos, [a.replace("85+", "85 and over") for a in s.index], fontsize=14)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_xticks([])
    ax.set_xlim(0, 135)
    clean_axes(ax, keep=())
    start_40 = ypos[list(s.index).index("40-44")] + 0.5
    ax.axhline(start_40, color=INK_2, lw=1.2, ls=(0, (4, 3)))
    ax.text(135, start_40 - 0.15, "Routine screening\noften starts at 40", ha="right", va="top",
            fontsize=13, color=INK_2, linespacing=1.15)
    y45 = ypos[list(s.index).index("45-49")]
    ax.annotate(f"Ages 40 to 49:\nabout 2 in 3\nmammograms ({n['age_40_49']:.0%})",
                xy=(s["45-49"] * 100 + 11, y45 - 0.1), xytext=(108, y45 - 3.0), fontsize=14, color=PINK,
                weight="bold", ha="center", va="center",
                arrowprops=dict(arrowstyle="-|>", color=PINK, lw=1.5, shrinkA=6, shrinkB=2))
    fig.text(MARGIN, 0.105, "Share of screening mammograms with high tissue density, by age group",
             fontsize=11.5, color=INK_2)
    return fig


def slide_by_bmi(n):
    s = n["by_bmi"]
    labels = ["Under 25", "25 to 29.9", "30 to 34.9", "35 and over"]
    fig, y = new_slide(4, "Mammographic density is higher at lower BMI",
                       "Tissue density shapes what a mammogram can show, at every BMI.")
    ax = fig.add_axes((MARGIN, 0.33, 1 - 2 * MARGIN, y - 0.39))
    bars = ax.bar(labels, s * 100, color=[PINK, GRAY, GRAY, GRAY], width=0.7)
    for b, v in zip(bars, s):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{v:.0%}", ha="center", fontsize=20,
                weight="bold", color=INK)
    ax.set_ylim(0, 100)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=14, length=0, pad=8)
    clean_axes(ax)
    ax.annotate("Lower BMI:\nabout 2 in 3 mammograms", xy=(0.3, s.iloc[0] * 100 + 8), xytext=(1.75, 86),
                fontsize=14, color=PINK, weight="bold", ha="center", va="center",
                arrowprops=dict(arrowstyle="-|>", color=PINK, lw=1.5, connectionstyle="arc3,rad=0.25"))
    fig.text(MARGIN, 0.272, "Share of screening mammograms with high tissue density, by BMI group",
             fontsize=11.5, color=INK_2)
    box_top, box_bottom = 0.235, 0.115
    fig.add_artist(FancyBboxPatch((MARGIN, box_bottom), 1 - 2 * MARGIN, box_top - box_bottom,
                                  boxstyle="round,pad=0,rounding_size=0.015", color=PINK_TINT, lw=0,
                                  transform=fig.transFigure))
    put_text(fig, box_top - 0.022, "Higher BMI is still linked to higher breast cancer risk after menopause. "
             "Density is one factor among many.", 15, "bold", x=MARGIN + 0.03, width_frac=1 - 2 * MARGIN - 0.06)
    return fig


def slide_can_we_predict(n):
    p, w = n["pred"], n["pred_w"]
    fig, y = new_slide(5, "Risk factors alone cannot tell us the result",
                       "Only the mammogram itself shows the tissue density, one more reason to get screened.")
    ax = fig.add_axes((MARGIN + 0.02, 0.29, 1 - 2 * MARGIN - 0.04, y - 0.33))
    bins = np.linspace(0, 1, 21)
    hist, _ = np.histogram(p, bins=bins, weights=w / w.sum() * 100)
    centers = (bins[:-1] + bins[1:]) / 2
    in_zone = (centers > 0.35) & (centers < 0.65)
    ax.axvspan(0.35, 0.65, color=PINK_TINT, lw=0, zorder=0)
    ax.bar(centers, hist, width=0.046, color=np.where(in_zone, PINK, GRAY), zorder=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, hist.max() * 1.6)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"], fontsize=13)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_yticks([])
    clean_axes(ax)
    ax.set_xlabel("Predicted chance of high tissue density", fontsize=13.5, labelpad=8)
    ax.text(0.5, hist.max() * 1.55, f"{n['uncertain']:.0%} of mammograms", ha="center", va="top",
            fontsize=22, weight="bold", color=PINK)
    ax.text(0.5, hist.max() * 1.36, "land between 35% and 65%:\nclose to a coin flip", ha="center", va="top",
            fontsize=14.5, color=INK)
    put_text(fig, 0.205, f"Only {n['confident']:.0%} get a confident prediction (below 20% or above 80%).",
             15.5, "bold")
    put_text(fig, 0.145, f"Predictions from 8 common risk factors, checked on {n['test_mammograms'] / 1e6:.1f}M "
             "mammograms that were not used to build the predictions.", 12.5, color=INK_2)
    return fig


def slide_family_history(n):
    overall, diff, mix = n["fam_overall"], n["fam_diff_by_age"], n["fam_60plus"]
    fig, y = new_slide(6, "Family history seems unrelated to tissue density, until you compare by age",
                       "Look closer before drawing conclusions: an average can hide a real pattern.")
    fig.text(MARGIN, y - 0.03, "All ages together: no difference", fontsize=16.5, weight="bold", va="top")
    top_h = 0.095
    ax1 = fig.add_axes((MARGIN + 0.3, y - 0.075 - top_h, 1 - 2 * MARGIN - 0.42, top_h))
    vals = [overall["No"], overall["Yes"]]
    ax1.barh([1, 0], [v * 100 for v in vals], color=GRAY, height=0.62)
    for yy, v in zip([1, 0], vals):
        ax1.text(v * 100 + 1.5, yy, f"{v:.1%}", va="center", fontsize=14, color=INK)
    ax1.set_yticks([1, 0], ["No family history", "Family history"], fontsize=13.5)
    ax1.tick_params(axis="y", length=0, pad=8)
    ax1.set_xlim(0, 60)
    ax1.set_xticks([])
    clean_axes(ax1, keep=())

    y2 = y - 0.075 - top_h - 0.05
    fig.text(MARGIN, y2, "Within each age group: higher with a family history", fontsize=16.5, weight="bold",
             va="top", color=PINK)
    fig.text(MARGIN, y2 - 0.035, "Difference in percentage points, ages 40 and over", fontsize=12.5,
             color=INK_2, va="top")
    ax2 = fig.add_axes((MARGIN + 0.17, 0.215, 1 - 2 * MARGIN - 0.3, y2 - 0.065 - 0.215))
    ypos = np.arange(len(diff))[::-1]
    ax2.barh(ypos, diff, color=PINK, height=0.66)
    for yy, v in zip(ypos, diff):
        ax2.text(v + 0.06, yy, f"+{v:.1f}", va="center", fontsize=13, color=INK)
    ax2.set_yticks(ypos, [a.replace("85+", "85 and over") for a in diff.index], fontsize=13)
    ax2.tick_params(axis="y", length=0, pad=8)
    ax2.set_xlim(0, diff.max() * 1.3)
    ax2.set_xticks([])
    clean_axes(ax2, keep=("left",))
    put_text(fig, 0.19, f"Why? Mammograms with a family history came more often from older women "
             f"({mix['Yes']:.0%} aged 60 and over vs. {mix['No']:.0%}), and density falls with age.",
             13.5, color=INK_2)
    return fig


def slide_takeaway(n):
    fig, y = new_slide(7, "No checklist replaces screening",
                       "Risk factors explain only part of the picture. The mammogram shows the rest.", ribbon=True)
    actions = [
        ("Know your screening schedule", "Ask when and how often screening is recommended for you."),
        ("Ask your doctor about your mammographic density", "It is often noted in the mammogram report."),
        ("Ask whether supplemental screening makes sense for you",
         "For some women with high density, extra imaging can be an option."),
    ]
    r = 0.04  # circle radius as a fraction of figure height
    rx = r * HEIGHT_IN / WIDTH_IN
    x_text = MARGIN + 2 * rx + 0.04
    y -= 0.07
    for i, (head, body) in enumerate(actions, start=1):
        cx, cy = MARGIN + rx, y - r
        fig.add_artist(Ellipse((cx, cy), 2 * rx, 2 * r, color=PINK, transform=fig.transFigure))
        fig.text(cx, cy, str(i), ha="center", va="center", fontsize=22, weight="bold", color=SURFACE)
        y_h = put_text(fig, y + 0.004, head, 18.5, "bold", x=x_text, width_frac=1 - MARGIN - x_text)
        y_b = put_text(fig, y_h - 0.01, body, 14.5, color=INK_2, x=x_text, width_frac=1 - MARGIN - x_text)
        y = min(y_b, cy - r) - 0.075
    put_text(fig, max(y - 0.01, 0.13), "Talk to your doctor about what is right for you.", 15.5, "bold", color=PINK)
    return fig


SLIDES = [("01_intro", slide_intro), ("02_overall", slide_overall), ("03_by_age", slide_by_age),
          ("04_by_bmi", slide_by_bmi), ("05_can_we_predict", slide_can_we_predict),
          ("06_family_history", slide_family_history), ("07_takeaway", slide_takeaway)]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = load_numbers()
    with PdfPages(OUT_DIR / "bcam_carousel.pdf") as pdf:
        for name, make in SLIDES:
            fig = make(n)
            fig.savefig(OUT_DIR / f"{name}.png", dpi=DPI)
            pdf.savefig(fig)
            plt.close(fig)
            print(f"Saved output/linkedin/{name}.png")
    print("Saved output/linkedin/bcam_carousel.pdf")

    print(f"\nNumbers used: total {n['total_mammograms']:,}; known density {n['known_mammograms']:,}; "
          f"overall {n['overall']:.4f}; ages 40-49 {n['age_40_49']:.4f}")
    print("By age:\n" + n["by_age"].round(4).to_string())
    print("By BMI:\n" + n["by_bmi"].round(4).to_string())
    print(f"Uncertain {n['uncertain']:.4f}; confident {n['confident']:.4f}; test {n['test_mammograms']:,}")
    print("Family history overall:", n["fam_overall"].round(4).to_dict(), "60+:", n["fam_60plus"].round(4).to_dict())
    print("Family history diff (pp) by age:\n" + n["fam_diff_by_age"].round(2).to_string())


if __name__ == "__main__":
    main()
