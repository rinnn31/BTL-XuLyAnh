from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
DETECTED_DIR = OUTPUT_DIR / "detected"
REPORT_DIR = OUTPUT_DIR / "reports"
VISUAL_DIR = OUTPUT_DIR / "visualizations"

for folder in (DETECTED_DIR, REPORT_DIR, VISUAL_DIR):
    folder.mkdir(parents=True, exist_ok=True)

DEFECT_NAMES = {
    1: "Open",
    2: "Short",
    3: "Mousebite",
    4: "Spur",
    5: "Copper",
    6: "Pin-hole",
}
