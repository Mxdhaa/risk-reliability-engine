import os

files = [
    "plot_figure5.py",
    "plot_figure5_rcre_stages.py",
    "plot_figure6.py",
    "src/plot_covid_rcre.py",
    "src/shap_analysis.py"
]

font_setup = """
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'serif']
"""

for fpath in files:
    if not os.path.exists(fpath):
        print(f"Skipping {fpath} (not found)")
        continue
    
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check if already patched
    if "font.family" in content:
        print(f"Already patched {fpath}")
        continue
        
    # Insert it after the matplotlib import
    lines = content.splitlines()
    inserted = False
    for idx, line in enumerate(lines):
        if "import matplotlib" in line or "import matplotlib.pyplot" in line:
            lines.insert(idx + 1, font_setup)
            inserted = True
            break
    if inserted:
        new_content = "\n".join(lines)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Successfully patched font settings in {fpath}")
    else:
        print(f"Could not find matplotlib import in {fpath}")
