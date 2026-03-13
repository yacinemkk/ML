import json
import re

py_file = "ipfix_home_pipeline.py"
ipynb_file = "IoT_Colab_Training.ipynb"

with open(py_file, "r") as f:
    code = f.read()

# Split the code into blocks based on `# ── Step` or `# ──` annotations.
cells = []

# Add a prominent markdown cell at the top
cells.append({
    "cell_type": "markdown",
    "metadata": {"id": "markdown-header"},
    "source": [
        "# IoT Device Identification with Adversarial Robustness\n",
        "## Google Colab Pipeline (IoT IPFIX Home)\n",
        "\n",
        "### Includes:\n",
        "1. Full preprocessing (Normalization & Standardization)\n",
        "2. Exact Model Hyperparameters (Tables 3, 4, 5, 6)\n",
        "3. Semantic-aware Centroid Attacks (Section 4.2)\n",
        "4. Exact Plot Generation for the manuscript (Figures 5, 7, 9, 10)\n"
    ]
})

import sys

cells.append({
    "cell_type": "code",
    "metadata": {"id": "setup-colab"},
    "execution_count": None,
    "outputs": [],
    "source": [
         "!pip install xgboost scikit-learn pandas numpy matplotlib seaborn tensorflow -q\n",
         "from google.colab import drive\n",
         "drive.mount('/content/drive')"
    ]
})

blocks = re.split(r'(?=# ───+)', code)
for block in blocks:
    if not block.strip():
        continue
    # Check if block contains if __name__ == "__main__":
    if 'if __name__ == "__main__":' in block:
        block = block.replace('if __name__ == "__main__":', '# Run Pipeline\n')
        block = block.replace('    main()', 'main()')
        
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in block.split("\n")]
    })

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open(ipynb_file, "w") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated successfully!")
