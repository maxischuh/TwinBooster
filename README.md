# :gemini::rocket: TwinBooster

[![arXiv](https://img.shields.io/badge/arXiv-2401.04478-b31b1b.svg)](https://arxiv.org/abs/2401.04478)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxischuh/TwinBooster/blob/main/twinbooster/twinbooster_example.ipynb)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Python version](https://img.shields.io/badge/python-v.3.8-blue)
![License](https://img.shields.io/badge/license-MIT-orange)

### Synergising Chemical Structures and Bioassay Descriptions for Enhanced Molecular Property Prediction in Drug Discovery

Maximilian G. Schuh, Davide Boldini, Stephan A. Sieber

@ Technical University of Munich, TUM School of Natural Sciences, Department of Bioscience, Center for Functional Protein Assemblies (CPA)

**Abstract**

The precise prediction of molecular properties can greatly accelerate the development of new drugs. However, in silico molecular property prediction approaches have been limited so far to assays for which large amounts of data are available. In this study, we develop a new computational approach leveraging both the textual description of the assay of interest and the chemical structure of target compounds. By combining these two sources of information via self-supervised learning, our tool can provide accurate predictions for assays where no measurements are available. Remarkably, our approach achieves state-of-the-art performance on the FS-Mol benchmark for zero-shot prediction, outperforming a wide variety of deep learning approaches. Additionally, we demonstrate how our tool can be used for tailoring screening libraries for the assay of interest, showing promising performance in a retrospective case study on a high-throughput screening campaign. By accelerating the early identification of active molecules in drug discovery and development, this method has the potential to streamline the identification of novel therapeutics.

## Usage

The pretrained model can be used by installing TwinBooster using pip and is easy to use.

```python
!pip install twinbooster==0.2.5

import pandas as pd
import twinbooster


# Download models
twinbooster.download_models()

# Init model
tb = twinbooster.TwinBooster()

# Provide SMILES-bioassay pairs
smiles = [
    "CC1=CC(=CC=C1)CS(=O)(=O)C2=NN=C(O2)[C@H](CC3=CC=CC=C3)NC(=O)OC(C)(C)C",
    "CCC(C)(C)NC(=O)C(C1=CC=C(O1)C)N(CC2=CC=CS2)C(=O)CN3C4=CC=CC=C4N=N3",
    "CC(C)(C)NC(=O)CSC1=NC(=CC(=O)N1)C2=CC=CC=C2",
    "CCOC(=O)C1C2C=CC3(C1C(=O)N(C3)CC4=CC=CO4)O2"
]

description = "HTS for small molecule inhibitors of CHOP to regulate the unfolded protein response to ER stress. Many genetic and environmental diseases result from defective protein folding within the secretory pathway so that aberrantly folded proteins are recognized by the cellular surveillance system and retained within the endoplasmic reticulum (ER). Under conditions of malfolded protein accumulation, the cell activates the Unfolded Protein Response (UPR) to clear the malfolded proteins, and if unsuccessful, initiates a cell death response. Preliminary studies have shown that CHOP is a crucial factor in the apoptotic arm of the UPR; XBP1 activates genes encoding ER protein chaperones and thereby mediates the adaptive UPR response to increase clearance of malfolded proteins. Inhibition of CHOP is hypothesized to enhance survival by preventing UPR programmed cell death. There are currently no known small molecule CHOP inhibitors either for laboratory or clinical use."

# Predict
pred, conf = tb.predict(smiles, description, get_confidence=True)
df = pd.DataFrame({"SMILES": smiles, "Prediction": pred, "Confidence": conf})

# Show predictions
df.head()

```

An example script can be found here ```./twinbooster/twinbooster_example.ipynb```.
