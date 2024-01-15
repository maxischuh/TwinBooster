# :gemini::rocket: TwinBooster

[![arXiv](https://img.shields.io/badge/arXiv-2401.04478-b31b1b.svg)](https://arxiv.org/abs/2401.04478)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxischuh/TwinBooster/blob/main/twinbooster/twinbooster_example.ipynb)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Python version](https://img.shields.io/badge/python-v.3.8-blue)
![License](https://img.shields.io/badge/license-MIT-orange)

### Synergising Large Language Models with Barlow Twins and Gradient Boosting for Enhanced Molecular Property Prediction

Maximilian G. Schuh, Davide Boldini, Stephan A. Sieber

@ Chair of Organic Chemistry II,
TUM School of Natural Sciences,
Technical University of Munich

**Abstract**

The success of drug discovery and development relies on the precise prediction of molecular activities and properties. While in silico molecular property prediction has shown remarkable potential, its use has been limited so far to assays for which large amounts of data are available. In this study, we use a fine-tuned large language model to integrate biological assays based on their textual information, coupled with Barlow Twins, a Siamese neural network using a novel self-supervised learning approach. This architecture uses both assay information and molecular fingerprints to extract the true molecular information. TwinBooster enables the prediction of properties of unseen bioassays and molecules by providing state-of-the-art zero-shot learning tasks. Remarkably, our artificial intelligence pipeline shows excellent performance on the FS-Mol benchmark. This breakthrough demonstrates the application of deep learning to critical property prediction tasks where data is typically scarce. By accelerating the early identification of active molecules in drug discovery and development, this method has the potential to help streamline the identification of novel therapeutics. 


## Usage

The pretrained model can be used by installing TwinBooster using pip.

```bash
pip install git+https://github.com/maxischuh/TwinBooster
```
or
```bash
pip install twinbooster
```

An example script can be found here ```./twinbooster/twinbooster_example.ipynb```.


_More coming soon_
