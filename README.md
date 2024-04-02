# Towards a genome-scale metabolic model of _Dunaliella salina_

## Introduction

This repository contains the code and data used to develop, evaluate and analyse a genome-scale metabolic model of _Dunaliella salina_.

## Data

The data used in this study is available in the `data` directory. The data is organized as follows:

- Biomass_composition.xlsx: Biomass composition of _Dunaliella salina_.
- model_with_media.xml: Model with media conditions defined.
- model_no_carotenoids: Model with beta-carotene and lutein contents removed from biomass

## Code

The code used in this study is available in the `code` directory. The code is organized as follows:

- 'model_validation': Jupyter notebook used to validate the model.
- 'carotenoid_production': Jupyter notebook used to simulate carotenoid production and create the phenotype phase plane.
- 'fseof': Python script used to perform FSEOF analysis.
- 'utils': Python script containing utility functions used in the study.

To reproduce most results, the packages listed in `requirements.txt` are required.

## Results

The results of this study are available in the `results` directory. The results are organized as follows:

- 'ppp': Phenotype phase planes of the model as PNG figures.
- 'fseof': Folders containing the results of FSEOF analysis.
