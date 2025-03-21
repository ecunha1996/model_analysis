# Reconstruction of genome-scale metabolic models for _Dunaliella salina_, _Nannochloropsis gaditana_, and _Pavlova lutheri_.

## Introduction

This repository contains the code and data used to develop, evaluate and analyse genome-scale metabolic models of _Dunaliella salina_,
_Nannochloropsis gaditana_, and _Pavlova lutheri_.

## Data

The data used in this study is available in the `data` directory. The data is organized as follows:

- absorption_spectra.xlsx: Absorption spectra of different pigmens retrived from https://doi.org/10.1016/j.dib.2019.103875.
- experimental: Experimental data for the three algae, including biomass composition, CO2 biofixation and light conversion factor calculations,  and growth data retrieved from https://doi.org/10.1039/D4FB00229F.
- models: Genome-scale metabolic models of _Dunaliella salina_, _Nannochloropsis gaditana_, and _Pavlova lutheri_.

## Code

The code used in this study is available in the `code` directory. The code is organized as follows:

- 'topological_validation': Jupyter notebook used to validate the topological properties of the model.
- 'basic_simulations': Jupyter notebook used to simulate the model under different conditions.
- 'light_evaluation': Jupyter notebook used to evaluate the light reactions of the model.
- 'light_absorption': Python script used to evaluate the photosynthetic properties of the models at different light intensities.
- 'utils': Python script containing utility functions used in the study.
- 
The following files refer to the publication https://doi.org/10.1016/j.ifacol.2024.10.007:
- 'model_validation': Jupyter notebook used to validate the model.
- 'carotenoid_production': Jupyter notebook used to simulate carotenoid production and create the phenotype phase plane.
- 'fseof': Python script used to perform FSEOF analysis.
- 'plot': Jupyter notebook used to plot diverse results for the publication.


To reproduce most results, the packages listed in `requirements.txt` are required.

## Results

The results of this study are available in the `results` directory. The results are organized as follows:

- 'figures': Figures generated in the study.
- 'npq_x': NPQ values of the models under different light intensities.
- 'ppp': Phenotype phase planes of the model as PNG figures.
- 'fseof': Folders containing the results of FSEOF analysis.
