# AI Workshohp 1

## Project topic
Counterfactual explanations in anomaly detection for images

## Goal
Investigate possibilities of counterfactual explanations for images in highly imbalanced datasets.

## Description
The main goal is to use MVTec datset and PatchCore anomaly detection algorithm, and provide counterfactual explanations for detected anomalies.

## Current state of the project
Progress so far and current state of the project is described in the main notebook (main.ipynb).

## Notes
Categories:
- bottle
- cable
- capsule
- carpet
- grid
- hazelnut
- leather
- metal_nut
- pill
- screw
- tile
- toothbrush
- transistor
- wood
- zipper

Model in patchcore_test.py for each chosen category should match the model that has been used in patchcore_train.py to train Patchcore on this category.

Categories from bottle to hazelnut have been trained on wide_resnet50_2 (index 0).
Categories from leather to zipper have been trained on resnet18 (index 1).

Possible solution to permission error when training a given category: delete (if exists) the subdirectory corresponding to the given category from the results/Patchcore/MVTecAD/ directory.

Omitted directories ---> .gitignore
