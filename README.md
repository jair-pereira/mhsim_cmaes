# Re-evaluating Algorithm Variations using Empirical Similarity
This repository contains the code, data, and additional figures for our GECCO 2023 poster.

In this study, we analyze whether high Component Similarity correlates to high Performance Similarity on nine CMA- ES variants on the COCO benchmark.
We observed a weak to moderate correlation between the similarity metrics, and that the metrics provide complementary insights into the algorithm analysis.

All the figures and processed data are in the repository.
If you want to generate it again, follow the steps below.

## Running the analysis
[1_process_bbob_data.py](https://github.com/jair-pereira/mhsim_cmaes/blob/main/1_process_bbob_data.py) downloads and processes the necessary data from the COCO Data Archive
[2_make_figures.py](https://github.com/jair-pereira/mhsim_cmaes/blob/main/2_make_figures.py) generates the following figures in the paper
1. Performance Similarity Heatmap
2. Component Similarity Heatmap
3. Pearson Correlation Index

Those figures are in the folder [data/](https://github.com/jair-pereira/mhsim_cmaes/tree/main/data)

### Dependencies
To run these scripts, it is necessary to have installed:
* cocopp (coco processing tool)
* numpy
* pandas
* sklearn
* scipy
* plotly
