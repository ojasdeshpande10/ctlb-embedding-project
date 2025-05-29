# CTLB Tweet Embedding Project

This repository contains scripts for processing and analyzing CTLB Twitter Data.

## Project Structure

### DataSampler
Contains scripts for sampling CTLB data across three years and writing the processed data to HDFS locations.

### Main Scripts

- **dep_anx_score_predictor.py**  
  Predicts depression and anxiety scores using ridge regression models in Apollo.

- **agg_scores_yr_week_cnty.py**  
  Aggregates predicted scores to year-week-county level for temporal and geographical analysis.

- **save_dlatk_table.py**  
  Utility script for saving scores or embeddings into DLATK (Differential Language Analysis ToolKit) table format.
