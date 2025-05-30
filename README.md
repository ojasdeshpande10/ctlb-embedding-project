# CTLB Tweet Embedding Project

This repository provides scripts for processing, analyzing, and aggregating embeddings and mental health scores derived from the **County Tweet Lexical Bank (CTLB)** dataset.

## Overview

The CTLB dataset comprises billions of geolocated tweets across U.S. counties over several years. In this project, we leverage **sentence-transformer embeddings** (generated via a separate pipeline — see below) to analyze mental health indicators like depression and anxiety at various levels of aggregation — user, weekly, and county level.

The embedding generation was handled in a different repository and stored in **HDFS on the Apollo server** for efficient distributed access. This repo provides downstream analysis utilities built on top of these embeddings.

---

## Project Pipeline

### 1. Message-Level Embeddings

Message embeddings were generated using a Transformer-based model in a separate repo and stored in **Parquet format** on the Apollo HDFS cluster:

Each row includes:
- `user_id`
- `message_id`
- `user-year-week`
- `embedding`
- Location metadata (e.g., FIPS code)

---

### 2. User and Region-Level Aggregation

Embeddings are aggregated using grouping scripts to generate:
- **User-week embeddings**
- **User-county embeddings**
- **County-level weekly average embeddings**

These grouped embeddings are used for downstream prediction and spatial/temporal trend analysis.

---

## Key Scripts

- `dep_anx_score_predictor.py`  
  Predicts **depression** and **anxiety** scores using Ridge Regression models trained on Facebook data. Input is user-level or county-level aggregated embeddings. Output includes mental health scores written back to HDFS or local disk.

- `agg_scores_yr_week_cnty.py`  
  Aggregates predicted scores to the **year-week-county** level for trend visualization and correlation studies.

- `save_dlatk_table.py`  
  Converts predicted scores or aggregated embeddings into a format compatible with **DLATK** (Differential Language Analysis Toolkit) for further linguistic analysis.

---


