# Fixed Income Carry & Returns Pipeline

## Overview

This project implements a fixed income analytics pipeline for European government bonds (France, Germany, Italy). It computes carry, yield-driven PnL, and DV01-normalized returns using market data and bond metadata.

The pipeline:
- cleans and standardizes raw data,
- computes daily carry using QuantLib,
- decomposes PnL into carry and yield components,
- normalizes returns using DV01,
- validates internal consistency.

---

## Key Concepts

| Metric | Description |
|---|---|
| `yield` | Yield in % (e.g. 1.396 = 1.396%) |
| `price` | Price (base 100) |
| `dv01` | Sensitivity for unit notional |
| `dy_bp` | Daily yield change (bp) |
| `pnl_carry` | Carry (PnL from time) |
| `pnl_yield` | Yield-driven PnL |
| `pnl_total_unit` | Total PnL |
| `carry_bp_equiv` | Carry in bp |
| `ret_total_per_dv01` | DV01-normalized return |

**Core identity:**

pnl_total_unit ≈ ret_total_per_dv01 × dv01

---

## Features

- QuantLib bond modeling (accurate accrual & coupons)
- Fallback carry approximation when metadata is missing
- Robust data cleaning and filtering
- DV01-normalized return framework
- Multi-country support (FR, DE, IT)

---

## Project Structure

.
├── data/
│   ├── benchmark_mids_cs.csv
│   └── metadata_bonds.csv
├── pf_V2.ipynb
└── README.md

---

## Pipeline Steps

### 1. Data Loading & Cleaning

- Load market + metadata
- Convert to daily frequency
- Deduplicate (last obs per day)
- Filter missing values
- Restrict to FR / DE / IT
- Merge metadata (coupon, maturity, frequency)

---

### 2. Carry Computation

Using QuantLib:

Carry = (ΔAccrued + Coupons received) / 100

Fallback:

Carry ≈ (price × yield) / 252

---

### 3. Return Decomposition

dy_bp = Δyield × 100  
pnl_yield = -dv01 × dy_bp  
pnl_total_unit = pnl_yield + pnl_carry  
carry_bp_equiv = pnl_carry / dv01  
ret_total_per_dv01 = -dy_bp + carry_bp_equiv  

---

### 4. Data Quality Filters

- Remove bonds with < 20 observations
- Filter abnormal carry
- Clip extreme returns

---

### 5. Validation Checks

- Yield moves in reasonable range
- Carry magnitude realistic
- Identity check:

pnl_total_unit / (ret_total_per_dv01 × dv01) ≈ 1

---

## Configuration

| Parameter | Value |
|---|---|
| COUNTRIES_UNIVERSE | FR, DE, IT |
| DV01_REF | 0.09 |
| CLIP_RET_BP | 20 |
| MIN_OBS_PER_ISIN | 20 |
| MAX_DAILY_CARRY | 0.005 |

---

## Dependencies

- numpy  
- pandas  
- QuantLib  

Install:

pip install numpy pandas QuantLib

---

## Usage

jupyter notebook pf_V2.ipynb

Example:

data = load_and_clean_raw()  
data = compute_returns(data)  

---

## Output

- Clean bond time series  
- Carry + yield PnL  
- DV01-normalized returns  
- Filtered dataset  

---

## Notes

- Built for systematic fixed income strategies  
- Focus on robustness and unit consistency  
- Easily extendable to other markets and strategies  

---

## Extensions

- DV01-neutral portfolios  
- Signal generation (carry, RV, momentum)  
- Transaction cost modeling  
- Backtesting framework  
```
