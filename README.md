# Lab-dependence risk in the AI build-out

Replication package for the paper *Lab-dependence risk in the AI build-out: backlog
concentration, tail co-movement and the limits of VaR as frontier AI labs go public*.
Daily data, 4 January 2021 to 5 November 2026, 69 listed firms and 11 ETFs.

The paper asks whether equities exposed to OpenAI and Anthropic through contracted
backlog carry a distinct tail risk, and whether standard risk models capture it.

## Layout

| Folder | Contents |
|:--|:--|
| `code` | Analysis scripts, numbered in run order |
| `data` | Prices, FRED series, SEC XBRL facts, Anthropic S-1/A text |
| `results` | Every table behind the paper: 57 CSVs, 3 SHAP arrays, 1 text output |
| `results_reference_20261118` | Verified copy of `results`, for checking reproduction |
| `figures` | The 16 figures, 300 dpi PNG |

## Running it

```bash
pip install -r code/requirements.txt
```

Scripts `01` and `02` refetch the raw data. The cached files in `data/` are the ones the
paper uses; Yahoo Finance re-adjusts historical prices for later splits and dividends, so
a fresh download will not reproduce the published numbers. Run from `03`:

```bash
cd code
for s in 03_tail_risk 04_bubble_credit 05_cross_section_events 05b_robustness \
         05c_tail_beta_panel 06_ml_tail_events 06b_ml_stress_days 07_var_backtest \
         07b_placebo_var 08_commitment_stress 09_ipo_cohort 13_audit_checks \
         13b_audit_robustness; do python $s.py || break; done
```

About 27 minutes, of which `06_ml_tail_events.py` is 21. Rebuilding the figures and the
manuscript additionally needs pandoc and Node with Playwright:

```bash
python 10_figures.py && python 11_build_manuscripts.py && python 12_submission_docs.py
```

Refetching from SEC EDGAR requires your own contact address in `UA` in
`02_sec_fundamentals.py`; EDGAR rejects requests without one.

## Checking it

```bash
python code/verify_reproduction.py      # results/ against the reference snapshot
python code/check_prose_numbers.py      # numbers quoted in the paper against results/
python code/13_audit_checks.py          # nine estimators against closed-form values
```

Two runs in one environment produce the same bytes. LightGBM's deterministic mode fixes a
result for a given build rather than across processor architectures, so its tables can move
in the third decimal on different hardware; everything else is exact. `event_pooled.txt`
carries a statsmodels timestamp and always differs, which the checker reports separately
and uses to detect a reference that is only a copy of the run being checked.

## Data

All public: Yahoo Finance for prices, FRED for rates and credit spreads, SEC EDGAR for
fundamentals and filings, each listed with its source URL in the data-sources document
accompanying the paper. The frontier-lab dependence scores and the event list are
hand-coded and live in `code/common.py`.

## Licence

MIT, see LICENSE.
