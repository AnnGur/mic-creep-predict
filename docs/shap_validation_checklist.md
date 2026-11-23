# SHAP validation checklist for the domain expert

Project: MIC Creep prediction, Klebsiella pneumoniae and Acinetobacter baumannii against meropenem (Vivli AMR Data Challenge 2026).

**Purpose.** SHAP shows how much each feature moved the model's prediction. It shows what the model relies on, not what causes resistance. Before any SHAP result is quoted in a report or publication, a microbiologist or epidemiologist should confirm it is biologically and epidemiologically plausible, or flag it as an artefact.

**Version under review (frozen 2026-11-23).** K. pneumoniae: model trained 2026-07-22 (ATLAS plus SIDERO-WT and Venatorx, 93 features). A. baumannii: ATLAS-only retrain of 2026-11-23 (20,828 training rows, 84 features, 60-trial Optuna search, seeded). If either model is retrained after this date, the numbers below no longer apply and the review must be repeated.

**Source.** `reports/api/api_kpneumoniae_shap_importance.json` and `reports/api/api_abaumannii_shap_importance.json` (mean absolute SHAP on the 2019-2022 test set, regression model, log2 MIC units). Beeswarm plots: `reports/model/shap_beeswarm_{species}.png`.

**How to fill this in.** For each row, mark Plausible (Y / N / Unsure) and add a comment. Rows marked with a question need an answer before the result is used.

## Klebsiella pneumoniae (training 2008-2018, test 2019-2022)

| Rank | Feature | Mean abs SHAP | Question for the expert | Plausible? | Comment |
|---|---|---|---|---|---|
| 1 | lag1_pct_resistant_ctry | 0.752 | Previous-year resistance rate of the isolate's country. Is it acceptable that the top driver is local persistence rather than isolate biology? | | |
| 2 | KPC_pos | 0.605 | Is KPC the expected main driver of meropenem MIC in this dataset? | | |
| 3 | KPC_x_year | 0.285 | KPC by year interaction. Does a changing KPC effect over time make sense (KPC plateaued)? | | |
| 4 | OXA_pos | 0.275 | OXA-48-like enzymes raise meropenem MIC only modestly. Is the size of this effect reasonable? | | |
| 5 | NDM_pos | 0.254 | NDM is expected to be the fastest-rising mechanism but ranks below KPC because the training years are KPC-dominated. Is this explanation acceptable? | | |
| 6-7 | OXA_x_year, NDM_x_year | 0.182, 0.173 | Do the signs and sizes of the time interactions match the epidemiology you know? | | |
| 8-10 | ctry_China, ctry_India, ctry_Italy | 0.147, 0.122, 0.093 | Country effects. Plausible given known high-resistance settings, or do they mainly reflect which sites contribute isolates? | | |
| 11 | ward_icu | 0.089 | Is an ICU effect plausible? Does it overlap with the missing hospital- vs community-acquired information (ATLAS has no such flag)? | | |
| 13 | year | 0.065 | The explicit creep signal is small compared with the lag and gene features. Is that expected? | | |

## Acinetobacter baumannii (ATLAS only, training 2008-2018, test 2019-2022)

| Rank | Feature | Mean abs SHAP | Question for the expert | Plausible? | Comment |
|---|---|---|---|---|---|
| 1 | lag1_pct_resistant_ctry | 0.650 | Same question as above, and here it dominates even more. | | |
| 2 | year | 0.194 | Year ranks second although the observed prevalence is flat across 2019-2022. What is the model learning from it? | | |
| 3 | ward_icu | 0.155 | Is an ICU effect plausible for A. baumannii? | | |
| 4-5 | age_paediatric, age_elderly | 0.115, 0.081 | Are age effects of this size plausible, or do they proxy for hospital type or country? | | |
| 6 | spec_wound | 0.066 | Wound isolates matter for the military-relevance proxy. Is the direction plausible? | | |
| 7, 11-12 | ctry_Germany, ctry_United States, ctry_France | 0.053, 0.043, 0.041 | Country effects for countries that show comparatively low resistance in this data (please check the direction in the beeswarm). Do they reflect real epidemiology or sampling? | | |
| 8 | pct_censored_year | 0.050 | A methodology control for the changing assay panel, not biology. Confirm it should not be interpreted clinically. | | |
| 9 | proj_OXA_prev | 0.048 | A projected OXA prevalence curve. Is it reasonable that the OXA gene flags themselves do not appear in the top 12 when OXA-type carbapenemases dominate this species? | | |

## Cross-cutting points to decide

1. **Autoregressive lag feature.** `lag1_pct_resistant_ctry` is the top feature for both species. It helps prediction but means the model partly repeats last year's local rate. Should the report present SHAP results with and without it?
2. **Composition effects.** For A. baumannii the raw prevalence rise in 2019-2022 is largely explained by which countries contribute isolates (see `reports/model/prevalence_trend_check_abaumannii.md`). Country and lag features may carry that. Does this change how the SHAP results should be described?
3. **Gene data coverage.** Carbapenemase gene results are not available for every isolate and not for the supplementary datasets. Missing genes may make the gene features less informative than they should be.
4. **The 2011 to 2012 jump in A. baumannii prevalence (44% to 62%).** Part is composition, part appears within countries. Is a change in surveillance sampling the likely cause?
5. **Study labels.** The ATLAS file labels isolates TEST, ATLAS or Inform. Please confirm these are Pfizer-run studies and that no Merck/SMART data is included (SMART is excluded by the challenge rules).

## Sign-off

| Reviewer | Date | Species reviewed | Overall conclusion |
|---|---|---|---|
| | | | |
