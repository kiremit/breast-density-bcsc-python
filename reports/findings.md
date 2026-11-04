# High mammographic density in the BCSC Risk Factors data – key findings

**Data:** Breast Cancer Surveillance Consortium (BCSC) Risk Factors dataset, 2005–2017. 6,028,759 screening
mammograms with a known BI-RADS density (out of 6,788,436). "High density" = BI-RADS c or d (heterogeneously or extremely dense tissue).
**Methods:** weighted shares, a weighted logistic regression (adjusted odds ratios, OR) and an XGBoost model with SHAP
explanations, trained on 80% of the data and evaluated on the other 20%. Details: `notebooks/02_modeling.ipynb`.

## Key findings

1. **Almost half of all screening mammograms show high mammographic density.** 47.3% (2,854,485 of 6,028,759)
   were rated BI-RADS c or d. High density is common, not unusual.

2. **High mammographic density is the norm before 50 and becomes less common with age.** The share is 71–72% at ages 18–34,
   63% at 45–49, 53% at 50–54 and levels off at about 32–33% from age 70. After adjusting for the other factors,
   mammograms at ages 40–44 still have 1.68× the odds of high density compared with ages 50–54 (95% CI 1.67–1.69),
   while those at ages 75–79 have 0.45× the odds (CI 0.45–0.46).

3. **Body mass index is the strongest single predictor.** 66% of mammograms with a BMI under 25 show high density,
   compared with 45% (BMI 25–30), 32% (30–35) and 22% (35+). Adjusted OR for BMI 35+ vs. under 25: 0.14
   (CI 0.14–0.14). BMI also ranks first in the model's SHAP importance (mean |SHAP| 0.37, vs. 0.25 for both age and
   menopause/HRT). Density is one factor among many: higher BMI is still linked to higher breast cancer risk
   after menopause.

4. **Hormone therapy after menopause is linked to higher mammographic density.** After menopause, 49.1% of
   mammograms with current hormone therapy show high density vs. 38.1% without (at ages 55–59: 51.6% vs. 43.3%).
   Adjusted OR 1.34 (CI 1.33–1.35), about the same as being premenopausal (OR 1.33 vs. postmenopausal without HRT).

5. **Mammograms of Asian/Pacific Islander women show the highest share of high density.** 65.1% vs. 46.5% for
   non-Hispanic white women; adjusted OR 1.71 (CI 1.70–1.72). Other groups are close to non-Hispanic white women
   after adjustment (ORs 0.92–1.01).

6. **Reproductive history and past biopsies matter, family history barely does.** Having no children (OR 1.46) or
   a first birth at 30 or later (OR 1.21) raises the odds compared with a first birth at 20–24. A previous breast
   biopsy is linked to higher odds (OR 1.44, CI 1.43–1.44). Breast cancer in a first-degree relative barely changes
   density (47.1% vs. 47.4%; OR 1.08) and is the least important factor in the model (mean |SHAP| < 0.01).

7. **These factors only partly predict density.** On held-out data, both models reach a ROC AUC of 0.72
   (logistic regression 0.720, XGBoost 0.720; 0.5 = guessing) and a PR AUC of 0.70 (no-skill 0.47). Both are well
   calibrated (Brier 0.212 vs. 0.249 for always predicting the average). XGBoost does not beat the simpler model, so
   the factors mostly add up rather than interact. Only the mammogram itself shows the tissue density.

## Limitations

- **No patient ID.** The data are aggregated counts per combination of risk factors and year. The same woman can
  appear in several years, so observations are not independent. The confidence intervals above treat every mammogram
  as independent and are therefore too narrow. Read them as "estimated very precisely", not as exact 95% intervals.
- **Mammograms, not women.** All shares are per mammogram. Women who are screened more often count more.
- **Missing data.** Weighted by mammograms, age at first period is missing for 61%, BMI for 36%, age at first birth for
  32%, menopause/HRT status for 16% and density itself for 11%. Missing values were modeled as their own category.
  If data are not missing at random, estimates can be biased. Surgical menopause is never recorded with current HRT
  use in the data, so it falls into "postmenopausal, no HRT".
- **Association, not causation.** These are observational associations adjusted for a limited set of factors.
  They do not show that changing BMI or stopping hormone therapy would change breast density.
- **Population and measurement.** BCSC registries are US screening populations, not a random sample of all women.
  BI-RADS density is a radiologist's visual judgement and varies between readers.
- **Small groups.** Women aged 18–29 make up only 0.14% of mammograms. The tree model gives this group almost no
  separate weight (SHAP ≈ 0), while the logistic regression estimates an OR of 2.25.
