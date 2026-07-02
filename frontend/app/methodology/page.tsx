export default function MethodologyPage() {
  return (
    <main className="max-w-4xl mx-auto px-4 py-10 space-y-10">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 tracking-tight">Methodology</h1>
        <p className="mt-2 text-gray-600 text-base">
          How the MIC Creep prediction models were built, validated, and deployed.
        </p>
      </div>

      {/* Problem */}
      <section className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm space-y-3">
        <h2 className="text-xl font-semibold text-gray-800">Problem Definition</h2>
        <p className="text-sm text-gray-600 leading-relaxed">
          Standard susceptibility testing classifies each isolate as Susceptible (S),
          Intermediate (I), or Resistant (R) against a fixed MIC breakpoint. This binary
          framing misses <strong>MIC Creep</strong> - the gradual upward drift of MIC values
          within the susceptible range that precedes full resistance by years.
        </p>
        <p className="text-sm text-gray-600 leading-relaxed">
          The two pathogens require different modelling strategies because they exhibit
          fundamentally different MIC distributions with respect to meropenem:
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-2">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
            <p className="text-xs font-semibold text-blue-900 uppercase tracking-wide mb-1">K. pneumoniae - creep problem</p>
            <p className="text-xs text-blue-800 leading-relaxed">
              Unimodal distribution that shifted upward over 2008-2018. We predict
              continuous <strong>log&#x2082;(MIC)</strong> via XGBoost regression, then derive
              population MIC&#x2090; using a P(R)-corrected ceiling estimate.
            </p>
          </div>
          <div className="bg-orange-50 border border-orange-200 rounded-lg p-3">
            <p className="text-xs font-semibold text-orange-900 uppercase tracking-wide mb-1">A. baumannii - switch problem</p>
            <p className="text-xs text-orange-800 leading-relaxed">
              Strongly bimodal: susceptible peak ~0.25 mg/L, resistant peak 32 mg/L,
              almost nothing in between. MIC&#x2090; has been at the ATLAS panel ceiling
              since 2008. Primary metric = <strong>resistance prevalence trend</strong>{" "}
              (XGBoost classifier + logistic regression on logit scale).
            </p>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-3 mt-1">
          {[
            { label: "Primary antibiotic", value: "Meropenem (last-resort carbapenem)" },
            { label: "R breakpoint", value: "MIC > 8 mg/L (EUCAST 2024)" },
          ].map(({ label, value }) => (
            <div key={label} className="bg-gray-50 rounded-lg p-3">
              <p className="text-xs text-gray-500 uppercase tracking-wide">{label}</p>
              <p className="text-sm font-medium text-gray-800 mt-0.5">{value}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Data */}
      <section className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm space-y-3">
        <h2 className="text-xl font-semibold text-gray-800">Data Sources</h2>
        <p className="text-sm text-gray-600 leading-relaxed">
          <strong>ATLAS</strong> (Antimicrobial Testing Leadership and Surveillance, Pfizer),
          accessed via the Vivli AMR Register under a Data Use Agreement.
          Raw isolate-level data is never publicly exposed per the Vivli DUA.
          K. pneumoniae additionally uses two supplementary surveillance datasets
          (SIDERO-WT 2014-2022 and Venatorx 2016-2022) that share the same ATLAS
          MIC panel ceiling; A. baumannii uses ATLAS only (supplementary datasets for
          A. bau are clinical trial enrollment cohorts with selection bias toward
          treatment-failure isolates, which distorts the MIC distribution).
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="bg-gray-800 text-white">
                <th className="px-3 py-2 text-left font-medium">Parameter</th>
                <th className="px-3 py-2 text-center font-medium">K. pneumoniae</th>
                <th className="px-3 py-2 text-center font-medium">A. baumannii</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {[
                ["Datasets", "ATLAS + SIDERO-WT + Venatorx", "ATLAS only"],
                ["Total isolates (modelled)", "91,110", "34,365"],
                ["Countries", "55", "54"],
                ["Train period", "2008-2018 (n=64,429)", "2008-2018 (n=20,828)"],
                ["Test period", "2019-2022 (n=26,681)", "2019-2022 (n=13,537)"],
                ["% censored MIC", "~75%", "~10%"],
                ["% resistant train (MIC > 8 mg/L)", "~7-15%", "~44-70%"],
              ].map(([param, kp, ab]) => (
                <tr key={param} className="even:bg-gray-50">
                  <td className="px-3 py-2 text-gray-700 font-medium">{param}</td>
                  <td className="px-3 py-2 text-gray-600 text-center">{kp}</td>
                  <td className="px-3 py-2 text-gray-600 text-center">{ab}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-xs text-amber-800">
          <strong>Censored MIC treatment:</strong> values reported as &ldquo;&gt;8&rdquo; or &ldquo;&le;0.5&rdquo;
          are resolved using boundary doubling-dilution imputation (&gt;8 → 16 mg/L; &le;0.5 → 0.25 mg/L)
          before log&#x2082; transformation. This is standard epidemiological convention
          (Turnidge &amp; Paterson 2007). The high censoring rate in K. pneumoniae reflects
          the large susceptible population clustered at the assay floor.
        </div>
      </section>

      {/* Features */}
      <section className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm space-y-3">
        <h2 className="text-xl font-semibold text-gray-800">Feature Engineering</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="bg-gray-800 text-white">
                <th className="px-3 py-2 text-left font-medium">Feature</th>
                <th className="px-3 py-2 text-left font-medium">Type</th>
                <th className="px-3 py-2 text-left font-medium">Role</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {[
                ["year", "Continuous", "Primary MIC creep temporal signal"],
                ["gender_male", "Binary", "Sex covariate"],
                ["age_paediatric / age_elderly", "Binary", "Age group (adult is reference)"],
                ["military_proxy", "Binary", "Wound + male + 18-60 - combat infection proxy"],
                ["spec_wound / blood / respiratory / urine / peritoneal", "OHE (5)", "Specimen source"],
                ["ward_icu / ward_emergency / ward_surgical / ward_paediatric_ward / ward_clinic", "OHE (5)", "Hospital setting - top SHAP driver for A. baumannii"],
                ["ctry_* (50-65 cols)", "OHE", "Country of isolation; countries with < 50 training isolates collapsed to 'Other'"],
                ["KPC_pos, NDM_pos, OXA_pos, VIM_pos, IMP_pos, GES_pos", "Binary x6", "Carbapenemase gene PCR results"],
                ["pct_censored_year", "Float", "Annual censoring rate - methodology control for panel artifact"],
              ].map(([feat, type, role]) => (
                <tr key={feat} className="even:bg-gray-50">
                  <td className="px-3 py-2 font-mono text-xs text-gray-700">{feat}</td>
                  <td className="px-3 py-2 text-gray-600 text-xs">{type}</td>
                  <td className="px-3 py-2 text-gray-600 text-xs">{role}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-gray-500">
          K. pneumoniae: 83 features (55 country dummies). A. baumannii: 76 features (54 country dummies).
          The time split is strictly chronological - 2008-2018 train, 2019-2022 test, never shuffled.
          Pre-2008 years excluded due to sparse sampling (n &lt; 4,000/year).
        </p>
      </section>

      {/* Models */}
      <section className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm space-y-4">
        <h2 className="text-xl font-semibold text-gray-800">Models</h2>

        <div>
          <h3 className="text-base font-semibold text-gray-700 mb-2">K. pneumoniae - Regression pipeline</h3>
          <p className="text-sm text-gray-600 leading-relaxed mb-2">
            XGBoost Regressor (primary) + Random Forest baseline, both predicting log&#x2082;(MIC).
            Resistant isolates receive 3x sample weight to prevent underfitting the clinically
            relevant tail. Hyperparameters tuned with 60 Optuna trials using time-aware
            cross-validation on training data only.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="bg-gray-800 text-white">
                  <th className="px-3 py-2 text-left font-medium">Model</th>
                  <th className="px-3 py-2 text-center font-medium">RMSE (all)</th>
                  <th className="px-3 py-2 text-center font-medium">RMSE (resistant)</th>
                  <th className="px-3 py-2 text-center font-medium">R²</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {[
                  ["Random Forest baseline", "1.56", "2.87", "0.63"],
                  ["XGBoost tuned (primary)", "2.05", "2.56", "0.69"],
                ].map(([model, rmse, rmseR, r2]) => (
                  <tr key={model} className="even:bg-gray-50">
                    <td className="px-3 py-2 text-gray-700 font-medium text-xs">{model}</td>
                    <td className="px-3 py-2 text-gray-600 text-center text-xs">{rmse}</td>
                    <td className="px-3 py-2 text-gray-600 text-center text-xs">{rmseR}</td>
                    <td className="px-3 py-2 text-gray-600 text-center text-xs">{r2}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-gray-500 mt-1">Metrics on 2019-2022 test set. Units: log&#x2082; MIC.</p>
        </div>

        <div>
          <h3 className="text-base font-semibold text-gray-700 mb-2">A. baumannii - Prevalence pipeline (primary)</h3>
          <p className="text-sm text-gray-600 leading-relaxed mb-2">
            Because MIC&#x2090; is at the ATLAS panel ceiling in every year of the study period,
            MIC&#x2090; trajectory carries no information for A. baumannii. The primary model is
            a <strong>binary XGBoost classifier</strong> that predicts P(R) per isolate.
            Recency-weighted training (<code className="bg-gray-100 px-1 rounded text-xs">w = 0.85^(max_year - year)</code>)
            gives more influence to recent resistance patterns, accounting for the fact that OXA-type
            carbapenemases consolidated their dominance post-2015.
          </p>
          <p className="text-sm text-gray-600 leading-relaxed mb-2">
            Predicted per-isolate P(R) is aggregated to year-level resistance prevalence,
            then a <strong>logistic regression on the logit scale</strong> fits the temporal trend
            to provide a smooth estimate and enables forecasting.
            95% confidence intervals per observed year use the Wilson score method.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="bg-gray-800 text-white">
                  <th className="px-3 py-2 text-left font-medium">Metric</th>
                  <th className="px-3 py-2 text-center font-medium">Value</th>
                  <th className="px-3 py-2 text-left font-medium">Interpretation</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {[
                  ["P(R) classifier AUC-ROC", "0.981", "Near-perfect separation of resistant vs susceptible isolates"],
                  ["Logistic trend slope", "+0.098 logit/yr", "Steady increase in resistance probability year on year"],
                  ["Trend R² (2008-2022)", "0.659", "66% of year-to-year prevalence variance explained by time alone"],
                  ["Trend p-value", "0.0024", "Resistance increase is statistically significant"],
                  ["Resistance prevalence 2022", "~70%", "Up from ~44% in 2008"],
                  ["Forecast 2026", "~77%", "Extrapolated from logistic trend"],
                ].map(([metric, val, interp]) => (
                  <tr key={metric} className="even:bg-gray-50">
                    <td className="px-3 py-2 text-gray-700 text-xs font-medium">{metric}</td>
                    <td className="px-3 py-2 text-gray-800 font-bold text-xs text-center">{val}</td>
                    <td className="px-3 py-2 text-gray-600 text-xs">{interp}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-sm text-gray-600 leading-relaxed mt-2">
            A regression model (XGBoost, same architecture) is also trained as a secondary artefact
            for individual MIC prediction and SHAP explainability, but it does not drive the
            population-level surveillance output.
          </p>
        </div>
      </section>

      {/* P(R)-corrected MIC90 */}
      <section className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm space-y-3">
        <h2 className="text-xl font-semibold text-gray-800">Population-level MIC&#x2090; Estimation</h2>
        <p className="text-sm text-gray-600 leading-relaxed">
          Year-level MIC&#x2090; is the 90th-percentile MIC across all isolates collected in a given year.
          Naively taking the 90th percentile of individual regression predictions underestimates this value
          when resistance prevalence is high, because the resistant subpopulation is right-censored at the
          panel ceiling (32 mg/L in ATLAS). We use a <strong>country-stratified hybrid estimate</strong>:
        </p>
        <ol className="text-sm text-gray-600 space-y-1 list-decimal list-inside leading-relaxed">
          <li>
            A logistic P(R) classifier predicts resistance probability per isolate (AUC-ROC &ge; 0.98).
          </li>
          <li>
            If <strong>global P(R) &gt; 10%</strong> for the year, MIC&#x2090; = 32 mg/L (panel ceiling).
            At this resistance level, the 90th-percentile isolate falls within the resistant population
            by arithmetic necessity, and ATLAS censors all resistant values at 32 mg/L.
          </li>
          <li>
            If global P(R) &le; 10% (transitional period): compute per-country P(R); apply the ceiling
            to countries exceeding 10%; compute Q0.90 of regression predictions for others; weight by
            country sample size to recover the population MIC&#x2090;.
          </li>
        </ol>
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-xs text-amber-800">
          <strong>Why not a pure country-stratified approach throughout?</strong> When many countries
          are individually in the 5-10% transition zone, the weighted average of per-country Q0.90
          values undershoots the true population MIC&#x2090;. The global P(R) check correctly assigns
          the ceiling once the overall population crosses the threshold - matching the observed ATLAS
          MIC&#x2090; of 32 mg/L from 2019 onward for K. pneumoniae.
        </div>
        <p className="text-sm text-gray-600 leading-relaxed">
          For <strong>A. baumannii</strong>, global P(R) has exceeded 10% in every year of the
          study period. MIC&#x2090; is therefore always at the ceiling (32 mg/L) and is not a useful
          surveillance metric. See the prevalence pipeline above.
        </p>
      </section>

      {/* Evaluation */}
      <section className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm space-y-3">
        <h2 className="text-xl font-semibold text-gray-800">Evaluation</h2>

        <p className="text-sm text-gray-600 leading-relaxed">
          Models are evaluated on the held-out test set (2019-2022) and with a
          6-fold <strong>walk-forward cross-validation</strong> (test years 2014-2019) using fixed
          Optuna-tuned hyperparameters - no re-tuning per fold.
        </p>

        <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 text-xs text-gray-700 space-y-2">
          <p className="font-semibold">Walk-forward CV findings (K. pneumoniae):</p>
          <p>
            Mean RMSE 1.92 &plusmn; 0.31 across 6 folds; R&#178; 0.56 &plusmn; 0.12.
            The official test R&#178; (0.69) sits at the optimistic end of the fold
            distribution - the model performs best once resistance is established
            (P(R) &gt; 10%). In early folds (2014-2016) the global P(R) threshold is
            not yet crossed and the country-stratified hybrid MIC&#x2090; (4-10 mg/L)
            substantially outperforms the global estimate (0.1 mg/L) against observed
            values of 0.5-16 mg/L.
          </p>
          <p className="font-semibold mt-1">Walk-forward CV findings (A. baumannii):</p>
          <p>
            Mean RMSE 2.43 &plusmn; 0.21; R&#178; 0.29 &plusmn; 0.08. Regression quality is
            low but expected: the bimodal MIC distribution creates a difficult regression target.
            MIC&#x2090; was correctly predicted at ceiling (32 mg/L) in all 6 folds because
            P(R) exceeded 10% in every year from 2014 onward. The P(R) classifier mean error
            across folds was -2.3 percentage points.
          </p>
        </div>

        <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-xs text-amber-800">
          <strong>Why RMSE on the resistant subset is the primary regression metric:</strong> K. pneumoniae
          has ~75% censored MIC values at the assay floor. R&#178; on the full dataset is dominated by
          this floor spike. The resistant subset (7-17% of isolates) is where clinical decisions are
          made - that is where predictive accuracy matters.
        </div>
      </section>

      {/* SHAP */}
      <section className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm space-y-3">
        <h2 className="text-xl font-semibold text-gray-800">Explainability (SHAP)</h2>
        <p className="text-sm text-gray-600 leading-relaxed">
          SHAP (SHapley Additive exPlanations) values are computed for the XGBoost regression model
          to explain individual predictions. Each feature receives a SHAP value showing how much
          it shifted the predicted log&#x2082;(MIC) up or down for each isolate.
          Mean |SHAP| (average absolute contribution across all test isolates) represents
          global feature importance. Biological interpretation was validated by the domain expert
          against known AMR epidemiology.
        </p>
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 text-xs text-gray-600">
          <strong>Note on artifacts:</strong>{" "}
          The isolate-level{" "}
          <code className="bg-white border border-gray-200 px-1 rounded">is_censored</code>{" "}
          flag was removed after v1 - it dominated SHAP for both species (A. baumannii rank 1
          at mean|SHAP| 1.78; K. pneumoniae rank 2 at 0.54) despite being a data artifact, not
          biology. Its removal allowed <code className="bg-white border border-gray-200 px-1 rounded">year</code> to
          emerge as the top predictor for A. baumannii (0.29) and carbapenemase genes to dominate
          K. pneumoniae - both biologically expected. The year-level aggregate{" "}
          <code className="bg-white border border-gray-200 px-1 rounded">pct_censored_year</code>{" "}
          is retained as a methodology control.
        </div>
      </section>

      {/* Reproducibility */}
      <section className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm space-y-2">
        <h2 className="text-xl font-semibold text-gray-800">Reproducibility</h2>
        <ul className="text-sm text-gray-600 space-y-1 list-disc list-inside leading-relaxed">
          <li>All code: <a href="https://github.com/AnnGur/mic-creep-predict" className="text-blue-600 hover:underline">github.com/AnnGur/mic-creep-predict</a></li>
          <li>
            Pipeline (requires ATLAS data in <code className="bg-gray-100 px-1 rounded text-xs">data/raw/</code>):
          </li>
        </ul>
        <div className="bg-gray-900 text-gray-100 rounded-lg p-3 font-mono text-xs space-y-0.5 ml-4">
          <p><span className="text-gray-400"># K. pneumoniae</span></p>
          <p>python scripts/pipeline/2_run_feature_engineering.py --species kpneumoniae</p>
          <p>python scripts/pipeline/3_run_model_training.py --species kpneumoniae</p>
          <p>python scripts/pipeline/4_run_export.py --species kpneumoniae</p>
          <p>python scripts/pipeline/5_run_walkforward_cv.py --species kpneumoniae</p>
          <p className="mt-1"><span className="text-gray-400"># A. baumannii (ATLAS-only)</span></p>
          <p>python scripts/pipeline/2_run_feature_engineering.py --species abaumannii --atlas-only</p>
          <p>python scripts/pipeline/3_run_model_training.py --species abaumannii</p>
          <p>python scripts/pipeline/4_run_export.py --species abaumannii</p>
          <p>python scripts/pipeline/5_run_walkforward_cv.py --species abaumannii</p>
        </div>
        <ul className="text-sm text-gray-600 space-y-1 list-disc list-inside leading-relaxed mt-1">
          <li>Model artefacts: Hugging Face Hub (<code className="bg-gray-100 px-1 rounded text-xs">AnnGur/mic-creep-kpneumoniae</code>), loaded by the API at startup</li>
          <li>Raw data: Vivli AMR Register - Data Use Agreement required; not redistributable</li>
          <li>Report notebook: <code className="bg-gray-100 px-1 rounded text-xs">notebooks/final_report.ipynb</code> - pre-computed fallback metrics included</li>
        </ul>
      </section>

      <p className="text-xs text-gray-400 pb-4">
        Vivli AMR Surveillance Data Challenge 2026 &middot; Anna Gurina (Technical Lead) &middot; Inna Kucherova (Research Lead / Domain Expert)
      </p>
    </main>
  );
}
