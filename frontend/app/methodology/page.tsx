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
          We frame this as a <strong>regression problem</strong>: predict the continuous
          log&#x2082;(MIC) value for each isolate given its year, country, demographics,
          specimen source, and carbapenemase gene profile.
        </p>
        <div className="grid grid-cols-2 gap-3 mt-2">
          {[
            { label: "Target variable", value: "log₂(MIC) - continuous" },
            { label: "Antibiotic", value: "Meropenem (last-resort carbapenem)" },
            { label: "Pathogens", value: "K. pneumoniae + A. baumannii" },
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
        <h2 className="text-xl font-semibold text-gray-800">Data Source</h2>
        <p className="text-sm text-gray-600 leading-relaxed">
          <strong>ATLAS</strong> (Antimicrobial Testing Leadership and Surveillance, Pfizer),
          accessed via the Vivli AMR Register under a Data Use Agreement.
          Raw isolate-level data is never publicly exposed per the Vivli DUA.
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
                ["Total isolates (modelled)", "84,016", "34,365"],
                ["Countries", "56", "65"],
                ["Train period", "2008-2018 (n=57,335)", "2008-2018 (n=20,828)"],
                ["Test period", "2019-2022 (n=26,681)", "2019-2022 (n=13,537)"],
                ["% censored MIC", "~75%", "~10%"],
                ["% resistant (MIC > 8 mg/L)", "~7-17%", "~44-68%"],
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
          (Turnidge &amp; Paterson 2007).
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
                ["ward_icu / ward_emergency / ward_surgical / ward_paediatric_ward / ward_clinic", "OHE (5)", "Hospital setting (general ward is reference) - top SHAP driver for A. baumannii"],
                ["ctry_* (64-80 cols)", "OHE", "Country of isolation"],
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
          Total: 90 features per species (country dummy counts vary slightly). Pre-2008 years
          excluded from training due to sparse sampling (n &lt; 4,000/year). The time split is
          strictly chronological - 2008-2018 train, 2019-2022 test, never shuffled.
        </p>
      </section>

      {/* Models */}
      <section className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm space-y-3">
        <h2 className="text-xl font-semibold text-gray-800">Models and Tradeoffs</h2>
        <p className="text-sm text-gray-600 leading-relaxed">
          Two models are trained independently for each species, representing a spectrum
          from interpretable baseline to maximum accuracy on the clinically relevant resistant tail.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="bg-gray-800 text-white">
                <th className="px-3 py-2 text-left font-medium">Criterion</th>
                <th className="px-3 py-2 text-center font-medium">Random Forest</th>
                <th className="px-3 py-2 text-center font-medium">XGBoost Tuned</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {[
                ["Model type", "Advanced baseline", "Primary model"],
                ["Interpretability", "High - feature importances", "Moderate - requires SHAP"],
                ["Complexity", "Moderate - 200 parallel trees", "High - boosted ensemble + Optuna tuning"],
                ["Captures non-linearities", "Yes", "Yes"],
                ["Hyperparameter tuning", "n_estimators=200, sklearn defaults", "60 Optuna trials, time-aware CV"],
                ["Sample weighting", "3x weight on resistant isolates", "3x weight on resistant isolates"],
                ["Key strength", "Stable, interpretable, good R2", "Best RMSE on resistant subset"],
                ["Key weakness", "Weaker on resistant tail", "Black-box without SHAP"],
              ].map(([criterion, rf, xgb]) => (
                <tr key={criterion} className="even:bg-gray-50">
                  <td className="px-3 py-2 text-gray-700 font-medium text-xs">{criterion}</td>
                  <td className="px-3 py-2 text-gray-600 text-center text-xs">{rf}</td>
                  <td className="px-3 py-2 text-gray-600 text-center text-xs">{xgb}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 text-xs text-blue-800">
          <strong>Tradeoff conclusion:</strong> XGBoost achieves the best RMSE on the clinically
          critical resistant subset (MIC &gt; 8 mg/L). Random Forest balances accuracy and
          transparency for regulatory or audit contexts. Both are trained and evaluated identically.
        </div>
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 text-xs text-gray-700">
          <strong>XGBoost Quantile model (Q0.90):</strong> A third model is trained with{" "}
          <code className="bg-white border border-gray-200 px-1 rounded">objective=&quot;reg:quantileerror&quot;,
          quantile_alpha=0.9</code> using the same Optuna-tuned hyperparameters as the mean model.
          It directly optimises the 90th-percentile loss and is used for per-patient worst-case
          endpoint prediction. It is not used for the year-level MIC&#x2090; trend chart (see
          Population-level MIC&#x2090; Estimation below).
        </div>
      </section>

      {/* P(R)-corrected MIC90 */}
      <section className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm space-y-3">
        <h2 className="text-xl font-semibold text-gray-800">Population-level MIC&#x2090; Estimation</h2>
        <p className="text-sm text-gray-600 leading-relaxed">
          Year-level MIC&#x2090; is the 90th-percentile MIC across all isolates collected in a given year.
          Naively taking the 90th percentile of individual regression predictions underestimates this value
          for species with a bimodal MIC distribution (susceptible floor + resistant ceiling).
        </p>
        <p className="text-sm text-gray-600 leading-relaxed">
          We use a <strong>P(R)-corrected ceiling estimate</strong>:
        </p>
        <ol className="text-sm text-gray-600 space-y-1 list-decimal list-inside leading-relaxed">
          <li>
            A logistic classifier is trained on the same features to predict{" "}
            <strong>P(R)</strong> - the probability that each isolate is resistant (MIC &gt; 8 mg/L).
          </li>
          <li>
            If the mean P(R) for a given year exceeds 10%, the 90th percentile of the MIC
            distribution falls arithmetically within the resistant subpopulation.
          </li>
          <li>
            Resistant isolates in ATLAS are censored at the panel ceiling (32 mg/L for meropenem).
            Therefore, when P(R) &gt; 10%, predicted MIC&#x2090; = 32 mg/L.
          </li>
          <li>
            Below 10% resistance, predicted MIC&#x2090; = Q0.90 of individual regression predictions directly.
          </li>
        </ol>
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-xs text-amber-800">
          <strong>Why not use the quantile model here?</strong> The Q0.90 regression model predicts the
          90th percentile of the log&#x2082;(MIC) distribution for individual isolates. Aggregating these
          per-isolate Q0.90 predictions with a median across a year does not recover the population
          MIC&#x2090; - it gives the median of individual upper bounds, which is dominated by the
          susceptible floor in K. pneumoniae (~75% susceptible). The P(R)-corrected approach is
          methodologically correct for population-level surveillance.
        </div>
      </section>

      {/* Evaluation */}
      <section className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm space-y-3">
        <h2 className="text-xl font-semibold text-gray-800">Evaluation</h2>
        <p className="text-sm text-gray-600 leading-relaxed">
          All models are evaluated on the held-out test set (2019-2022) using three metrics,
          reported for both the full test set and the <strong>resistant subset</strong> (MIC &gt; 8 mg/L):
        </p>
        <div className="grid grid-cols-3 gap-3">
          {[
            { name: "RMSE", desc: "Root Mean Squared Error - primary metric. Lower = better. Units: log₂ MIC." },
            { name: "MAE", desc: "Mean Absolute Error - average absolute deviation. Units: log₂ MIC." },
            { name: "R²", desc: "Coefficient of determination. Misleading for bimodal distributions; use with caution." },
          ].map(({ name, desc }) => (
            <div key={name} className="bg-gray-50 rounded-lg p-3">
              <p className="text-sm font-bold text-gray-800">{name}</p>
              <p className="text-xs text-gray-500 mt-1 leading-relaxed">{desc}</p>
            </div>
          ))}
        </div>
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-xs text-amber-800">
          <strong>Why RMSE on resistant subset is the primary metric:</strong> K. pneumoniae has
          ~75% censored MIC values at the assay floor. R² on the full dataset is dominated by this
          floor spike and is near zero even for a reasonable model. The resistant subset (7-17% of
          isolates) is where clinical decisions are made - that is where predictive accuracy matters.
        </div>
        <p className="text-sm text-gray-600 leading-relaxed">
          The train/test split is strictly time-ordered. Random shuffling is explicitly disabled.
          This simulates real deployment: train on 11 years of historical data (2008-2018), predict on 4
          unseen future years (2019-2022). XGBoost hyperparameters are tuned using time-aware cross-validation
          on the training set only - the test set is never seen during tuning.
        </p>
      </section>

      {/* SHAP */}
      <section className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm space-y-3">
        <h2 className="text-xl font-semibold text-gray-800">Explainability (SHAP)</h2>
        <p className="text-sm text-gray-600 leading-relaxed">
          SHAP (SHapley Additive exPlanations) values are computed for the XGBoost model to
          explain individual predictions. Each feature receives a SHAP value showing how much
          it shifted the predicted log&#x2082;(MIC) up (toward resistance) or down (toward
          susceptibility) for each isolate.
        </p>
        <p className="text-sm text-gray-600 leading-relaxed">
          <strong>Mean |SHAP|</strong> is reported - the average absolute contribution across all
          test isolates, representing global feature importance. Biological interpretation was
          validated by the domain expert against known AMR epidemiology.
        </p>
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 text-xs text-gray-600">
          <strong>Note on artifacts:</strong>{" "}
          The isolate-level{" "}
          <code className="bg-white border border-gray-200 px-1 rounded">is_censored</code>{" "}
          flag was removed after v1 - it dominated SHAP for both species (A. baumannii rank 1
          at mean|SHAP| 1.78; K. pneumoniae rank 2 at 0.54) despite being a data artifact, not
          biology. Its removal allowed <code className="bg-white border border-gray-200 px-1 rounded">year</code> to
          emerge as the top predictor for A. baumannii (0.33) and carbapenemase genes to dominate
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
          <li>Pipeline: <code className="bg-gray-100 px-1 rounded text-xs">bash run_all.sh</code> (requires ATLAS raw data in <code className="bg-gray-100 px-1 rounded text-xs">data/raw/</code>)</li>
          <li>Model artefacts: Hugging Face Hub (loaded by the API at startup)</li>
          <li>Raw data: Vivli AMR Register - Data Use Agreement required; not redistributable</li>
          <li>Report notebook: <code className="bg-gray-100 px-1 rounded text-xs">notebooks/final_report.ipynb</code> - Colab-compatible with pre-computed fallback metrics</li>
        </ul>
      </section>

      <p className="text-xs text-gray-400 pb-4">
        Vivli AMR Surveillance Data Challenge 2026 - Anna Gurina (Technical Lead) &middot; Inna Kucherova (Research Lead / Domain Expert)
      </p>
    </main>
  );
}
