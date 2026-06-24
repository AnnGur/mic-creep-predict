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
                ["Total isolates", "89,572", "37,540"],
                ["Countries", "81", "79"],
                ["Train period", "2004-2018 (n=62,891)", "2004-2018 (n=24,003)"],
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
                ["ctry_* (64-80 cols)", "OHE", "Country of isolation"],
                ["KPC_pos, NDM_pos, OXA_pos, VIM_pos, IMP_pos, GES_pos", "Binary x6", "Carbapenemase gene PCR results"],
                ["is_censored", "Binary", "Data artifact - isolate at MIC panel floor"],
                ["pct_censored_year", "Float", "Data artifact - annual censoring rate (methodology control)"],
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
          Total: 91 features for K. pneumoniae, 89 for A. baumannii (country dummy counts differ).
          The time split is strictly chronological - 2004-2018 train, 2019-2022 test, never shuffled.
        </p>
      </section>

      {/* Models */}
      <section className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm space-y-3">
        <h2 className="text-xl font-semibold text-gray-800">Models and Tradeoffs</h2>
        <p className="text-sm text-gray-600 leading-relaxed">
          Three models are trained independently for each species, representing a spectrum
          from maximally interpretable to maximally accurate.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="bg-gray-800 text-white">
                <th className="px-3 py-2 text-left font-medium">Criterion</th>
                <th className="px-3 py-2 text-center font-medium">Linear Regression</th>
                <th className="px-3 py-2 text-center font-medium">Random Forest</th>
                <th className="px-3 py-2 text-center font-medium">XGBoost Tuned</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {[
                ["Model type", "Interpretable baseline", "Advanced baseline", "Primary model"],
                ["Interpretability", "Highest - year coefficient = MIC creep rate directly", "High - feature importances", "Moderate - requires SHAP"],
                ["Complexity", "Low - single linear equation", "Moderate - 200 parallel trees", "High - boosted ensemble + Optuna tuning"],
                ["Captures non-linearities", "No", "Yes", "Yes"],
                ["Hyperparameter tuning", "None", "n_estimators=200, sklearn defaults", "60 Optuna trials, time-aware CV"],
                ["Sample weighting", "None", "3x weight on resistant isolates", "3x weight on resistant isolates"],
                ["Key strength", "Directly readable coefficients", "Stable, interpretable, good R2", "Best RMSE on resistant subset"],
                ["Key weakness", "Cannot capture gene-country interactions", "Weaker on resistant tail", "Black-box without SHAP"],
              ].map(([criterion, lr, rf, xgb]) => (
                <tr key={criterion} className="even:bg-gray-50">
                  <td className="px-3 py-2 text-gray-700 font-medium text-xs">{criterion}</td>
                  <td className="px-3 py-2 text-gray-600 text-center text-xs">{lr}</td>
                  <td className="px-3 py-2 text-gray-600 text-center text-xs">{rf}</td>
                  <td className="px-3 py-2 text-gray-600 text-center text-xs">{xgb}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 text-xs text-blue-800">
          <strong>Tradeoff conclusion:</strong> XGBoost achieves the best RMSE on the clinically
          critical resistant subset (MIC &gt; 8 mg/L). Linear Regression provides directly readable
          MIC creep rate coefficients. Random Forest balances accuracy and transparency for regulatory
          or audit contexts. All three are trained and evaluated identically.
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
          This simulates real deployment: train on 15 years of historical data, predict on 4
          unseen future years. XGBoost hyperparameters are tuned using time-aware cross-validation
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
          <code className="bg-white border border-gray-200 px-1 rounded">is_censored</code> and{" "}
          <code className="bg-white border border-gray-200 px-1 rounded">pct_censored_year</code> rank
          highly in SHAP because censored observations are structurally at the MIC panel floor - not
          because they are biological predictors. They are flagged as data artifacts throughout.
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
