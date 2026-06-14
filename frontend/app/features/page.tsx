import ShapChart from "../components/ShapChart";

const API = "https://mic-creep-predict.onrender.com";

async function getImportance() {
  const res = await fetch(`${API}/api/features/importance`, { next: { revalidate: 3600 } });
  if (!res.ok) throw new Error("Failed to fetch feature importance");
  return res.json();
}

const CATEGORIES = [
  { colour: "#dc2626", label: "Resistance gene (KPC, NDM, OXA, VIM...)" },
  { colour: "#2563eb", label: "Country effect" },
  { colour: "#7c3aed", label: "Temporal signal (MIC creep)" },
  { colour: "#d97706", label: "Specimen type" },
  { colour: "#16a34a", label: "Demographics" },
  { colour: "#9ca3af", label: "Data artifact (not a biological predictor)" },
];

export default async function FeaturesPage() {
  const result = await getImportance();
  const data = result.data;

  const topFeature = data[0];
  const geneFeatures = data.filter((d: { feature: string }) =>
    ["KPC_pos", "NDM_pos", "OXA_pos", "VIM_pos", "IMP_pos", "GES_pos"].includes(d.feature)
  );
  const yearFeature = data.find((d: { feature: string }) => d.feature === "year");

  return (
    <main className="max-w-5xl mx-auto px-4 py-10">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 tracking-tight">
          What Drives Resistance?
        </h1>
        <p className="mt-2 text-gray-600 text-base">
          SHAP (SHapley Additive exPlanations) values show how much each feature
          contributes to the model's MIC predictions. Higher value = stronger influence
          on the predicted resistance level.
        </p>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 mb-6">
        <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm text-center">
          <p className="text-2xl font-bold text-red-600">{topFeature.feature}</p>
          <p className="text-xs text-gray-500 mt-1">Top resistance driver</p>
        </div>
        <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm text-center">
          <p className="text-2xl font-bold text-blue-600">{geneFeatures.length}</p>
          <p className="text-xs text-gray-500 mt-1">Carbapenemase genes tracked</p>
        </div>
        <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm text-center">
          <p className="text-2xl font-bold text-purple-600">
            #{yearFeature ? data.indexOf(yearFeature) + 1 : "-"}
          </p>
          <p className="text-xs text-gray-500 mt-1">Rank of &ldquo;year&rdquo; (MIC creep signal)</p>
        </div>
      </div>

      {/* Chart */}
      <section className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm mb-6">
        <h2 className="text-lg font-semibold text-gray-800 mb-1">
          Top 20 Features by Mean |SHAP| Value
        </h2>
        <p className="text-sm text-gray-500 mb-2">
          Measured in log₂ MIC units. Hover over bars for details.
        </p>

        {/* Colour legend */}
        <div className="flex flex-wrap gap-x-4 gap-y-1 mb-5">
          {CATEGORIES.map((c) => (
            <span key={c.label} className="flex items-center gap-1.5 text-xs text-gray-600">
              <span className="inline-block w-3 h-3 rounded-sm" style={{ background: c.colour }} />
              {c.label}
            </span>
          ))}
        </div>

        <ShapChart data={data} />
      </section>

      {/* Interpretation */}
      <section className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm mb-6 space-y-3">
        <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">
          Key findings
        </h3>
        <div className="space-y-2 text-sm text-gray-600 leading-relaxed">
          <p>
            <span className="font-semibold text-red-600">Carbapenemase genes dominate.</span>{" "}
            KPC, OXA, and NDM are the top 3 biological drivers. An isolate carrying KPC shifts
            the predicted MIC by ~{topFeature.mean_abs_shap.toFixed(1)} log₂ units on average -
            equivalent to a {Math.round(2 ** topFeature.mean_abs_shap)}x increase in MIC.
          </p>
          <p>
            <span className="font-semibold text-purple-600">Year matters.</span>{" "}
            The &ldquo;year&rdquo; feature ranks #{yearFeature ? data.indexOf(yearFeature) + 1 : "-"},
            confirming a genuine temporal creep signal independent of gene prevalence -
            the core finding of this project.
          </p>
          <p>
            <span className="font-semibold text-blue-600">Geography is the second axis.</span>{" "}
            China, India, Italy, and Greece appear as strong country effects, reflecting
            regional differences in carbapenem use, infection control, and dominant resistance clones.
          </p>
          <p>
            <span className="font-semibold text-gray-500">Note on artifacts.</span>{" "}
            <code className="text-xs bg-gray-100 px-1 rounded">is_censored</code> and{" "}
            <code className="text-xs bg-gray-100 px-1 rounded">pct_censored_year</code> appear
            high but are data-structure artifacts: censored observations are structurally
            at the MIC panel floor or ceiling, not a biological signal.
          </p>
        </div>
      </section>

      <p className="text-xs text-gray-400 leading-relaxed">
        SHAP values computed via TreeExplainer on the tuned XGBoost model (test set 2019-2022).
        Mean absolute SHAP shown - the average magnitude of each feature's contribution across
        all test isolates. Biological interpretation validated against known AMR epidemiology.
      </p>
    </main>
  );
}
