import MicTrendChart from "./components/MicTrendChart";
import ResistanceTrendChart from "./components/ResistanceTrendChart";
import SpeciesSwitcher from "./components/SpeciesSwitcher";

const API = "https://mic-creep-predict.onrender.com";

const SPECIES_LABEL: Record<string, string> = {
  kpneumoniae: "K. pneumoniae",
  abaumannii:  "A. baumannii",
};

const SPECIES_AUC: Record<string, number> = {
  kpneumoniae: 0.982,
  abaumannii:  0.981,
};

async function getTrend(species: string) {
  const res = await fetch(`${API}/api/trend/mic90?species=${species}`, {
    next: { revalidate: 3600 },
  });
  if (!res.ok) throw new Error("Failed to fetch trend data");
  return res.json();
}

const LEGEND_ITEMS = [
  {
    colour: "#2563eb",
    style: "solid",
    label: "Actual MICₐ (observed)",
    desc:
      "90th-percentile MIC measured in ATLAS surveillance. The sharp rise in 2017-2018 reflects the spread of carbapenem-resistant strains entering the dataset. The plateau at 32 mg/L is a panel ceiling artifact - ATLAS reports all values >= 32 mg/L as exactly 32.",
  },
  {
    colour: "#16a34a",
    style: "dashed",
    label: "Model predicted MICₐ (2019-2022 test period)",
    desc:
      "XGBoost regression predictions on held-out test years the model never saw during training. The model correctly captures the upward trajectory but underestimates the ceiling because most training isolates were susceptible.",
  },
  {
    colour: "#d97706",
    style: "dashed",
    label: "Forecast (2023-2026, extrapolated)",
    desc:
      "Model projections beyond the available data. Treat as indicative - the real trajectory depends on emergence of new resistance mechanisms and surveillance methodology changes.",
  },
];

export default async function Home({
  searchParams,
}: {
  searchParams: Promise<{ species?: string }>;
}) {
  const { species = "kpneumoniae" } = await searchParams;
  const speciesLabel = SPECIES_LABEL[species] ?? "K. pneumoniae";
  const auc = SPECIES_AUC[species] ?? 0.982;

  const trend = await getTrend(species);
  const forecast2026 = trend.data.find(
    (d: { year: number }) => d.year === 2026
  )?.forecast_mic90?.toFixed(1);
  const trainYears = trend.data.filter(
    (d: { source: string }) => d.source === "train_actual"
  ).length;
  const latestResistance = trend.data
    .filter((d: { source: string; pct_resistant?: number }) =>
      d.source === "test_actual_and_predicted" && d.pct_resistant != null
    )
    .slice(-1)[0]?.pct_resistant?.toFixed(1);

  return (
    <main className="max-w-5xl mx-auto px-4 py-10 bg-gray-50 min-h-screen">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 tracking-tight">
          MIC Creep Watch
        </h1>
        <p className="mt-2 text-gray-600 text-base leading-relaxed">
          Tracking antibiotic resistance drift in{" "}
          <span className="font-semibold text-gray-800 italic">Klebsiella pneumoniae</span>{" "}
          and{" "}
          <span className="font-semibold text-gray-800 italic">Acinetobacter baumannii</span>{" "}
          - two WHO critical-priority pathogens - against{" "}
          <span className="font-semibold text-gray-800">Meropenem</span>, a last-resort
          carbapenem antibiotic used when all other options have failed.
        </p>
        <div className="mt-3 rounded-lg bg-blue-50 border border-blue-100 px-4 py-3 text-sm text-blue-800">
          <strong>What is Meropenem?</strong><br />A broad-spectrum carbapenem antibiotic
          reserved for life-threatening infections caused by multidrug-resistant bacteria.
          Rising MIC values - even while still technically &ldquo;susceptible&rdquo; - signal
          creeping resistance that standard clinical breakpoints may miss until it is too late.
        </div>
      </div>

      {/* Species switcher */}
      <div className="mb-4">
        <SpeciesSwitcher current={species} />
      </div>

      {/* Trend chart */}
      <section className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm mb-4">
        <h2 className="text-lg font-semibold text-gray-800 mb-1">
          MIC&#x2090; Trend - <em>{speciesLabel}</em> + Meropenem (2004-2026)
        </h2>
        <p className="text-sm text-gray-500 mb-5">
          90th-percentile MIC value by year across{" "}
          <strong>{trainYears + 4}</strong> years of ATLAS surveillance data.
          Model trained on 2004-2018; tested on 2019-2022; forecast extends to 2026.
        </p>
        <MicTrendChart data={trend.data} eucastR={trend.eucast_r_mg_l} />
      </section>

      {/* Legend explainer */}
      <section className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm mb-6 space-y-4">
        <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">
          How to read this chart
        </h3>
        {LEGEND_ITEMS.map((item) => (
          <div key={item.label} className="flex gap-3">
            <div className="mt-1 flex-shrink-0 flex items-center gap-1">
              <svg width="28" height="10">
                <line
                  x1="0" y1="5" x2="28" y2="5"
                  stroke={item.colour}
                  strokeWidth="2.5"
                  strokeDasharray={item.style === "dashed" ? "5 3" : "0"}
                />
              </svg>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-800">{item.label}</p>
              <p className="text-xs text-gray-500 leading-relaxed mt-0.5">{item.desc}</p>
            </div>
          </div>
        ))}
        <div className="flex gap-3 pt-1 border-t border-gray-100">
          <div className="mt-1 flex-shrink-0">
            <svg width="28" height="10">
              <line x1="0" y1="5" x2="28" y2="5" stroke="#dc2626" strokeWidth="1.5" strokeDasharray="6 3" />
            </svg>
          </div>
          <div>
            <p className="text-sm font-medium text-gray-800">
              EUCAST resistance breakpoint - {trend.eucast_r_mg_l} mg/L
            </p>
            <p className="text-xs text-gray-500 leading-relaxed mt-0.5">
              Isolates with MIC &ge; 8 mg/L are classified as resistant. The MIC&#x2090; crossed this
              threshold around 2018 and has not returned - indicating a structural shift in
              population-level resistance, not transient fluctuation.
            </p>
          </div>
        </div>
      </section>

      {/* Resistance rate chart */}
      <section className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm mb-6">
        <h2 className="text-lg font-semibold text-gray-800 mb-1">
          Resistance Rate - <em>{speciesLabel}</em> + Meropenem (2004-2022)
        </h2>
        <p className="text-sm text-gray-500 mb-5">
          Percentage of isolates with MIC &ge; 8 mg/L (EUCAST R breakpoint).
          Blue = observed rate; green dashed = P(R) classifier probability on the 2019-2022 test set
          (AUC-ROC = {auc.toFixed(3)}).
        </p>
        <ResistanceTrendChart data={trend.data} auc={auc} />
      </section>

      {/* Key stats row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
        {[
          { label: "Years of data", value: String(trainYears + 4) },
          {
            label: "Forecast MICₐ (2026)",
            value: forecast2026 ? `${forecast2026} mg/L` : "-",
          },
          {
            label: "Resistance rate (2022)",
            value: latestResistance ? `${latestResistance}%` : "-",
          },
          { label: "P(R) AUC-ROC", value: auc.toFixed(3) },
        ].map((s) => (
          <div
            key={s.label}
            className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm text-center"
          >
            <p className="text-2xl font-bold text-blue-600">{s.value}</p>
            <p className="text-xs text-gray-500 mt-1">{s.label}</p>
          </div>
        ))}
      </div>

      {/* Methodology note */}
      <p className="text-xs text-gray-400 leading-relaxed">
        Data source: ATLAS (Pfizer) via Vivli AMR Register · Model: XGBoost Regressor trained on
        log&#x2082;(MIC) · EUCAST 2024 breakpoint for Meropenem: R &ge; 8 mg/L ·
        Plateau at 32 mg/L reflects ATLAS panel ceiling (right-censoring), not a true biological plateau ·
        Forecast is model extrapolation beyond training period - treat as indicative only ·
        Raw patient-level data is not redistributed per Vivli data-use agreement.
      </p>
    </main>
  );
}
