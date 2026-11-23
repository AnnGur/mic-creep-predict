import MicTrendChart from "./components/MicTrendChart";
import ResistanceTrendChart from "./components/ResistanceTrendChart";
import SpeciesSwitcher from "./components/SpeciesSwitcher";

const API = "https://mic-creep-predict.onrender.com";

const SPECIES_LABEL: Record<string, string> = {
  kpneumoniae: "K. pneumoniae",
  abaumannii:  "A. baumannii",
};

const SPECIES_AUC: Record<string, number> = {
  kpneumoniae: 0.971,
  abaumannii:  0.845,
};

async function getTrend(species: string) {
  const res = await fetch(`${API}/api/trend/mic90?species=${species}`, {
    next: { revalidate: 300 },
  });
  if (!res.ok) throw new Error("Failed to fetch trend data");
  return res.json();
}

const LEGEND_ITEMS_KPNEUMONIAE = [
  {
    colour: "#2563eb",
    style: "solid",
    label: "Actual MICₐ (observed)",
    desc:
      "90th-percentile MIC measured in ATLAS surveillance. MICₐ rose from ~0.25 mg/L in 2008 to 32 mg/L by 2018, reflecting the emergence of carbapenem-resistant strains. The plateau at 32 mg/L is a panel ceiling artifact - ATLAS reports all values ≥ 32 mg/L as exactly 32.",
  },
  {
    colour: "#16a34a",
    style: "dashed",
    label: "Model predicted MICₐ (2019-2022 test period)",
    desc:
      "XGBoost regression predictions on held-out test years the model never saw during training. The model correctly captures the ceiling once resistance prevalence exceeds 10%.",
  },
  {
    colour: "#d97706",
    style: "dashed",
    label: "Forecast (2023-2026, extrapolated)",
    desc:
      "Model projections beyond the available data. Treat as indicative - the real trajectory depends on emergence of new resistance mechanisms and surveillance methodology changes.",
  },
];

const LEGEND_ITEMS_ABAUMANNII = [
  {
    colour: "#2563eb",
    style: "solid",
    label: "Actual MICₐ (observed)",
    desc:
      "90th-percentile MIC in ATLAS surveillance. A. baumannii has been at the panel ceiling (32 mg/L) since 2008, reflecting the long-established dominance of OXA-type carbapenemases. MICₐ cannot rise further within the ATLAS measurement range - resistance trajectory is tracked through prevalence (see chart below).",
  },
  {
    colour: "#16a34a",
    style: "dashed",
    label: "Model predicted MICₐ (2019-2022 test period)",
    desc:
      "XGBoost regression correctly predicts the ceiling for test years. Because resistance prevalence exceeds 10% in all years, the P(R)-corrected MICₐ is fixed at 32 mg/L.",
  },
  {
    colour: "#d97706",
    style: "dashed",
    label: "Forecast (2023-2026, extrapolated)",
    desc:
      "MICₐ forecast is constrained to the panel ceiling. Resistance burden is better described by the prevalence trend chart below.",
  },
];

export default async function Home({
  searchParams,
}: {
  searchParams: Promise<{ species?: string }>;
}) {
  const { species = "kpneumoniae" } = await searchParams;
  const speciesLabel = SPECIES_LABEL[species] ?? "K. pneumoniae";
  const auc = SPECIES_AUC[species] ?? 0.971;
  const legendItems = species === "abaumannii" ? LEGEND_ITEMS_ABAUMANNII : LEGEND_ITEMS_KPNEUMONIAE;

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

      {/* For A. baumannii: prevalence chart comes FIRST (primary metric) */}
      {species === "abaumannii" && (
        <section className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm mb-4">
          <h2 className="text-lg font-semibold text-gray-800 mb-1">
            Resistance Prevalence - <em>{speciesLabel}</em> + Meropenem (2008-2022)
          </h2>
          <p className="text-sm text-gray-500 mb-1">
            <strong>Primary metric for A. baumannii.</strong> MIC&#x2090; has been at the ATLAS panel
            ceiling (32 mg/L) since 2008 and carries no trend information. Resistance burden is
            tracked through prevalence: percentage of isolates with MIC &ge; 8 mg/L.
            Blue = observed rate; orange = observed rate re-weighted to a fixed country mix;
            green dashed = P(R) classifier (AUC-ROC = {auc.toFixed(3)}).
          </p>
          <p className="text-xs text-gray-500 mb-2">
            The observed rate rises from 44% (2008) to about 70% (2019-2022), with a step between
            2011 and 2012. Part of the rise in the test years reflects which countries contribute
            isolates: with a fixed country mix the rate is 59-64% in 2019-2022 and the 2008-2022
            trend is about 0.028 logit/yr (p = 0.03) instead of 0.078 on the raw rate.
          </p>
          <p className="text-xs text-gray-400 mb-5">
            Raw-rate logistic trend fitted on 2008-2018: +0.098 logit/yr, R&#178; = 0.66, p = 0.002.
            It already overshoots the observed test years (77% fitted for 2022 against 67% observed)
            and extrapolates to about 83% by 2026; treat this as an upper-end projection, not a
            measured creep rate.
          </p>
          <ResistanceTrendChart data={trend.data} auc={auc} />
        </section>
      )}

      {/* MIC90 trend chart */}
      <section className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm mb-4">
        <h2 className="text-lg font-semibold text-gray-800 mb-1">
          MIC&#x2090; Trend - <em>{speciesLabel}</em> + Meropenem (2008-2026)
        </h2>
        <p className="text-sm text-gray-500 mb-5">
          {species === "abaumannii" ? (
            <>
              MIC&#x2090; (solid blue) is at the ATLAS panel ceiling throughout - resistance was
              already established before 2008. The MIC&#x2050; line (dashed, light blue) shows the
              median isolate consolidating at the ceiling by 2017, confirming population-wide
              resistance saturation.
            </>
          ) : (
            <>
              90th-percentile MIC value by year across{" "}
              <strong>{trainYears + 4}</strong> years of ATLAS surveillance data.
              Model trained on 2008-2018; tested on 2019-2022; forecast extends to 2026.
            </>
          )}
        </p>
        <MicTrendChart
          data={trend.data}
          eucastR={trend.eucast_r_mg_l}
          showMic50={species === "abaumannii"}
        />
      </section>

      {/* Legend explainer */}
      <section className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm mb-6 space-y-4">
        <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">
          How to read this chart
        </h3>
        {legendItems.map((item) => (
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

      {/* Resistance rate chart — shown below for K.pn, shown above (primary) for A.bau */}
      {species !== "abaumannii" && (
        <section className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm mb-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-1">
            Resistance Rate - <em>{speciesLabel}</em> + Meropenem (2008-2022)
          </h2>
          <p className="text-sm text-gray-500 mb-5">
            Percentage of isolates with MIC &ge; 8 mg/L (EUCAST R breakpoint).
            Blue = observed rate; green dashed = P(R) classifier probability on the 2019-2022 test set
            (AUC-ROC = {auc.toFixed(3)}).
          </p>
          <ResistanceTrendChart data={trend.data} auc={auc} />
        </section>
      )}

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
