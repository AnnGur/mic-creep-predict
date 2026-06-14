import CountryBarChart from "../components/CountryBarChart";
import CountryTable from "../components/CountryTable";

const API = "https://mic-creep-predict.onrender.com";

async function getCountryStats() {
  const res = await fetch(`${API}/api/country-stats?min_n=10`, { next: { revalidate: 3600 } });
  if (!res.ok) throw new Error("Failed to fetch country stats");
  return res.json();
}

export default async function CountriesPage() {
  const stats = await getCountryStats();
  const data = stats.data;

  const highResistance = data.filter((d: { pct_resistant: number }) => d.pct_resistant >= 50).length;
  const topCountry = data[0];

  return (
    <main className="max-w-5xl mx-auto px-4 py-10">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 tracking-tight">
          Resistance by Country
        </h1>
        <p className="mt-2 text-gray-600 text-base">
          Meropenem resistance rates in{" "}
          <em className="font-medium text-gray-800">K. pneumoniae</em> and{" "}
          <em className="font-medium text-gray-800">A. baumannii</em> by country,
          based on ATLAS surveillance data (test period 2019-2022).
          Resistance defined as MIC ≥ {stats.eucast_r_mg_l} mg/L (EUCAST 2024).
        </p>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
        {[
          { label: "Countries surveyed", value: String(data.length) },
          { label: "Countries >50% resistant", value: String(highResistance) },
          { label: "Most resistant country", value: topCountry.country },
          { label: `${topCountry.country} resistance rate`, value: `${topCountry.pct_resistant.toFixed(1)}%` },
        ].map((s) => (
          <div key={s.label} className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm text-center">
            <p className="text-2xl font-bold text-blue-600 truncate">{s.value}</p>
            <p className="text-xs text-gray-500 mt-1">{s.label}</p>
          </div>
        ))}
      </div>

      {/* Bar chart */}
      <section className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm mb-6">
        <h2 className="text-lg font-semibold text-gray-800 mb-1">
          Top 20 Countries by Resistance Rate
        </h2>
        <p className="text-sm text-gray-500 mb-4">
          Colour coding: <span className="text-red-600 font-medium">red ≥50%</span> ·{" "}
          <span className="text-orange-500 font-medium">orange ≥20%</span> ·{" "}
          <span className="text-blue-600 font-medium">blue &lt;20%</span>.
          Dashed line marks 50% resistance threshold.
        </p>
        <CountryBarChart data={data} />
      </section>

      {/* Full sortable table */}
      <section className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm mb-6">
        <h2 className="text-lg font-semibold text-gray-800 mb-1">
          All Countries — Sortable Table
        </h2>
        <p className="text-sm text-gray-500 mb-4">
          Click any column header to sort. Only countries with ≥10 isolates are shown.
        </p>
        <CountryTable data={data} />
      </section>

      <p className="text-xs text-gray-400 leading-relaxed">
        Data source: ATLAS (Pfizer) via Vivli AMR Register · Test period: 2019-2022 ·
        Min isolates per country: 10 · EUCAST breakpoint: R ≥ {stats.eucast_r_mg_l} mg/L ·
        Raw patient-level data is not redistributed per Vivli data-use agreement.
      </p>
    </main>
  );
}
