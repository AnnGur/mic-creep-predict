"use client";

import { useState } from "react";

const API = "https://mic-creep-predict.onrender.com";

type PredictResult = {
  log2_mic: number;
  mic_mg_l: number;
  interpretation: "susceptible" | "intermediate" | "resistant";
  eucast_breakpoint_mg_l: number;
};

const SPECIMEN_TYPES = ["wound", "blood", "respiratory", "urine", "peritoneal", "other"];
const AGE_GROUPS = ["Paediatric", "Adult", "Elderly"];
const GENES = [
  { key: "kpc_pos", label: "KPC", desc: "most common in Europe/Americas" },
  { key: "ndm_pos", label: "NDM", desc: "dominant in South Asia" },
  { key: "oxa_pos", label: "OXA-48", desc: "common in Middle East/Europe" },
  { key: "vim_pos", label: "VIM", desc: "Greece, Italy" },
  { key: "imp_pos", label: "IMP", desc: "Japan, Asia-Pacific" },
  { key: "ges_pos", label: "GES", desc: "rare, emerging" },
];

const INTERP_STYLE = {
  resistant:     { bg: "bg-red-50",    border: "border-red-200",    text: "text-red-700",    badge: "bg-red-100 text-red-800" },
  intermediate:  { bg: "bg-orange-50", border: "border-orange-200", text: "text-orange-700", badge: "bg-orange-100 text-orange-800" },
  susceptible:   { bg: "bg-green-50",  border: "border-green-200",  text: "text-green-700",  badge: "bg-green-100 text-green-800" },
};

export default function PredictForm({ countries, species = "kpneumoniae" }: { countries: string[]; species?: string }) {
  const [year, setYear]             = useState(2024);
  const [country, setCountry]       = useState("Ukraine");
  const [sex, setSex]               = useState("Male");
  const [ageGroup, setAgeGroup]     = useState("Adult");
  const [specimen, setSpecimen]     = useState("wound");
  const [genes, setGenes]           = useState<Record<string, boolean>>({
    kpc_pos: false, ndm_pos: false, oxa_pos: false,
    vim_pos: false, imp_pos: false, ges_pos: false,
  });
  const [result, setResult]         = useState<PredictResult | null>(null);
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState<string | null>(null);

  function toggleGene(key: string) {
    setGenes((g) => ({ ...g, [key]: !g[key] }));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          species,
          year,
          country,
          sex,
          age_group: ageGroup,
          specimen_type: specimen,
          ...genes,
        }),
      });
      if (!res.ok) throw new Error(`API error ${res.status}`);
      setResult(await res.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  const style = result ? INTERP_STYLE[result.interpretation] : null;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Form */}
      <form onSubmit={handleSubmit} className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm space-y-5">
        {/* Year */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Isolation year: <span className="text-blue-600 font-semibold">{year}</span>
          </label>
          <input
            type="range" min={2004} max={2030} value={year}
            onChange={(e) => setYear(Number(e.target.value))}
            className="w-full accent-blue-600"
          />
          <div className="flex justify-between text-xs text-gray-400 mt-0.5">
            <span>2004</span><span>2030</span>
          </div>
        </div>

        {/* Country */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Country</label>
          <select
            value={country}
            onChange={(e) => setCountry(e.target.value)}
            className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {countries.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>

        {/* Sex + Age */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Sex</label>
            <div className="flex gap-2">
              {["Male", "Female"].map((s) => (
                <button
                  key={s} type="button"
                  onClick={() => setSex(s)}
                  className={`flex-1 rounded-lg border px-3 py-1.5 text-sm font-medium transition-colors ${
                    sex === s
                      ? "bg-blue-600 border-blue-600 text-white"
                      : "border-gray-300 text-gray-600 hover:border-gray-400"
                  }`}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Age group</label>
            <select
              value={ageGroup}
              onChange={(e) => setAgeGroup(e.target.value)}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {AGE_GROUPS.map((a) => <option key={a}>{a}</option>)}
            </select>
          </div>
        </div>

        {/* Specimen */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Specimen type</label>
          <div className="flex flex-wrap gap-2">
            {SPECIMEN_TYPES.map((s) => (
              <button
                key={s} type="button"
                onClick={() => setSpecimen(s)}
                className={`rounded-full border px-3 py-1 text-xs font-medium transition-colors capitalize ${
                  specimen === s
                    ? "bg-blue-600 border-blue-600 text-white"
                    : "border-gray-300 text-gray-600 hover:border-gray-400"
                }`}
              >
                {s}
              </button>
            ))}
          </div>
        </div>

        {/* Resistance genes */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Carbapenemase genes detected
          </label>
          <div className="grid grid-cols-2 gap-2">
            {GENES.map(({ key, label, desc }) => (
              <label
                key={key}
                className={`flex items-start gap-2 rounded-lg border p-2.5 cursor-pointer transition-colors ${
                  genes[key]
                    ? "border-red-300 bg-red-50"
                    : "border-gray-200 hover:border-gray-300"
                }`}
              >
                <input
                  type="checkbox"
                  checked={genes[key]}
                  onChange={() => toggleGene(key)}
                  className="mt-0.5 accent-red-600"
                />
                <div>
                  <p className="text-sm font-semibold text-gray-800">{label}</p>
                  <p className="text-xs text-gray-400">{desc}</p>
                </div>
              </label>
            ))}
          </div>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full rounded-xl bg-blue-600 px-4 py-3 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          {loading ? "Predicting..." : "Predict MIC"}
        </button>
      </form>

      {/* Result panel */}
      <div className="flex flex-col gap-4">
        {!result && !error && (
          <div className="rounded-2xl border border-dashed border-gray-300 bg-white p-8 flex items-center justify-center text-gray-400 text-sm h-full">
            Fill in the form and click Predict MIC to see the result.
          </div>
        )}

        {error && (
          <div className="rounded-2xl border border-red-200 bg-red-50 p-6 text-red-700 text-sm">
            {error}
          </div>
        )}

        {result && style && (
          <div className={`rounded-2xl border ${style.border} ${style.bg} p-6 shadow-sm`}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-800">Prediction result</h3>
              <span className={`rounded-full px-3 py-1 text-sm font-bold uppercase tracking-wide ${style.badge}`}>
                {result.interpretation}
              </span>
            </div>

            <div className="flex gap-6 mb-4">
              <div className="text-center">
                <p className={`text-4xl font-bold ${style.text}`}>
                  {result.mic_mg_l.toFixed(2)}
                </p>
                <p className="text-xs text-gray-500 mt-1">MIC (mg/L)</p>
              </div>
              <div className="text-center">
                <p className={`text-4xl font-bold ${style.text}`}>
                  {result.log2_mic.toFixed(2)}
                </p>
                <p className="text-xs text-gray-500 mt-1">log₂(MIC)</p>
              </div>
              <div className="text-center">
                <p className="text-4xl font-bold text-gray-400">
                  {result.eucast_breakpoint_mg_l}
                </p>
                <p className="text-xs text-gray-500 mt-1">EUCAST R breakpoint</p>
              </div>
            </div>

            <p className={`text-sm ${style.text} leading-relaxed`}>
              {result.interpretation === "resistant" &&
                "This isolate profile is predicted to be resistant to Meropenem. MIC exceeds the EUCAST breakpoint - last-resort carbapenem treatment is unlikely to be effective."}
              {result.interpretation === "intermediate" &&
                "This isolate falls in the 'Susceptible, Increased Exposure' (I) zone. Standard Meropenem dosing may be insufficient - higher doses or extended/continuous infusion are required to achieve effective drug concentrations at the infection site."}
              {result.interpretation === "susceptible" &&
                "This isolate profile is predicted susceptible to Meropenem. Standard dosing should achieve effective drug concentrations."}
            </p>

            <p className="text-xs text-gray-400 mt-4">
              Model: XGBoost trained on ATLAS 2004-2018. Predictions outside this distribution
              carry higher uncertainty. Not for clinical use.
            </p>
          </div>
        )}

        {/* Context box */}
        <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm text-sm text-gray-600 space-y-2">
          <p className="font-semibold text-gray-800">How this works</p>
          <p>
            The model predicts log₂(MIC) using year, country, demographics, specimen type,
            and carbapenemase gene flags. It was trained on ~70,000 ATLAS isolates (2004-2018)
            and validated on 2019-2022 data.
          </p>
          <p>
            The model includes a <span className="font-medium">military proxy</span> feature,
            automatically set to 1 when you select wound specimen + Male + Adult. It captures
            the epidemiological pattern of combat-related infections, where bacteria circulate
            in high-antibiotic environments and tend to be more resistant. All other
            combinations are still predicted - just without this extra signal.
          </p>
        </div>
      </div>
    </div>
  );
}
