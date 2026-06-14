import PredictForm from "../components/PredictForm";

const API = "https://mic-creep-predict.onrender.com";

async function getCountries() {
  const res = await fetch(`${API}/api/countries`, { next: { revalidate: 86400 } });
  if (!res.ok) throw new Error("Failed to fetch countries");
  const data = await res.json();
  return data.countries as string[];
}

export default async function PredictPage() {
  const countries = await getCountries();

  return (
    <main className="max-w-5xl mx-auto px-4 py-10">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 tracking-tight">
          MIC Predictor
        </h1>
        <p className="mt-2 text-gray-600 text-base">
          Enter isolate characteristics to predict Meropenem MIC for{" "}
          <em className="font-medium text-gray-800">K. pneumoniae</em> and{" "}
          <em className="font-medium text-gray-800">A. baumannii</em>.
          The model returns a predicted MIC value and EUCAST interpretation.
        </p>
        <div className="mt-3 rounded-lg bg-amber-50 border border-amber-200 px-4 py-2 text-sm text-amber-800">
          For research purposes only. Not validated for clinical use.
        </div>
      </div>

      <PredictForm countries={countries} />

      {/* EUCAST breakpoints reference */}
      <section className="mt-8 rounded-2xl border border-gray-200 bg-white p-6 shadow-sm">
        <h2 className="text-base font-semibold text-gray-800 mb-3">
          EUCAST Clinical Breakpoints - Meropenem (2024)
        </h2>
        <table className="w-full text-sm mb-4">
          <thead>
            <tr className="border-b border-gray-200">
              <th className="text-left py-2 pr-4 text-xs font-semibold text-gray-500 uppercase tracking-wide">Category</th>
              <th className="text-left py-2 pr-4 text-xs font-semibold text-gray-500 uppercase tracking-wide">Symbol</th>
              <th className="text-left py-2 pr-4 text-xs font-semibold text-gray-500 uppercase tracking-wide">MIC threshold</th>
              <th className="text-left py-2 text-xs font-semibold text-gray-500 uppercase tracking-wide">Clinical meaning</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            <tr>
              <td className="py-2 pr-4 font-medium text-green-700">Susceptible</td>
              <td className="py-2 pr-4 font-mono text-gray-700">S</td>
              <td className="py-2 pr-4 text-gray-600">MIC ≤ 2 mg/L</td>
              <td className="py-2 text-gray-600">Standard dosing achieves effective concentrations.</td>
            </tr>
            <tr>
              <td className="py-2 pr-4 font-medium text-orange-600">Susceptible, Increased Exposure</td>
              <td className="py-2 pr-4 font-mono text-gray-700">I</td>
              <td className="py-2 pr-4 text-gray-600">MIC &gt; 2 and ≤ 8 mg/L</td>
              <td className="py-2 text-gray-600">Higher doses or extended infusion required. Not a treatment failure - a dosing challenge.</td>
            </tr>
            <tr>
              <td className="py-2 pr-4 font-medium text-red-700">Resistant</td>
              <td className="py-2 pr-4 font-mono text-gray-700">R</td>
              <td className="py-2 pr-4 text-gray-600">MIC &gt; 8 mg/L</td>
              <td className="py-2 text-gray-600">Treatment failure likely even at maximum doses. Alternative agents required.</td>
            </tr>
          </tbody>
        </table>
        <p className="text-xs text-gray-400">
          Breakpoints apply to <em>Enterobacterales</em> (including <em>K. pneumoniae</em>) and{" "}
          <em>Acinetobacter</em> spp. Source:{" "}
          <a
            href="https://www.eucast.org/clinical_breakpoints/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-500 hover:underline"
          >
            EUCAST Clinical Breakpoint Tables v14.0 (2024)
          </a>.
        </p>
      </section>
    </main>
  );
}
