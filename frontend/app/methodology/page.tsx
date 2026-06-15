export default function MethodologyPage() {
  return (
    <main className="max-w-4xl mx-auto px-4 py-10">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 tracking-tight">Methodology</h1>
        <p className="mt-2 text-gray-600 text-base">
          How MIC Creep Watch detects and predicts resistance drift in{" "}
          <em className="font-medium text-gray-800">K. pneumoniae</em> and{" "}
          <em className="font-medium text-gray-800">A. baumannii</em>.
        </p>
      </div>

      {/* Research Question */}
      <Section title="Research Question">
        <p className="text-gray-700 leading-relaxed">
          Can we predict the trajectory of antimicrobial MIC creep for{" "}
          <em>Klebsiella pneumoniae</em> and <em>Acinetobacter baumannii</em> using global
          surveillance data, and identify the molecular and geographic drivers of resistance
          escalation?
        </p>
        <div className="mt-4 rounded-lg border-l-4 border-blue-400 bg-blue-50 px-5 py-4 text-sm text-blue-900 leading-relaxed">
          <strong>MIC creep</strong> is the gradual upward drift in Minimum Inhibitory
          Concentration (MIC) values across a bacterial population over time. It is a
          pre-resistance signal that becomes clinically critical when the population MIC
          <sub>90</sub> approaches or crosses the EUCAST resistance breakpoint. For both
          species + meropenem, EUCAST 2024 sets <strong>R &gt; 8 mg/L</strong>.
        </div>
      </Section>

      {/* Data */}
      <Section title="Data">
        <p className="text-gray-700 mb-4 leading-relaxed">
          <strong>Dataset:</strong> ATLAS (Antimicrobial Testing Leadership and Surveillance),
          Pfizer/Vivli. Controlled access via Vivli AMR Surveillance Platform.
        </p>
        <div className="overflow-x-auto rounded-xl border border-gray-200">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-800 text-white">
                <Th>Species</Th>
                <Th>Antibiotic</Th>
                <Th>Isolates</Th>
                <Th>Years</Th>
                <Th>Countries</Th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              <tr className="hover:bg-gray-50">
                <td className="px-4 py-3 text-gray-800 italic">K. pneumoniae</td>
                <td className="px-4 py-3 text-gray-600">Meropenem</td>
                <td className="px-4 py-3 text-gray-600">89,572</td>
                <td className="px-4 py-3 text-gray-600">2004-2022</td>
                <td className="px-4 py-3 text-gray-600">66</td>
              </tr>
              <tr className="hover:bg-gray-50">
                <td className="px-4 py-3 text-gray-800 italic">A. baumannii</td>
                <td className="px-4 py-3 text-gray-600">Meropenem</td>
                <td className="px-4 py-3 text-gray-600">ATLAS cohort</td>
                <td className="px-4 py-3 text-gray-600">2004-2022</td>
                <td className="px-4 py-3 text-gray-600">66</td>
              </tr>
            </tbody>
          </table>
        </div>
      </Section>

      {/* Key Finding */}
      <Section title="Key Finding - MIC Creep is Confirmed">
        <div className="rounded-lg border-l-4 border-green-500 bg-green-50 px-5 py-4 text-sm text-green-900 leading-relaxed mb-4">
          For <em>K. pneumoniae</em> + meropenem: MIC<sub>90</sub> slope ={" "}
          <strong>+1.97 mg/L/yr</strong>, R&sup2; = 0.67, p = 6.3 &times; 10<sup>-6</sup>.
          Resistance rate rose from 5% (2007) to 20% (2024) - a 4x increase in 17 years.
        </div>
        <p className="text-gray-700 leading-relaxed mb-3">
          The mechanistic signature is <strong>divergence between MIC<sub>50</sub> and MIC<sub>90</sub></strong>:
          MIC<sub>50</sub> stays flat at ~0.03 mg/L for 20 years while MIC<sub>90</sub> climbs.
          This is the fingerprint of a growing resistant subpopulation, not a population-wide shift.
        </p>
        <p className="text-gray-700 leading-relaxed">
          Carbapenemase gene drivers in <em>K. pneumoniae</em>: KPC dominated 2004-2016, then NDM
          and OXA rose sharply from 2017 onward. NDM now peaks at ~9% (2022) and bypasses
          ceftazidime-avibactam, one of the last-resort salvage drugs. In{" "}
          <em>A. baumannii</em>, OXA-type carbapenemases dominate throughout.
        </p>
      </Section>

      {/* Model */}
      <Section title="Model">
        <p className="text-gray-700 leading-relaxed mb-4">
          Independent <strong>XGBoost regression models</strong> are trained for each species with
          Optuna hyperparameter tuning (60 trials). The target is{" "}
          <code className="text-xs bg-gray-100 px-1.5 py-0.5 rounded font-mono">log&#x2082;(MIC)</code>.
          A <strong>3x sample weight</strong> is applied to resistant isolates (MIC &ge; 8 mg/L)
          to prevent the model from ignoring the clinically relevant tail.
        </p>

        <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-2">
          Train / Test Split
        </h3>
        <p className="text-gray-700 text-sm mb-5 leading-relaxed">
          Strictly time-ordered: <strong>Train 2004-2018 | Test 2019-2022</strong>.
          No random shuffling - that would constitute data leakage.
        </p>

        <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-2">
          Key Features
        </h3>
        <div className="overflow-x-auto rounded-xl border border-gray-200">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-800 text-white">
                <Th>Feature</Th>
                <Th>Type</Th>
                <Th>Rationale</Th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100 text-gray-700">
              {[
                ["year", "continuous", "Primary creep driver"],
                ["KPC_pos ... GES_pos", "binary", "Carbapenemase gene presence"],
                ["ctry_*", "OHE", "Country of isolation (65 dummies)"],
                ["spec_*", "OHE", "Specimen source (5 categories)"],
                ["is_censored", "binary", "Data-structure artifact (panel floor)"],
                ["pct_censored_year", "float", "Controls for panel methodology changes"],
                ["military_proxy", "binary", "Wound + male + 18-60; combat-wound proxy"],
              ].map(([feat, type, rationale]) => (
                <tr key={feat} className="hover:bg-gray-50">
                  <td className="px-4 py-3">
                    <code className="text-xs bg-gray-100 px-1.5 py-0.5 rounded font-mono">{feat}</code>
                  </td>
                  <td className="px-4 py-3 text-gray-500">{type}</td>
                  <td className="px-4 py-3">{rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      {/* Results */}
      <Section title="Results - K. pneumoniae">
        <div className="overflow-x-auto rounded-xl border border-gray-200 mb-4">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-800 text-white">
                <Th>Model</Th>
                <Th>RMSE (all)</Th>
                <Th>RMSE (resistant subset)</Th>
                <Th>MAE (resistant)</Th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100 text-gray-700">
              <tr className="hover:bg-gray-50">
                <td className="px-4 py-3">RF baseline</td>
                <td className="px-4 py-3">1.558</td>
                <td className="px-4 py-3">2.869</td>
                <td className="px-4 py-3">2.180</td>
              </tr>
              <tr className="hover:bg-gray-50 font-semibold">
                <td className="px-4 py-3">XGBoost tuned</td>
                <td className="px-4 py-3">1.758</td>
                <td className="px-4 py-3 text-green-700">1.960</td>
                <td className="px-4 py-3 text-green-700">1.127</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="text-sm text-gray-600 leading-relaxed">
          All metrics in log&#x2082; units.{" "}
          <strong>RMSE on the resistant subset is the clinically relevant figure.</strong>{" "}
          XGBoost improves it by 32% vs the RF baseline. The higher overall RMSE is expected:
          XGBoost is less aggressive at fitting the censoring floor, which is the right trade-off.
        </p>
      </Section>

      {/* Caveats */}
      <Section title="Important Caveats">
        <ul className="space-y-3 text-sm text-gray-700 leading-relaxed">
          <li className="flex gap-3">
            <span className="mt-0.5 flex-shrink-0 text-amber-500 font-bold">!</span>
            <span>
              <strong>Panel ceiling artifact:</strong> Post-2018 MIC<sub>90</sub> is capped at
              32 mg/L by the instrument range. The true MIC<sub>90</sub> may be 64, 128 mg/L or higher.
            </span>
          </li>
          <li className="flex gap-3">
            <span className="mt-0.5 flex-shrink-0 text-amber-500 font-bold">!</span>
            <span>
              <strong><code className="text-xs bg-gray-100 px-1 rounded font-mono">is_censored</code> in SHAP plots</strong>{" "}
              is a data-structure variable, not biology. Censored isolates are structurally at the MIC floor.
            </span>
          </li>
          <li className="flex gap-3">
            <span className="mt-0.5 flex-shrink-0 text-amber-500 font-bold">!</span>
            <span>
              <strong>NDM underrepresented in training:</strong> NDM only became prevalent after 2018
              and is therefore underweighted in SHAP. Its importance will rise when models are retrained
              on post-2020 data.
            </span>
          </li>
        </ul>
      </Section>

      {/* API */}
      <Section title="API Endpoints">
        <div className="overflow-x-auto rounded-xl border border-gray-200">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-800 text-white">
                <Th>Endpoint</Th>
                <Th>Method</Th>
                <Th>Description</Th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100 text-gray-700">
              {[
                ["GET /health", "GET", "Service status, loaded models"],
                ["GET /api/trend/mic90", "GET", "MIC90 by year: actual + predicted + forecast. ?species=kpneumoniae"],
                ["GET /api/country-stats", "GET", "Resistance rate and MIC90 by country. ?species=kpneumoniae"],
                ["GET /api/features/importance", "GET", "Top 20 SHAP features. ?species=kpneumoniae"],
                ["POST /api/predict", "POST", "Single-isolate MIC prediction. Include species in body."],
                ["GET /api/countries", "GET", "Countries known to the model. ?species=kpneumoniae"],
              ].map(([ep, method, desc]) => (
                <tr key={ep} className="hover:bg-gray-50">
                  <td className="px-4 py-3">
                    <code className="text-xs bg-gray-900 text-red-400 px-2 py-0.5 rounded font-mono whitespace-nowrap">{ep}</code>
                  </td>
                  <td className="px-4 py-3 text-gray-500">{method}</td>
                  <td className="px-4 py-3">{desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-gray-400 mt-3">
          Full interactive docs at{" "}
          <a href="https://mic-creep-predict.onrender.com/docs" target="_blank" rel="noopener noreferrer"
            className="text-blue-500 hover:underline">
            mic-creep-predict.onrender.com/docs
          </a>{" "}
          (Swagger UI) and{" "}
          <a href="https://mic-creep-predict.onrender.com/redoc" target="_blank" rel="noopener noreferrer"
            className="text-blue-500 hover:underline">
            /redoc
          </a>.
        </p>
      </Section>

      <p className="text-xs text-gray-400 leading-relaxed mt-2">
        Data source: ATLAS (Pfizer) via Vivli AMR Register, controlled access.
        Model: XGBoost + Optuna, trained on ATLAS 2004-2018.
        Raw patient-level data is not redistributed per Vivli data-use agreement.
      </p>
    </main>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm mb-6">
      <h2 className="text-lg font-semibold text-gray-800 mb-4 pb-2 border-b border-gray-100">
        {title}
      </h2>
      {children}
    </section>
  );
}

function Th({ children }: { children: React.ReactNode }) {
  return (
    <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide">
      {children}
    </th>
  );
}
