"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

type ShapRow = {
  rank: number;
  feature: string;
  mean_abs_shap: number;
  note: string;
};

function categorise(feature: string): { label: string; colour: string } {
  if (feature === "is_censored" || feature === "pct_censored_year")
    return { label: "Data artifact", colour: "#9ca3af" };
  if (feature.startsWith("KPC") || feature.startsWith("NDM") || feature.startsWith("OXA") ||
      feature.startsWith("VIM") || feature.startsWith("IMP") || feature.startsWith("GES"))
    return { label: "Resistance gene", colour: "#dc2626" };
  if (feature.startsWith("ctry_"))
    return { label: "Country effect", colour: "#2563eb" };
  if (feature === "year")
    return { label: "Temporal (MIC creep)", colour: "#7c3aed" };
  if (feature.startsWith("spec_"))
    return { label: "Specimen type", colour: "#d97706" };
  if (feature.startsWith("age_") || feature.startsWith("gender_") || feature === "military_proxy")
    return { label: "Demographics", colour: "#16a34a" };
  return { label: "Other", colour: "#6b7280" };
}

function friendlyName(feature: string): string {
  if (feature === "is_censored") return "is_censored (artifact)";
  if (feature === "pct_censored_year") return "pct_censored_year (artifact)";
  if (feature.startsWith("ctry_")) return feature.replace("ctry_", "") + " (country)";
  if (feature.startsWith("spec_")) return feature.replace("spec_", "") + " specimen";
  if (feature === "age_paediatric") return "Age: paediatric";
  if (feature === "age_elderly") return "Age: elderly";
  if (feature === "gender_male") return "Sex: male";
  if (feature === "military_proxy") return "Military proxy";
  return feature;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload as ShapRow & { category: string; colour: string };
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-3 shadow-lg text-sm max-w-xs">
      <p className="font-semibold text-gray-800 mb-1">{d.feature}</p>
      <p style={{ color: d.colour }} className="text-xs font-medium mb-1">{d.category}</p>
      <p className="text-gray-700">Mean |SHAP|: <strong>{d.mean_abs_shap.toFixed(4)}</strong></p>
      {d.note && <p className="text-gray-500 text-xs mt-1 leading-relaxed">{d.note}</p>}
    </div>
  );
}

export default function ShapChart({ data }: { data: ShapRow[] }) {
  const rows = data.map((d) => {
    const { label, colour } = categorise(d.feature);
    return { ...d, displayName: friendlyName(d.feature), category: label, colour };
  });

  return (
    <ResponsiveContainer width="100%" height={520}>
      <BarChart
        data={rows}
        layout="vertical"
        margin={{ top: 4, right: 40, left: 160, bottom: 4 }}
      >
        <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e5e7eb" />
        <XAxis
          type="number"
          tickFormatter={(v) => v.toFixed(2)}
          tick={{ fontSize: 11 }}
          label={{ value: "Mean |SHAP value| (log₂ MIC units)", position: "insideBottom", offset: -2, fontSize: 11 }}
        />
        <YAxis
          type="category"
          dataKey="displayName"
          tick={{ fontSize: 11 }}
          width={155}
        />
        <Tooltip content={<CustomTooltip />} />
        <Bar dataKey="mean_abs_shap" radius={[0, 3, 3, 0]}>
          {rows.map((r) => (
            <Cell key={r.feature} fill={r.colour} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
