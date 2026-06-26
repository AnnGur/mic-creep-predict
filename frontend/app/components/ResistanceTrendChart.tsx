"use client";

import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

type TrendRow = {
  year: number;
  pct_resistant?: number;
  pct_resistant_prob?: number;
  pct_resistant_pred?: number;
  source: string;
};

type Props = {
  data: TrendRow[];
  auc?: number;
};

const COLOURS = {
  actual:    "#2563eb",
  predicted: "#16a34a",
  split:     "#6b7280",
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-3 shadow-lg text-sm">
      <p className="font-semibold text-gray-800 mb-1">{label}</p>
      {payload.map((p: { name: string; value: number; color: string }) => (
        <p key={p.name} style={{ color: p.color }}>
          {p.name}: <strong>{Number(p.value).toFixed(1)}%</strong>
        </p>
      ))}
    </div>
  );
}

export default function ResistanceTrendChart({ data, auc }: Props) {
  const rows = data
    .filter((d) => d.source !== "forecast_extrapolated")
    .map((d) => ({
      year:      d.year,
      actual:    d.pct_resistant ?? null,
      // prefer P(R) classifier probability; fall back to regression-threshold prediction
      predicted: d.pct_resistant_prob ?? d.pct_resistant_pred ?? null,
    }));

  const aucLabel = auc != null ? ` (AUC=${auc.toFixed(3)})` : "";

  return (
    <ResponsiveContainer width="100%" height={340}>
      <ComposedChart data={rows} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis
          dataKey="year"
          tick={{ fontSize: 12 }}
          label={{ value: "Year", position: "insideBottom", offset: -2, fontSize: 12 }}
        />
        <YAxis
          tickFormatter={(v) => `${v}%`}
          tick={{ fontSize: 12 }}
          domain={[0, 100]}
          label={{
            value: "% Resistant isolates",
            angle: -90,
            position: "insideLeft",
            offset: 14,
            fontSize: 12,
          }}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />

        <ReferenceLine
          x={2019}
          stroke={COLOURS.split}
          strokeDasharray="4 4"
          label={{ value: "Test ->", position: "insideTopLeft", fill: COLOURS.split, fontSize: 11 }}
        />

        <Line
          dataKey="actual"
          name="Actual % resistant"
          stroke={COLOURS.actual}
          strokeWidth={2.5}
          dot={{ r: 3 }}
          connectNulls
        />
        <Line
          dataKey="predicted"
          name={`P(R) classifier${aucLabel}`}
          stroke={COLOURS.predicted}
          strokeWidth={2}
          strokeDasharray="5 3"
          dot={{ r: 3 }}
          connectNulls
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
