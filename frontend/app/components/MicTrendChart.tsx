"use client";

import {
  ComposedChart,
  Line,
  Area,
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
  actual_mic90?: number;
  predicted_mic90?: number;
  forecast_mic90?: number;
  pct_resistant?: number;
  source: string;
};

type Props = {
  data: TrendRow[];
  eucastR: number;
};

const COLOURS = {
  actual: "#2563eb",
  predicted: "#16a34a",
  forecast: "#d97706",
  breakpoint: "#dc2626",
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-3 shadow-lg text-sm">
      <p className="font-semibold text-gray-800 mb-1">{label}</p>
      {payload.map((p: { name: string; value: number; color: string }) => (
        <p key={p.name} style={{ color: p.color }}>
          {p.name}: <strong>{Number(p.value).toFixed(2)} mg/L</strong>
        </p>
      ))}
    </div>
  );
}

export default function MicTrendChart({ data, eucastR }: Props) {
  // Flatten into one row per year for ComposedChart
  const rows = data.map((d) => ({
    year: d.year,
    actual: d.actual_mic90 ?? null,
    predicted: d.predicted_mic90 ?? null,
    forecast: d.forecast_mic90 ?? null,
  }));

  return (
    <ResponsiveContainer width="100%" height={400}>
      <ComposedChart data={rows} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis
          dataKey="year"
          tick={{ fontSize: 12 }}
          label={{ value: "Year", position: "insideBottom", offset: -2, fontSize: 12 }}
        />
        <YAxis
          tickFormatter={(v) => `${v}`}
          tick={{ fontSize: 12 }}
          label={{
            value: "MIC₉₀ (mg/L)",
            angle: -90,
            position: "insideLeft",
            offset: 12,
            fontSize: 12,
          }}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />

        {/* EUCAST resistance breakpoint */}
        <ReferenceLine
          y={eucastR}
          stroke={COLOURS.breakpoint}
          strokeDasharray="6 3"
          label={{
            value: `EUCAST R ≥ ${eucastR} mg/L`,
            position: "insideTopRight",
            fill: COLOURS.breakpoint,
            fontSize: 11,
          }}
        />

        {/* Train/test split */}
        <ReferenceLine
          x={2019}
          stroke="#6b7280"
          strokeDasharray="4 4"
          label={{ value: "Test →", position: "insideTopLeft", fill: "#6b7280", fontSize: 11 }}
        />

        <Line
          dataKey="actual"
          name="Actual MIC₉₀"
          stroke={COLOURS.actual}
          strokeWidth={2.5}
          dot={{ r: 3 }}
          connectNulls
        />
        <Line
          dataKey="predicted"
          name="Model predicted MIC₉₀"
          stroke={COLOURS.predicted}
          strokeWidth={2}
          strokeDasharray="5 3"
          dot={{ r: 3 }}
          connectNulls
        />
        <Area
          dataKey="forecast"
          name="Forecast MIC₉₀ (2023–2026)"
          stroke={COLOURS.forecast}
          fill={COLOURS.forecast}
          fillOpacity={0.12}
          strokeWidth={2}
          strokeDasharray="3 3"
          connectNulls
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
