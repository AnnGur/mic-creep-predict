"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
} from "recharts";

type CountryRow = {
  country: string;
  pct_resistant: number;
  pct_resistant_pred: number;
  mic90_actual: number;
  n: number;
};

type Props = { data: CountryRow[] };

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload as CountryRow;
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-3 shadow-lg text-sm min-w-[180px]">
      <p className="font-semibold text-gray-800 mb-1">{label}</p>
      <p className="text-red-600">Actual resistant: <strong>{d.pct_resistant.toFixed(1)}%</strong></p>
      <p className="text-green-700">Model predicted: <strong>{d.pct_resistant_pred.toFixed(1)}%</strong></p>
      <p className="text-gray-500 text-xs mt-1">MIC₉₀ (actual): {d.mic90_actual} mg/L</p>
      <p className="text-gray-500 text-xs">Isolates: {d.n.toLocaleString()}</p>
    </div>
  );
}

export default function CountryBarChart({ data }: Props) {
  const top20 = data.slice(0, 20);

  return (
    <ResponsiveContainer width="100%" height={520}>
      <BarChart
        data={top20}
        layout="vertical"
        margin={{ top: 4, right: 40, left: 110, bottom: 4 }}
      >
        <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e5e7eb" />
        <XAxis
          type="number"
          domain={[0, 70]}
          tickFormatter={(v) => `${v}%`}
          tick={{ fontSize: 11 }}
        />
        <YAxis
          type="category"
          dataKey="country"
          tick={{ fontSize: 11 }}
          width={105}
        />
        <Tooltip content={<CustomTooltip />} />
        <ReferenceLine x={50} stroke="#dc2626" strokeDasharray="4 3" />
        <Bar dataKey="pct_resistant" name="% Resistant (actual)" radius={[0, 3, 3, 0]}>
          {top20.map((entry) => (
            <Cell
              key={entry.country}
              fill={entry.pct_resistant >= 50 ? "#dc2626" : entry.pct_resistant >= 20 ? "#f97316" : "#2563eb"}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
