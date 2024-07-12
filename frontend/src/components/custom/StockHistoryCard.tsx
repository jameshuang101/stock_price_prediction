"ues client";

import { getHistoricalPrice, HistoricalPrice } from "@/utils/api";
import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export default async function StockHistoryCard({
  symbol,
}: {
  symbol?: string;
}) {
  const [data, setData] = useState<HistoricalPrice | null>(null);

  useEffect(() => {
    if (!symbol) setData(null);
    const fetchData = async () => {
      const res = await getHistoricalPrice({ symbol: symbol! });
      setData(res);
    };
    fetchData();
  }, [symbol]);

  const minVal = data?.historical.reduce((x, acc) => {
    return acc.low < x ? acc.low : x;
  }, Infinity);

  return (
    <div className="mx-auto container my-2 font-sans">
      <ResponsiveContainer width="100%" height={400} className="bg-slate-500">
        <LineChart
          height={400}
          width={1200}
          data={data?.historical}
          margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" color="white" />
          <YAxis yAxisId={0} color="white" />
          <Tooltip />
          {/* <CartesianGrid stroke="#f5f5f5" /> */}
          <Line
            type="monotone"
            dot={false}
            dataKey="high"
            stroke="green"
            yAxisId={0}
          />
          <Line
            type="monotone"
            dot={false}
            dataKey="low"
            stroke="red"
            yAxisId={0}
          />
          <ReferenceLine
            y={minVal}
            label={`$${minVal}, lowest price`}
            stroke="red"
            strokeDasharray="3 3"
          />
        </LineChart>
      </ResponsiveContainer>
      <div className="flex flex-col w-full rounded-xl border bg-white">
        <h1 className="text-2xl font-bold p-2">Historical Price</h1>
        <div className="flex flex-col h-screen overflow-auto">
          {data?.historical.map((item) => (
            <pre key={item.date} className="px-2">
              {JSON.stringify(item, null, 2)}
            </pre>
          ))}
        </div>
      </div>
    </div>
  );
}
