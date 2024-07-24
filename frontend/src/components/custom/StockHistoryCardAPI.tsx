"use client";

import { useHistoricalPriceQuery, HistoricalPrice } from "@/utils/api";
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
import { Card, CardContent, CardHeader } from "../ui/card";

export default function StockHistoryCard({ symbol }: { symbol?: string }) {
  const [data, setData] = useState<HistoricalPrice | null>(null);

  useEffect(() => {
    if (!symbol) setData(null);
    const fetchData = async () => {
      // eslint-disable-next-line react-hooks/rules-of-hooks
      const res = await useHistoricalPriceQuery({ symbol: symbol! });
      setData(res);
    };
    fetchData();
  }, [symbol]);

  const minVal = data?.historical.reduce((x, acc) => {
    return acc.low < x ? acc.low : x;
  }, Infinity);

  return (
    <Card>
      <CardHeader className="text-2xl font-bold">Historical Price</CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart
            // height={400}
            // width={1200}
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
      </CardContent>
    </Card>
  );
}
