"use client";
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
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../ui/card";

interface OHLCV {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface StockExported {
  name: string;
  symbol: string;
  value: OHLCV[];
}

export default function StockOHLCVChartCard({
  stock,
}: {
  stock: StockExported;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-xl">{stock.symbol}</CardTitle>
        <CardDescription>{stock.name}</CardDescription>
        <CardContent>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart
              data={stock.value}
              margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
            >
              <CartesianGrid />
              <XAxis dataKey="date" color="white" />
              <YAxis yAxisId={0} color="white" />
              <Tooltip />
              <Line
                type="monotone"
                dot={false}
                dataKey="open"
                stroke="green"
                yAxisId={0}
              />
              <ReferenceLine y={0} stroke="#000" />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </CardHeader>
    </Card>
  );
}
