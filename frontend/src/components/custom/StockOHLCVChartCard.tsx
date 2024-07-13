"use client";
import {
  CartesianGrid,
  Legend,
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
import { useState } from "react";
import { capitalize } from "lodash";
import { format } from "date-fns";

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
  const [showOpen, setShowOpen] = useState(true);
  const [showHigh, setShowHigh] = useState(true);
  const [showLow, setShowLow] = useState(true);
  const [showClose, setShowClose] = useState(true);
  const [showVolume, setShowVolume] = useState(true);

  const tooltipFormatter = (value: any, name: string, props: any) => {
    return [parseFloat(value).toFixed(2), capitalize(name)];
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-xl">{stock.symbol}</CardTitle>
        <CardDescription>{stock.name}</CardDescription>
        <CardContent>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart
              data={stock.value}
              margin={{ top: 5, right: 5, left: 5, bottom: 5 }}
            >
              <CartesianGrid />
              <XAxis dataKey="date" color="white" tickCount={20} />
              <YAxis yAxisId={0} color="white" />
              <Tooltip
                formatter={tooltipFormatter}
                labelFormatter={(label) => format(label, "MM-dd-yyyy")}
              />
              <Line
                type="monotone"
                dot={false}
                dataKey="open"
                stroke="orange"
                yAxisId={0}
              />
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
              <Line
                type="monotone"
                dot={false}
                dataKey="close"
                stroke="purple"
                yAxisId={0}
              />
              <ReferenceLine y={0} stroke="#000" />
              <Legend
                iconType="square"
                onClick={(data, index, e) => console.log(data, index, e)}
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </CardHeader>
    </Card>
  );
}
