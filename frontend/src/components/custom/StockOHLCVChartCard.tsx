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
import { memo, useState } from "react";
import { capitalCase } from "change-case";
import { format } from "date-fns";
import { Button } from "../ui/button";

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

function StockOHLCVChartCard({ stock }: { stock: StockExported }) {
  const [showOpen, setShowOpen] = useState(true);
  const [showHigh, setShowHigh] = useState(true);
  const [showLow, setShowLow] = useState(true);
  const [showClose, setShowClose] = useState(true);
  const [showVolume, setShowVolume] = useState(true);

  const tooltipFormatter = (value: any, name: string, props: any) => {
    return [parseFloat(value).toFixed(2), capitalCase(name)];
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-xl">{stock.symbol}</CardTitle>
        <CardDescription>{stock.name}</CardDescription>
        <CardContent className="p-0">
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={stock.value}>
              <CartesianGrid />
              <XAxis dataKey="date" color="white" tickCount={20} />
              <YAxis yAxisId={0} color="white" />
              <Tooltip
                formatter={tooltipFormatter}
                labelFormatter={(label) => format(label, "MMM dd, yyyy")}
              />
              <Line
                type="monotone"
                dot={false}
                dataKey="open"
                stroke="#f97316"
                yAxisId={0}
                hide={!showOpen}
              />
              <Line
                type="monotone"
                dot={false}
                dataKey="high"
                stroke="#22c55e"
                hide={!showHigh}
                yAxisId={0}
              />
              <Line
                type="monotone"
                dot={false}
                dataKey="low"
                stroke="#ef4444"
                hide={!showLow}
                yAxisId={0}
              />
              <Line
                type="monotone"
                dot={false}
                dataKey="close"
                stroke="#8b5cf6"
                hide={!showClose}
                yAxisId={0}
              />
              <Legend
                iconType="square"
                formatter={(value) => {
                  return capitalCase(value);
                }}
                content={(props) => (
                  <div className="flex flex-row space-x-2 justify-center w-full">
                    <Button
                      className={`rounded-lg px-2 py-1 h-fit hover:bg-gray-300/80 ${
                        !showOpen ? "bg-secondary" : "bg-gray-300"
                      }`}
                      variant="secondary"
                      onClick={() => setShowOpen(!showOpen)}
                    >
                      <div className="flex-row flex space-x-2 w-fit whitespace-nowrap items-center">
                        <div
                          className={`w-3 h-3 rounded-sm ${
                            showOpen ? "bg-orange-500" : "bg-gray-400"
                          }`}
                        />
                        <p
                          className={`${
                            showOpen ? undefined : "text-gray-700"
                          }`}
                        >
                          Open
                        </p>
                      </div>
                    </Button>
                    <Button
                      className={`rounded-lg px-2 py-1 h-fit hover:bg-gray-300/80 ${
                        !showHigh ? "bg-secondary" : "bg-gray-300"
                      }`}
                      variant="secondary"
                      onClick={() => setShowHigh(!showHigh)}
                    >
                      <div className="flex-row flex space-x-2 w-fit whitespace-nowrap items-center">
                        <div
                          className={`w-3 h-3 rounded-sm ${
                            showHigh ? "bg-green-500" : "bg-gray-400"
                          }`}
                        />
                        <p
                          className={`${
                            showHigh ? undefined : "text-gray-700"
                          }`}
                        >
                          High
                        </p>
                      </div>
                    </Button>
                    <Button
                      className={`rounded-lg px-2 py-1 h-fit hover:bg-gray-300/80 ${
                        !showLow ? "bg-secondary" : "bg-gray-300"
                      }`}
                      variant="secondary"
                      onClick={() => setShowLow(!showLow)}
                    >
                      <div className="flex-row flex space-x-2 w-fit whitespace-nowrap items-center">
                        <div
                          className={`w-3 h-3 rounded-sm ${
                            showLow ? "bg-red-500" : "bg-gray-400"
                          }`}
                        />
                        <p
                          className={`${showLow ? undefined : "text-gray-700"}`}
                        >
                          Low
                        </p>
                      </div>
                    </Button>
                    <Button
                      className={`rounded-lg px-2 py-1 h-fit hover:bg-gray-300/80 ${
                        !showClose ? "bg-secondary" : "bg-gray-300"
                      }`}
                      variant="secondary"
                      onClick={() => setShowClose(!showClose)}
                    >
                      <div className="flex-row flex space-x-2 w-fit whitespace-nowrap items-center">
                        <div
                          className={`w-3 h-3 rounded-sm ${
                            showClose ? "bg-violet-500" : "bg-gray-400"
                          }`}
                        />
                        <p
                          className={`${
                            showClose ? undefined : "text-gray-700"
                          }`}
                        >
                          Close
                        </p>
                      </div>
                    </Button>
                  </div>
                )}
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </CardHeader>
    </Card>
  );
}

const MemoizedStockOHLCVChartCard = memo(StockOHLCVChartCard);
export default MemoizedStockOHLCVChartCard;
