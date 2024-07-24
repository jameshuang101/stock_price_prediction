"use server";
// TODO: add date handlers

export interface HistoricalData {
  adjClose: number;
  change: number;
  changeOverTime: number;
  changePercent: number;
  close: number;
  date: string;
  high: number;
  label: string;
  low: number;
  open: number;
  unadjustedVolume: number;
  volume: number;
  vwap: number;
}

export interface HistoricalPrice {
  symbol: string;
  historical: HistoricalData[];
}

export async function useHistoricalPriceQuery({
  symbol,
  start,
  end,
}: {
  symbol: string;
  start?: string;
  end?: string;
}) {
  try {
    const url =
      start && end
        ? `https://financialmodelingprep.com/api/v3/historical-price-full/${symbol}?from=${start}&to=${end}`
        : `https://financialmodelingprep.com/api/v3/historical-price-full/${symbol}`;
    const res = await fetch(
      url + `?apikey=${process.env.FINANCIAL_MODELING_PREP_API_KEY}`
    );

    if (!res.ok) throw new Error("Failed to fetch data");

    return res.json();
  } catch (e) {
    console.error(e);
  }
}

export interface Company {
  symbol: string;
  name: string;
  currency: number;
  stockExchange: number;
  exchangeShortName: number;
}

export async function useCompanySymbolQuery({
  query,
  limit = 20,
  exchange = "NASDAQ",
}: {
  query: string;
  limit?: number;
  exchange?: string;
}) {
  try {
    const url =
      `https://financialmodelingprep.com/api/v3/search?query=${query}` +
      (!!limit ? `&limit=${limit}` : "") +
      (exchange.length ? `&exchange=${exchange}` : "");

    const res = await fetch(
      url + `&apikey=${process.env.FINANCIAL_MODELING_PREP_API_KEY}`
    );

    if (!res.ok) throw new Error("Failed to fetch data");

    return res.json() as Promise<Company[]>;
  } catch (e) {
    console.error(e);
  }
}
