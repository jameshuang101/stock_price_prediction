import Image from "next/image";
import Link from "next/link";
import {
  Home,
  Package,
  Package2,
  PanelLeft,
  LineChart as LineChartIcon,
  ShoppingCart,
  Users2,
} from "lucide-react";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { LazyFormInput } from "@/components/custom/LazyFormInput";
import appl from "../../../../artifacts/datasets/AAPL/AAPL_20y.json";
import goog from "../../../../artifacts/datasets/GOOG/GOOG_20y.json";
import msft from "../../../../artifacts/datasets/MSFT/MSFT_20y.json";
import StockHistoryCard from "@/components/custom/StockHistoryCardAPI";
import StockOHLCVChartCard from "@/components/custom/StockOHLCVChartCard";
import MemoizedStockOHLCVChartCard from "@/components/custom/StockOHLCVChartCard";

export default function AnalyticsPage() {
  interface OHLCV {
    date: number;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
  }

  function normalizeJSON(jsonData: any) {
    return Object.entries(jsonData).map(([epoch, data]: [string, any]) => ({
      date: new Date(parseInt(epoch) * 1000).toISOString().split("T")[0],
      open: data.Open as number,
      high: data.High as number,
      low: data.Low as number,
      close: data.Close as number,
      volume: data.Volume as number,
    }));
  }

  const data = [
    {
      name: "Apple",
      symbol: "AAPL",
      value: normalizeJSON(appl),
    },
    {
      name: "Google",
      symbol: "GOOG",
      value: normalizeJSON(goog),
    },
    {
      name: "Microsoft",
      symbol: "MSFT",
      value: normalizeJSON(msft),
    },
  ];

  return (
    <>
      <header className="sticky top-0 z-30 flex items-center gap-4 px-4 border-b h-14 bg-background sm:static sm:h-auto sm:border-0 sm:bg-transparent sm:px-6">
        <Sheet>
          <SheetTrigger asChild>
            <Button size="icon" variant="outline" className="sm:hidden">
              <PanelLeft className="w-5 h-5" />
              <span className="sr-only">Toggle Menu</span>
            </Button>
          </SheetTrigger>
          <SheetContent side="left" className="sm:max-w-xs">
            <nav className="grid gap-6 text-lg font-medium">
              <Link
                href="#"
                className="flex items-center justify-center w-10 h-10 gap-2 text-lg font-semibold rounded-full group shrink-0 bg-primary text-primary-foreground md:text-base"
              >
                <Package2 className="w-5 h-5 transition-all group-hover:scale-110" />
                <span className="sr-only">Acme Inc</span>
              </Link>
              <Link
                href="#"
                className="flex items-center gap-4 px-2.5 text-muted-foreground hover:text-foreground"
              >
                <Home className="w-5 h-5" />
                Dashboard
              </Link>
              <Link
                href="#"
                className="flex items-center gap-4 px-2.5 text-foreground"
              >
                <ShoppingCart className="w-5 h-5" />
                Orders
              </Link>
              <Link
                href="#"
                className="flex items-center gap-4 px-2.5 text-muted-foreground hover:text-foreground"
              >
                <Package className="w-5 h-5" />
                Products
              </Link>
              <Link
                href="#"
                className="flex items-center gap-4 px-2.5 text-muted-foreground hover:text-foreground"
              >
                <Users2 className="w-5 h-5" />
                Customers
              </Link>
              <Link
                href="#"
                className="flex items-center gap-4 px-2.5 text-muted-foreground hover:text-foreground"
              >
                <LineChartIcon className="w-5 h-5" />
                Settings
              </Link>
            </nav>
          </SheetContent>
        </Sheet>
        <Breadcrumb className="hidden md:flex">
          <BreadcrumbList>
            <BreadcrumbItem>
              <BreadcrumbLink asChild>
                <Link href="#">Dashboard</Link>
              </BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator />
            <BreadcrumbItem>
              <BreadcrumbPage>Analytics</BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
        <div className="relative flex-1 ml-auto md:grow-0">
          {/* <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" /> */}
          <LazyFormInput className="w-full rounded-lg bg-background pl-8 md:w-[200px] lg:w-[336px]" />
        </div>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="outline"
              size="icon"
              className="overflow-hidden rounded-full"
            >
              <Image
                src="/images/placeholder.jpg"
                width={36}
                height={36}
                alt="Avatar"
                className="overflow-hidden rounded-full"
              />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuLabel>My Account</DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem>Settings</DropdownMenuItem>
            <DropdownMenuItem>Support</DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem>Logout</DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </header>
      <main className="grid items-start flex-1 gap-4 p-4 sm:px-6 sm:py-0 md:gap-8 lg:grid-cols-2">
        <div className="grid items-start gap-4 auto-rows-max md:gap-8 lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle className="text-4xl">Overview</CardTitle>
              <CardDescription>
                idk maybe add some stock overview info here
              </CardDescription>
            </CardHeader>
          </Card>
          <div className="grid gap-4 grid-cols-1 sm:grid-cols-2 ">
            <StockHistoryCard symbol="AAPL" />
            {data.map((stock) => (
              <MemoizedStockOHLCVChartCard key={stock.symbol} stock={stock} />
            ))}
          </div>
        </div>
      </main>
    </>
  );
}
