import {
  Card,
  CardHeader,
  CardDescription,
  CardTitle,
  CardContent,
  CardFooter,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

export default function MetricsCard({
  title,
  amount,
  difference,
  value,
}: {
  title: string;
  amount: number;
  difference: number;
  value: number;
}) {
  // TODO: format the value to be nicer
  return (
    <Card x-chunk="dashboard-05-chunk-2">
      <CardHeader className="pb-2">
        <CardDescription>{title}</CardDescription>
        <CardTitle className="text-4xl">${amount}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="text-xs text-muted-foreground">
          {difference >= 0 ? "+" : "-"}
          {difference}% from last month
        </div>
      </CardContent>
      <CardFooter>
        <Progress value={value} aria-label="12% increase" />
      </CardFooter>
    </Card>
  );
}
