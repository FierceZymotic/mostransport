export type RiskLevel = "low" | "medium" | "high";

/** Canonical risk of the latest prediction (see dashboard/state.ts); "unknown" = no prediction yet. */
export type RiskState = RiskLevel | "unknown";

/** Map-facing vehicle view: marker position, canonical risk and the marker title route text. */
export interface Vehicle {
  id: string;
  route: string;
  lat: number;
  lon: number;
  risk: RiskState;
}
