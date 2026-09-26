export type RiskLevel = "low" | "medium" | "high";

export interface Vehicle {
  id: string;
  route: string;
  lat: number;
  lon: number;
  speed: number;
  delayMinutes: number;
  risk: RiskLevel;
  reason: string;
  segment: string;
}