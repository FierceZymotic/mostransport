export interface VehicleState {
  unitId: number;

  timestamp: number;

  longitude: number;
  latitude: number;
  locationValid: boolean;

  speed: number;
  speedMax: number;

  course: number;
  track: number;

  altitude: number;
  nsat: number;
  pdop: number;
}