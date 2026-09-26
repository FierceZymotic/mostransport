export interface TelemetryEvent {
  unitId: number;
  eventTime: Date;

  location: {
    latitude: number;
    longitude: number;
    valid: boolean;
  };

  speed: number;
  heading: number;
  altitude: number;
}