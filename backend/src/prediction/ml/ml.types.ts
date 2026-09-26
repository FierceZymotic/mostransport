export interface PredictionRequest {
  request_id: string;
  prediction_time: string;

  vehicle_context: {
    unit_id: string;
    tr_id: string;
    route_id: string;
  };

  schedule_context: {
    target_action_id: string;
    target_time_begin: string;
    target_lat: number;
    target_lon: number;
    current_deviation_seconds: number;
    manual_fill: boolean;
  };

  telemetry: PredictionTelemetry[];
}

export interface PredictionTelemetry {
  event_time: string;
  lat: number | null;
  lon: number | null;
  location_valid: boolean;
  speed: number;
}

export interface PredictionResponse {
  request_id: string;
  status: 'success';

  prediction: {
    delay_seconds: number;
    target_time: string;
    reason: string | null;
  };

  generated_at: string;
  model_version: string;
  feature_schema_version: string;
}