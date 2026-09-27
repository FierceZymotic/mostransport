// Realtime dashboard state: pure functions only (no React), shared by the map, the vehicle
// list and the selected-vehicle card so that all three show the same values.
//
// Per unit (key = String(unitId)) two independently owned parts are kept:
//   telemetry  <- WebSocket "telemetry" events (and the initial /dashboard/summary seed)
//   prediction <- /prediction/dashboard/alerts rows (prediction history), refreshed whenever a
//                 WebSocket "prediction" event arrives
// A telemetry update never touches the prediction and vice versa.
import type { RiskLevel, RiskState, Vehicle } from "../types";

// ---------------------------------------------------------------- Backend payloads (read-only)

/** WS `{event: "telemetry", payload}` (backend TelemetryStreamService.publish). */
export interface TelemetryPayload {
  unitId: number | string;
  timestamp?: number;
  latitude?: number;
  longitude?: number;
  locationValid?: boolean;
  speed?: number;
}

/** One row of GET /prediction/dashboard/alerts (prediction history, all units). */
export interface AlertRow {
  id: string;
  unitId: string;
  routeId: string;
  status: string;
  delaySeconds: number;
  reason: string;
  generatedAt: string;
}

/**
 * WS `{event: "prediction", payload, meta}`. The payload is the ML response: it carries no
 * unit or trip id, only `request_id` (= alerts `id`), so it is attributed to a unit through
 * the alerts row the Backend persisted before publishing it.
 */
export interface PredictionEventPayload {
  request_id?: string;
  prediction?: { delay_seconds?: number; target_time?: string };
  generated_at?: string;
  model_version?: string;
}

/** Only the telemetry-owned fields of GET /prediction/dashboard/summary rows are used. */
export interface SummaryVehicle {
  id: string;
  lat?: number | null;
  lon?: number | null;
  speed?: number | null;
}

// ---------------------------------------------------------------- state

export interface TelemetryState {
  lat: number | null;
  lon: number | null;
  speedKmh: number | null;
  /** Client time (ms) the last live telemetry event was received; null = only the summary seed. */
  receivedAt: number | null;
}

export interface PredictionState {
  requestId: string;
  delaySeconds: number;
  /** ML generation time (UTC wall clock), ms since epoch. */
  generatedAt: number;
  routeId: string | null;
  reason: string | null;
  targetTime: string | null;
  deviationStatus: string | null;
}

export interface UnitState {
  unitId: string;
  telemetry: TelemetryState | null;
  prediction: PredictionState | null;
}

export type DashboardState = Readonly<Record<string, UnitState>>;

/** Details only the WS prediction event carries, kept by request_id until its alerts row is known. */
export interface PredictionEventDetails {
  targetTime: string | null;
  deviationStatus: string | null;
}

export function unitKey(unitId: unknown): string {
  return String(unitId ?? "").trim();
}

function finite(value: unknown): number | null {
  const n = typeof value === "number" ? value : typeof value === "string" && value.trim() !== "" ? Number(value) : NaN;
  return Number.isFinite(n) ? n : null;
}

function withUnit(state: DashboardState, key: string, update: (unit: UnitState) => UnitState): DashboardState {
  const current = state[key] ?? { unitId: key, telemetry: null, prediction: null };
  return { ...state, [key]: update(current) };
}

// ---------------------------------------------------------------- speed

/**
 * Speed is km/h end to end: organizer NDTP spec §6.1 defines G6CellNav00 `speedAvg` as u16 km/h,
 * the Backend parser forwards it unscaled (VehicleState.speed = speedAvg), and the WS payload and
 * vehicle_last_state carry that value. So the display is the identity (no ×3.6, no ÷10).
 */
export function normalizeSpeedKmh(raw: unknown): number | null {
  const value = finite(raw);
  return value === null || value < 0 ? null : value;
}

export function formatSpeed(speedKmh: number | null): string {
  return speedKmh === null ? "нет данных" : `${Math.round(speedKmh)} км/ч`;
}

// ---------------------------------------------------------------- telemetry ownership

export function applyTelemetry(state: DashboardState, payload: TelemetryPayload, receivedAt: number): DashboardState {
  const key = unitKey(payload.unitId);
  if (!key) return state;
  const lat = finite(payload.latitude);
  const lon = finite(payload.longitude);
  // Only a valid fix moves the marker; an invalid packet still updates speed and freshness.
  const validFix = payload.locationValid !== false && lat !== null && lon !== null;
  return withUnit(state, key, (unit) => ({
    ...unit,
    telemetry: {
      lat: validFix ? lat : (unit.telemetry?.lat ?? null),
      lon: validFix ? lon : (unit.telemetry?.lon ?? null),
      speedKmh: normalizeSpeedKmh(payload.speed) ?? unit.telemetry?.speedKmh ?? null,
      receivedAt,
    },
  }));
}

/** Initial positions from /dashboard/summary; never overrides live telemetry already received. */
export function seedFromSummary(state: DashboardState, vehicles: SummaryVehicle[]): DashboardState {
  let next = state;
  for (const vehicle of vehicles) {
    const key = unitKey(vehicle.id);
    if (!key || next[key]?.telemetry?.receivedAt) continue;
    const lat = finite(vehicle.lat);
    const lon = finite(vehicle.lon);
    next = withUnit(next, key, (unit) => ({
      ...unit,
      telemetry: {
        lat: lat !== null && lon !== null ? lat : null,
        lon: lat !== null && lon !== null ? lon : null,
        speedKmh: normalizeSpeedKmh(vehicle.speed),
        receivedAt: null,
      },
    }));
  }
  return next;
}

// ---------------------------------------------------------------- prediction ownership

/** Deterministic order: newer ML generation time wins; equal times fall back to the request id. */
export function isNewerPrediction(candidate: PredictionState, current: PredictionState | null): boolean {
  if (!current) return true;
  if (candidate.generatedAt !== current.generatedAt) return candidate.generatedAt > current.generatedAt;
  return candidate.requestId > current.requestId;
}

export function predictionFromAlert(row: AlertRow, details?: PredictionEventDetails): PredictionState | null {
  const generatedAt = Date.parse(row?.generatedAt ?? "");
  const delaySeconds = finite(row?.delaySeconds);
  if (!row?.id || !unitKey(row.unitId) || !Number.isFinite(generatedAt) || delaySeconds === null) return null;
  const routeId = String(row.routeId ?? "").trim();
  return {
    requestId: String(row.id),
    delaySeconds,
    generatedAt,
    // The alerts API substitutes 'unknown' for a missing trip id.
    routeId: routeId && routeId !== "unknown" ? routeId : null,
    // ... and 'prediction' for a missing reason.
    reason: row.reason && row.reason !== "prediction" ? row.reason : null,
    targetTime: details?.targetTime ?? null,
    deviationStatus: details?.deviationStatus ?? null,
  };
}

/** Merge prediction history (any order) into state: per unit only a newer prediction replaces. */
export function mergeAlerts(
  state: DashboardState,
  rows: AlertRow[],
  details: Readonly<Record<string, PredictionEventDetails>> = {},
): DashboardState {
  let next = state;
  for (const row of rows) {
    const candidate = predictionFromAlert(row, details[String(row?.id)]);
    if (!candidate) continue;
    const key = unitKey(row.unitId);
    const current = next[key]?.prediction ?? null;
    // Same prediction seen again: only attach WS-only details that were not known before.
    const sameWithNewDetails = current !== null && current.requestId === candidate.requestId
      && ((candidate.targetTime !== null && current.targetTime === null) || (candidate.deviationStatus !== null && current.deviationStatus === null));
    if (!isNewerPrediction(candidate, current) && !sameWithNewDetails) continue;
    next = withUnit(next, key, (unit) => ({ ...unit, prediction: candidate }));
  }
  return next;
}

/** Latest prediction per unit from a history array, independent of row order. */
export function latestPredictionByUnitId(rows: AlertRow[]): Record<string, PredictionState> {
  const merged = mergeAlerts({}, rows);
  return Object.fromEntries(Object.entries(merged).map(([key, unit]) => [key, unit.prediction as PredictionState]));
}

export function predictionEventDetails(payload: PredictionEventPayload, meta: unknown): PredictionEventDetails {
  const status = meta && typeof meta === "object" ? (meta as { current_deviation_status?: unknown }).current_deviation_status : undefined;
  return {
    targetTime: typeof payload?.prediction?.target_time === "string" ? payload.prediction.target_time : null,
    deviationStatus: typeof status === "string" ? status : null,
  };
}

// ---------------------------------------------------------------- display helpers

const MINUS = "−";

/** Signed delay in seconds, rounded to whole seconds: 10.9 -> "+11 сек", 3661 -> "+1 ч 1 мин 1 сек". */
export function formatDelay(delaySeconds: number): string {
  const total = Math.round(Math.abs(delaySeconds));
  if (total === 0) return "0 сек";
  const sign = delaySeconds < 0 ? MINUS : "+";
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const parts: string[] = [];
  if (h) parts.push(`${h} ч`);
  if (m) parts.push(`${m} мин`);
  if (s) parts.push(`${s} сек`);
  return `${sign}${parts.join(" ")}`;
}

export interface ReasonDisplay {
  kind: "degraded" | "none" | "unknown";
  text: string;
  raw: string | null;
}

/**
 * Dispatcher-facing text for the stored prediction reason. `degraded:current_deviation_unavailable`
 * marks a SUCCESSFUL prediction made without schedule facts; the raw value is kept for diagnostics.
 */
export function displayReason(raw: string | null): ReasonDisplay {
  if (!raw) return { kind: "none", text: "Особых причин не указано", raw: null };
  if (raw.startsWith("degraded:current_deviation_unavailable")) {
    return { kind: "degraded", text: "Недостаточно фактических данных по графику", raw };
  }
  return { kind: "unknown", text: "Причина не расшифрована", raw };
}

export interface ReasonPresentation {
  text: string;
  /** Raw backend reason, for a diagnostics tooltip only. */
  tooltip: string | undefined;
  /** "warning" = successful prediction with reduced inputs (degraded), never an error state. */
  tone: "warning" | "neutral";
}

/** What the card renders for a prediction reason: the mapped text, never the raw backend string. */
export function reasonPresentation(reason: ReasonDisplay): ReasonPresentation {
  return { text: reason.text, tooltip: reason.raw ?? undefined, tone: reason.kind === "degraded" ? "warning" : "neutral" };
}

/**
 * Canonical risk = organizer delay classes (dataset README §3, target_class: early / ontime / late,
 * thresholds -60 s / +120 s) applied to the latest predicted delay. late -> high, early -> medium,
 * on time -> low; a unit without a prediction has no risk level (never shown as "low").
 */
export const ORGANIZER_EARLY_THRESHOLD_S = -60;
export const ORGANIZER_LATE_THRESHOLD_S = 120;

export interface RiskView {
  level: RiskState;
  label: string;
  status: string;
}

const RISK_LABEL: Record<RiskState, string> = {
  low: "Низкий риск",
  medium: "Средний риск",
  high: "Высокий риск",
  unknown: "Нет прогноза",
};

export function riskLevelForDelay(delaySeconds: number): RiskLevel {
  if (delaySeconds > ORGANIZER_LATE_THRESHOLD_S) return "high";
  if (delaySeconds < ORGANIZER_EARLY_THRESHOLD_S) return "medium";
  return "low";
}

export function riskOf(prediction: PredictionState | null): RiskView {
  if (!prediction) return { level: "unknown", label: RISK_LABEL.unknown, status: "Прогноз: нет данных" };
  const level = riskLevelForDelay(prediction.delaySeconds);
  const status = level === "high"
    ? `Прогноз задержки ${formatDelay(prediction.delaySeconds)}`
    : level === "medium"
      ? `Прогноз опережения ${formatDelay(prediction.delaySeconds)}`
      : "По графику";
  return { level, label: RISK_LABEL[level], status };
}

export const SECTION_UNAVAILABLE = "данные недоступны";

/** The Backend provides no segment / stop interval, so none is shown (never derived from ids or geometry). */
export function sectionLabel(_unit: UnitState): string {
  return SECTION_UNAVAILABLE;
}

export function routeLabel(unit: UnitState): string {
  const routeId = unit.prediction?.routeId;
  return routeId ? `маршрут ${routeId}` : "маршрут недоступен";
}

export function formatAge(fromMs: number | null, nowMs: number): string {
  if (fromMs === null) return "нет данных";
  const seconds = Math.max(0, Math.round((nowMs - fromMs) / 1000));
  if (seconds < 60) return `${seconds} сек назад`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)} мин назад`;
  return `${Math.floor(seconds / 3600)} ч назад`;
}

// ---------------------------------------------------------------- view model

export interface VehicleView {
  id: string;
  /** Header route text; also used by the map marker title and the search. */
  route: string;
  lat: number;
  lon: number;
  hasPosition: boolean;
  risk: RiskState;
  riskLabel: string;
  status: string;
  delayText: string;
  speedText: string;
  sectionText: string;
  reason: ReasonDisplay | null;
  telemetryReceivedAt: number | null;
  telemetrySeedOnly: boolean;
  predictionGeneratedAt: number | null;
  prediction: PredictionState | null;
}

export function toVehicleView(unit: UnitState): VehicleView {
  const risk = riskOf(unit.prediction);
  const t = unit.telemetry;
  return {
    id: unit.unitId,
    route: routeLabel(unit),
    lat: t?.lat ?? NaN,
    lon: t?.lon ?? NaN,
    hasPosition: t?.lat != null && t?.lon != null,
    risk: risk.level,
    riskLabel: risk.label,
    status: risk.status,
    delayText: unit.prediction ? formatDelay(unit.prediction.delaySeconds) : "нет данных",
    speedText: formatSpeed(t?.speedKmh ?? null),
    sectionText: sectionLabel(unit),
    reason: unit.prediction ? displayReason(unit.prediction.reason) : null,
    telemetryReceivedAt: t?.receivedAt ?? null,
    telemetrySeedOnly: t !== null && t.receivedAt === null,
    predictionGeneratedAt: unit.prediction?.generatedAt ?? null,
    prediction: unit.prediction,
  };
}

/**
 * Views for all units with structural sharing: a unit whose state object did not change keeps
 * the same view object (applyTelemetry / mergeAlerts replace only the unit an event touched), so
 * one unit's packet does not hand every row and marker a fresh object.
 */
export function buildVehicleViews(units: DashboardState, cache: WeakMap<UnitState, VehicleView>): VehicleView[] {
  return Object.values(units).map((unit) => {
    let view = cache.get(unit);
    if (!view) {
      view = toVehicleView(unit);
      cache.set(unit, view);
    }
    return view;
  });
}

function compareUnitIds(a: string, b: string): number {
  return a.localeCompare(b, "en", { numeric: true });
}

/**
 * Marker input for the map: only map-visible fields, in stable unit-id order (independent of the
 * risk-sorted list). An unchanged marker keeps its object and, when no marker changed, the previous
 * array is returned, so realtime updates that do not move or recolour a marker leave the map alone.
 */
export function mapVehiclesFrom(views: VehicleView[], previous: Vehicle[]): Vehicle[] {
  const previousById = new Map(previous.map((vehicle) => [vehicle.id, vehicle]));
  const next = views
    .filter((view) => view.hasPosition)
    .map((view) => {
      const before = previousById.get(view.id);
      return before && before.lat === view.lat && before.lon === view.lon && before.risk === view.risk && before.route === view.route
        ? before
        : { id: view.id, route: view.route, lat: view.lat, lon: view.lon, risk: view.risk };
    })
    .sort((a, b) => compareUnitIds(a.id, b.id));
  return next.length === previous.length && next.every((vehicle, i) => vehicle === previous[i]) ? previous : next;
}

const RISK_ORDER: Record<RiskState, number> = { high: 0, medium: 1, low: 2, unknown: 3 };

/** Problematic vehicles first, then by unit id (numeric when possible). */
export function compareVehicleViews(a: VehicleView, b: VehicleView): number {
  const byRisk = RISK_ORDER[a.risk] - RISK_ORDER[b.risk];
  if (byRisk !== 0) return byRisk;
  return compareUnitIds(a.id, b.id);
}
