import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, Bus, Clock3, Radio, Search, Wifi } from "lucide-react";
import YandexMap from "./components/YandexMap";
import type { Vehicle } from "./types";
import {
  applyTelemetry,
  buildVehicleViews,
  compareVehicleViews,
  formatAge,
  mapVehiclesFrom,
  mergeAlerts,
  predictionEventDetails,
  seedFromSummary,
  type AlertRow,
  type DashboardState,
  type PredictionEventDetails,
  type PredictionEventPayload,
  reasonPresentation,
  type SummaryVehicle,
  type TelemetryPayload,
  type UnitState,
  type VehicleView,
} from "./dashboard/state";

const API_BASE = "http://localhost:3000";
const SUMMARY_URL = `${API_BASE}/prediction/dashboard/summary`;
const ALERTS_URL = `${API_BASE}/prediction/dashboard/alerts`;
const LIVE_URL = "ws://localhost:3000/live";
const MAX_PENDING_DETAILS = 200;

// The map re-renders only when its marker input or the selection actually changes.
const StableYandexMap = memo(YandexMap);

// Views keyed by unit state object (pure, so shared safely): an unchanged unit keeps its view.
const viewCache = new WeakMap<UnitState, VehicleView>();

function App() {
  const [units, setUnits] = useState<DashboardState>({});
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [liveConnected, setLiveConnected] = useState(false);
  const [dashboardConnected, setDashboardConnected] = useState(false);
  // WS-only prediction details by request_id, until the alerts row names the unit.
  const eventDetails = useRef<Record<string, PredictionEventDetails>>({});
  const refreshAlerts = useRef<() => void>(() => undefined);

  // Initial positions / speed (telemetry-owned fields only) from the last-state summary.
  useEffect(() => {
    let cancelled = false;
    let retryTimer: number | undefined;

    const loadSummary = () => {
      fetch(SUMMARY_URL)
        .then(async (response) => {
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
          }
          const payload = (await response.json()) as { vehicles?: SummaryVehicle[] };
          if (cancelled) {
            return;
          }
          if (Array.isArray(payload.vehicles)) {
            setUnits((current) => seedFromSummary(current, payload.vehicles ?? []));
          }
          setDashboardConnected(true);
        })
        .catch(() => {
          if (cancelled) {
            return;
          }
          setDashboardConnected(false);
          retryTimer = window.setTimeout(loadSummary, 4000);
        });
    };

    loadSummary();

    return () => {
      cancelled = true;
      window.clearTimeout(retryTimer);
    };
  }, []);

  // Prediction history: seeds the latest prediction per unit; re-read on every WS prediction.
  useEffect(() => {
    let cancelled = false;
    let retryTimer: number | undefined;
    let debounceTimer: number | undefined;

    const loadAlerts = () => {
      fetch(ALERTS_URL)
        .then(async (response) => {
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
          }
          const rows = (await response.json()) as AlertRow[];
          if (!cancelled && Array.isArray(rows)) {
            setUnits((current) => mergeAlerts(current, rows, eventDetails.current));
          }
        })
        .catch(() => {
          if (!cancelled) {
            window.clearTimeout(retryTimer);
            retryTimer = window.setTimeout(loadAlerts, 4000);
          }
        });
    };

    refreshAlerts.current = () => {
      window.clearTimeout(debounceTimer);
      debounceTimer = window.setTimeout(loadAlerts, 250);
    };
    loadAlerts();

    return () => {
      cancelled = true;
      refreshAlerts.current = () => undefined;
      window.clearTimeout(retryTimer);
      window.clearTimeout(debounceTimer);
    };
  }, []);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let reconnectTimer: number | undefined;
    let closed = false;

    const connectSocket = () => {
      socket = new WebSocket(LIVE_URL);

      socket.onopen = () => {
        setLiveConnected(true);
        refreshAlerts.current(); // predictions made while disconnected
      };

      socket.onerror = () => {
        setLiveConnected(false);
      };

      socket.onclose = () => {
        setLiveConnected(false);
        if (closed) {
          return;
        }
        window.clearTimeout(reconnectTimer);
        reconnectTimer = window.setTimeout(connectSocket, 2500);
      };

      socket.onmessage = (event) => {
        let message: { event?: string; payload?: unknown; meta?: unknown };
        try {
          message = JSON.parse(event.data);
        } catch {
          return; // ignore malformed payloads while the backend stream is warming up
        }
        if (message.event === "telemetry" && message.payload) {
          const payload = message.payload as TelemetryPayload;
          setUnits((current) => applyTelemetry(current, payload, Date.now()));
        } else if (message.event === "prediction" && message.payload) {
          const payload = message.payload as PredictionEventPayload;
          if (payload.request_id) {
            const details = eventDetails.current;
            details[payload.request_id] = predictionEventDetails(payload, message.meta);
            const keys = Object.keys(details);
            for (const key of keys.slice(0, Math.max(0, keys.length - MAX_PENDING_DETAILS))) {
              delete details[key];
            }
          }
          refreshAlerts.current();
        }
      };
    };

    connectSocket();

    return () => {
      closed = true;
      window.clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, []);

  // Unchanged units keep their view objects; the map gets id-ordered, map-only marker objects
  // that are reused (and the same array) unless a marker moved or changed risk / route.
  const vehicles = useMemo(
    () => buildVehicleViews(units, viewCache).sort(compareVehicleViews),
    [units]
  );
  // Derived from the previous render's value (React "storing information from previous renders"):
  // updated synchronously in the same render, reused when no marker changed.
  const [mapInput, setMapInput] = useState<{ source: VehicleView[] | null; vehicles: Vehicle[] }>({ source: null, vehicles: [] });
  let mapVehicles = mapInput.vehicles;
  if (mapInput.source !== vehicles) {
    mapVehicles = mapVehiclesFrom(vehicles, mapInput.vehicles);
    setMapInput({ source: vehicles, vehicles: mapVehicles });
  }
  const selected = selectedId ? vehicles.find((vehicle) => vehicle.id === selectedId) ?? null : null;
  const selectedMarker = selectedId ? mapVehicles.find((vehicle) => vehicle.id === selectedId) ?? null : null;
  const selectVehicle = useCallback((vehicle: Vehicle) => setSelectedId(vehicle.id), []);

  const sourceLabel = liveConnected ? "ЯНДЕКС КАРТЫ · LIVE DATA" : dashboardConnected ? "ЯНДЕКС КАРТЫ · DASHBOARD DATA" : "ЯНДЕКС КАРТЫ · WAITING FOR DATA";

  const filtered = useMemo(
    () =>
      vehicles.filter((vehicle) =>
        `${vehicle.id} ${vehicle.route}`.toLowerCase().includes(query.toLowerCase())
      ),
    [query, vehicles]
  );

  const highRisk = vehicles.filter((v) => v.risk === "high").length;

  return (
    <main className="app">
      <header className="topbar">
        <div>
          <div className="eyebrow">МОСКОВСКИЙ ТРАНСПОРТ · REAL-TIME</div>
          <h1>Мониторинг движения</h1>
        </div>
        <div className={`live ${liveConnected ? "" : "offline"}`}>
          <span className="live-dot" />
          <Radio size={17} />
          {liveConnected ? "Поток данных активен" : "Поток данных недоступен"}
        </div>
      </header>

      <section className="stats">
        <Stat icon={<Bus />} title="ТС в потоке" value={vehicles.length} />
        <Stat icon={<AlertTriangle />} title="Высокий риск" value={highRisk} danger />
        <Stat icon={<Clock3 />} title="Горизонт прогноза" value="10–15 мин" />
        <Stat icon={<Wifi />} title="Backend" value={liveConnected ? "Live" : dashboardConnected ? "Dashboard" : "Waiting"} />
      </section>

      <section className="workspace">
        <div className="map">
          <div className="map-label">{sourceLabel}</div>
          {mapVehicles.length ? (
            <StableYandexMap
              vehicles={mapVehicles}
              selected={selectedMarker}
              onSelect={selectVehicle}
            />
          ) : (
            <div className="empty-state">Ожидание live-данных от backend…</div>
          )}
        </div>

        <aside className="sidebar">
          <div className="sidebar-header">
            <div>
              <h2>Транспорт</h2>
              <span>{filtered.length} объектов</span>
            </div>
            <div className="search">
              <Search size={16} />
              <input
                placeholder="ТС / маршрут"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            </div>
          </div>

          <div className="vehicle-list">
            {filtered.map((vehicle) => (
              <button
                key={vehicle.id}
                className={`vehicle-card risk-${vehicle.risk} ${selectedId === vehicle.id ? "active" : ""}`}
                onClick={() => setSelectedId(vehicle.id)}
              >
                <div className={`risk-dot ${vehicle.risk}`} />
                <div className="vehicle-main">
                  <strong>{vehicle.id} · {vehicle.route}</strong>
                  <span className={`vehicle-status ${vehicle.risk}`}>
                    {vehicle.status}
                    {vehicle.risk === "low" ? ` · ${vehicle.delayText}` : ""}
                  </span>
                </div>
              </button>
            ))}
          </div>

          {selected ? <VehicleCard vehicle={selected} /> : (
            <div className="incident incident-empty">
              <span className="eyebrow">КАРТОЧКА ИНЦИДЕНТА</span>
              <p>Выберите ТС на карте или в списке</p>
            </div>
          )}
        </aside>
      </section>
    </main>
  );
}

function VehicleCard({ vehicle }: { vehicle: VehicleView }) {
  const reason = vehicle.reason ? reasonPresentation(vehicle.reason) : null;
  return (
    <div className={`incident risk-${vehicle.risk}`}>
      <div className="incident-title">
        <div>
          <span className="eyebrow">КАРТОЧКА ИНЦИДЕНТА</span>
          <h2>{vehicle.id} · {vehicle.route}</h2>
        </div>
      </div>

      <span className={`badge ${vehicle.risk}`}>
        <span className={`risk-dot ${vehicle.risk}`} />
        {vehicle.riskLabel}
      </span>

      <div className="incident-rows">
        <div className="incident-row">
          <span>Прогноз</span>
          <strong>{vehicle.delayText}</strong>
        </div>
        <div className="incident-row">
          <span>Скорость</span>
          <strong>{vehicle.speedText}</strong>
        </div>
        <div className="incident-row">
          <span>Участок</span>
          <strong>{vehicle.sectionText}</strong>
        </div>
      </div>

      {reason && (
        <div className={`reason ${reason.tone}`}>
          <AlertTriangle size={18} />
          <div>
            <span>Причина</span>
            <strong title={reason.tooltip}>{reason.text}</strong>
          </div>
        </div>
      )}

      <Freshness vehicle={vehicle} />
    </div>
  );
}

/**
 * Telemetry event times are backend model time (the demo clock may shift them by whole days),
 * so telemetry age is measured from when this dashboard received the last live packet.
 * Prediction age uses the ML generation time (UTC wall clock).
 */
function Freshness({ vehicle }: { vehicle: VehicleView }) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  const telemetry = vehicle.telemetryReceivedAt !== null
    ? formatAge(vehicle.telemetryReceivedAt, now)
    : vehicle.telemetrySeedOnly ? "нет live-данных" : "нет данных";

  return (
    <div className="freshness">
      <span>Телеметрия: {telemetry}</span>
      <span>Прогноз: {formatAge(vehicle.predictionGeneratedAt, now)}</span>
    </div>
  );
}

function Stat({
  icon,
  title,
  value,
  danger = false,
}: {
  icon: React.ReactNode;
  title: string;
  value: string | number;
  danger?: boolean;
}) {
  return (
    <div className="stat">
      <div className={`stat-icon ${danger ? "danger" : ""}`}>{icon}</div>
      <div>
        <span>{title}</span>
        <strong>{value}</strong>
      </div>
    </div>
  );
}

export default App;
