import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, Bus, Clock3, Radio, Search, Wifi } from "lucide-react";
import YandexMap from "./components/YandexMap";
import type { RiskLevel, Vehicle } from "./types";

function mapTelemetryToVehicle(payload: {
  unitId: number;
  latitude: number;
  longitude: number;
  speed: number;
}): Vehicle {
  const speedKmh = Math.max(0, Math.round((Number(payload.speed) || 0) * 3.6));

  return {
    id: String(payload.unitId),
    route: "LIVE",
    lat: Number(payload.latitude),
    lon: Number(payload.longitude),
    speed: speedKmh,
    delayMinutes: 0,
    risk: speedKmh < 10 ? "high" : speedKmh < 20 ? "medium" : "low",
    reason:
      speedKmh < 10
        ? "Снижение скорости в реальном времени"
        : speedKmh < 20
          ? "Небольшое отклонение от графика"
          : "Нормальный режим движения",
    segment: `Unit ${payload.unitId}`,
  };
}

const riskLabel: Record<RiskLevel, string> = {
  low: "Низкий",
  medium: "Средний",
  high: "Высокий",
};

type DashboardSummaryResponse = {
  total: number;
  highRisk: number;
  mediumRisk: number;
  lowRisk: number;
  vehicles: {
    id: string;
    route: string;
    lat: number;
    lon: number;
    speed: number;
    delayMinutes: number;
    risk: RiskLevel;
    reason: string;
    segment: string;
    updatedAt?: string;
  }[];
};

function App() {
  const [selected, setSelected] = useState<Vehicle | null>(null);
  const [query, setQuery] = useState("");
  const [liveVehicles, setLiveVehicles] = useState<Vehicle[]>([]);
  const [dashboardVehicles, setDashboardVehicles] = useState<Vehicle[]>([]);
  const [liveConnected, setLiveConnected] = useState(false);
  const [dashboardConnected, setDashboardConnected] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let retryTimer: number | undefined;

    const loadDashboard = () => {
      fetch("http://localhost:3000/prediction/dashboard/summary")
        .then(async (response) => {
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
          }

          const payload = (await response.json()) as DashboardSummaryResponse;

          if (cancelled) {
            return;
          }

          if (Array.isArray(payload.vehicles)) {
            const normalized = payload.vehicles.map((vehicle) => ({
              ...vehicle,
              speed: Number(vehicle.speed ?? 0),
              lat: Number(vehicle.lat ?? 0),
              lon: Number(vehicle.lon ?? 0),
              delayMinutes: Number(vehicle.delayMinutes ?? 0),
            }));

            setDashboardVehicles(normalized);
            setDashboardConnected(true);

            if (!selected) {
              setSelected(normalized[0] ?? null);
            }
          }
        })
        .catch(() => {
          if (cancelled) {
            return;
          }

          setDashboardConnected(false);
          retryTimer = window.setTimeout(loadDashboard, 4000);
        });
    };

    loadDashboard();

    return () => {
      cancelled = true;
      if (retryTimer) {
        window.clearTimeout(retryTimer);
      }
    };
  }, [selected]);

  useEffect(() => {
    const baseList = liveConnected ? liveVehicles : dashboardConnected ? dashboardVehicles : [];
    if (!baseList.length) {
      setSelected(null);
      return;
    }

    setSelected((current) => current && baseList.some((vehicle) => vehicle.id === current.id)
      ? current
      : baseList[0]);
  }, [liveConnected, dashboardConnected, liveVehicles, dashboardVehicles]);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let reconnectTimer: number | undefined;

    const connectSocket = () => {
      socket = new WebSocket("ws://localhost:3000/live");

      socket.onopen = () => {
        setLiveConnected(true);
      };

      socket.onerror = () => {
        setLiveConnected(false);
      };

      socket.onclose = () => {
        setLiveConnected(false);
        if (reconnectTimer) {
          window.clearTimeout(reconnectTimer);
        }
        reconnectTimer = window.setTimeout(connectSocket, 2500);
      };

      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as {
            event?: string;
            payload?: {
              unitId?: number;
              latitude?: number;
              longitude?: number;
              speed?: number;
            };
          };

          if (message.event !== "telemetry" || !message.payload) {
            return;
          }

          const nextVehicle = mapTelemetryToVehicle({
            unitId: Number(message.payload.unitId ?? 0),
            latitude: Number(message.payload.latitude ?? 0),
            longitude: Number(message.payload.longitude ?? 0),
            speed: Number(message.payload.speed ?? 0),
          });

          if (!Number.isFinite(nextVehicle.lat) || !Number.isFinite(nextVehicle.lon)) {
            return;
          }

          setLiveVehicles((current) => {
            const existingIndex = current.findIndex((vehicle) => vehicle.id === nextVehicle.id);

            if (existingIndex === -1) {
              return [nextVehicle, ...current].slice(0, 20);
            }

            const updated = [...current];
            updated[existingIndex] = { ...updated[existingIndex], ...nextVehicle };
            return updated;
          });
        } catch {
          // Ignore malformed websocket payloads while the backend stream is warming up.
        }
      };
    };

    connectSocket();

    return () => {
      if (reconnectTimer) {
        window.clearTimeout(reconnectTimer);
      }
      socket?.close();
    };
  }, []);

  const visibleVehicles = liveConnected
    ? liveVehicles
    : dashboardConnected
      ? dashboardVehicles
      : [];

  const sourceLabel = liveConnected ? "ЯНДЕКС КАРТЫ · LIVE DATA" : dashboardConnected ? "ЯНДЕКС КАРТЫ · DASHBOARD DATA" : "ЯНДЕКС КАРТЫ · WAITING FOR DATA";

  const filtered = useMemo(
    () =>
      visibleVehicles.filter((vehicle) =>
        `${vehicle.id} ${vehicle.route}`.toLowerCase().includes(query.toLowerCase())
      ),
    [query, visibleVehicles]
  );

  const highRisk = visibleVehicles.filter((v) => v.risk === "high").length;

  return (
    <main className="app">
      <header className="topbar">
        <div>
          <div className="eyebrow">МОСКОВСКИЙ ТРАНСПОРТ · REAL-TIME</div>
          <h1>Мониторинг движения</h1>
        </div>
        <div className="live">
          <span className="live-dot" />
          <Radio size={17} />
          Поток данных активен
        </div>
      </header>

      <section className="stats">
        <Stat icon={<Bus />} title="ТС в потоке" value={visibleVehicles.length} />
        <Stat icon={<AlertTriangle />} title="Высокий риск" value={highRisk} danger />
        <Stat icon={<Clock3 />} title="Горизонт прогноза" value="10–15 мин" />
        <Stat icon={<Wifi />} title="Backend" value={liveConnected ? "Live" : dashboardConnected ? "Dashboard" : "Waiting"} />
      </section>

      <section className="workspace">
        <div className="map">
          <div className="map-label">{sourceLabel}</div>
          {visibleVehicles.length ? (
            <YandexMap
              vehicles={visibleVehicles}
              selected={selected}
              onSelect={setSelected}
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
                className={`vehicle-card ${selected?.id === vehicle.id ? "active" : ""}`}
                onClick={() => setSelected(vehicle)}
              >
                <div className={`risk-dot ${vehicle.risk}`} />
                <div className="vehicle-main">
                  <strong>{vehicle.id}</strong>
                  <span>Маршрут {vehicle.route}</span>
                </div>
                <div className="delay">
                  {vehicle.delayMinutes ? `+${vehicle.delayMinutes} мин` : "По графику"}
                </div>
              </button>
            ))}
          </div>

          {selected && (
            <div className="incident">
              <div className="incident-title">
                <div>
                  <span className="eyebrow">КАРТОЧКА ИНЦИДЕНТА</span>
                  <h2>{selected.id}</h2>
                </div>
                <span className={`badge ${selected.risk}`}>{riskLabel[selected.risk]}</span>
              </div>

              <div className="incident-row">
                <span>Маршрут</span>
                <strong>{selected.route}</strong>
              </div>
              <div className="incident-row">
                <span>Прогноз</span>
                <strong>+{selected.delayMinutes} мин</strong>
              </div>
              <div className="incident-row">
                <span>Скорость</span>
                <strong>{selected.speed} км/ч</strong>
              </div>
              <div className="incident-row">
                <span>Участок</span>
                <strong>{selected.segment}</strong>
              </div>

              <div className="reason">
                <AlertTriangle size={18} />
                <div>
                  <span>Предполагаемая причина</span>
                  <strong>{selected.reason}</strong>
                </div>
              </div>
            </div>
          )}
        </aside>
      </section>
    </main>
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