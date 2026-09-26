import { useMemo, useState } from "react";
import { AlertTriangle, Bus, Clock3, Radio, Search, Wifi } from "lucide-react";
import { vehicles } from "./mock/vehicles";
import YandexMap from "./components/YandexMap";
import type { RiskLevel, Vehicle } from "./types";

const riskLabel: Record<RiskLevel, string> = {
  low: "Низкий",
  medium: "Средний",
  high: "Высокий",
};

function App() {
  const [selected, setSelected] = useState<Vehicle | null>(vehicles[0]);
  const [query, setQuery] = useState("");

  const filtered = useMemo(
    () =>
      vehicles.filter((vehicle) =>
        `${vehicle.id} ${vehicle.route}`.toLowerCase().includes(query.toLowerCase())
      ),
    [query]
  );

  const highRisk = vehicles.filter((v) => v.risk === "high").length;

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
        <Stat icon={<Bus />} title="ТС в потоке" value={vehicles.length} />
        <Stat icon={<AlertTriangle />} title="Высокий риск" value={highRisk} danger />
        <Stat icon={<Clock3 />} title="Горизонт прогноза" value="10–15 мин" />
        <Stat icon={<Wifi />} title="Backend" value="Mock" />
      </section>

      <section className="workspace">
        <div className="map">
          <div className="map-label">ЯНДЕКС КАРТЫ · MOCK DATA</div>
          <YandexMap
            vehicles={vehicles}
            selected={selected}
            onSelect={setSelected}
          />
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