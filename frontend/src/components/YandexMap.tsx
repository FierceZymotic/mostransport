import { useEffect, useState } from "react";
import { loadYandexMaps } from "../lib/ymaps";
import type { Vehicle } from "../types";

// One constant initial view: every re-render passes the identical value, so a realtime update
// can never re-apply (reset) the map location.
const DEFAULT_LOCATION = { center: [37.617, 55.755] as [number, number], zoom: 11 };

interface Props {
  vehicles: Vehicle[];
  selected: Vehicle | null;
  onSelect: (vehicle: Vehicle) => void;
}

export default function YandexMap({ vehicles, selected, onSelect }: Props) {
  const [api, setApi] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadYandexMaps().then(setApi).catch((e) => setError(e.message));
  }, []);

  if (error) {
    return <div className="map-error">Не удалось загрузить Яндекс Карты: {error}</div>;
  }

  if (!api) {
    return <div className="map-loading">Загрузка Яндекс Карт…</div>;
  }

  const {
    YMap,
    YMapDefaultSchemeLayer,
    YMapDefaultFeaturesLayer,
    YMapMarker,
    reactify,
  } = api;

  return (
    <YMap
      location={reactify.useDefault(DEFAULT_LOCATION)}
      className="yandex-map"
    >
      <YMapDefaultSchemeLayer />
      <YMapDefaultFeaturesLayer />

      {vehicles.map((vehicle) => (
        <YMapMarker
          key={vehicle.id}
          // Controlled position: the marker (keyed by unit id) is updated in place on telemetry.
          coordinates={[vehicle.lon, vehicle.lat]}
          zIndex={selected?.id === vehicle.id ? 10 : 1}
        >
          <button
            className={`ymarker ${vehicle.risk} ${
              selected?.id === vehicle.id ? "selected" : ""
            }`}
            onClick={() => onSelect(vehicle)}
            title={`${vehicle.id} · ${vehicle.route}`}
          >
            <span />
          </button>
        </YMapMarker>
      ))}
    </YMap>
  );
}
