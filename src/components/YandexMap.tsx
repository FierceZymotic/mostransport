import { useEffect, useState } from "react";
import { loadYandexMaps } from "../lib/ymaps";
import type { Vehicle } from "../types";

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

  const center: [number, number] = [37.617, 55.755];

  return (
    <YMap
      location={reactify.useDefault({ center, zoom: 11 })}
      className="yandex-map"
    >
      <YMapDefaultSchemeLayer />
      <YMapDefaultFeaturesLayer />

      {vehicles.map((vehicle) => (
        <YMapMarker
          key={vehicle.id}
          coordinates={reactify.useDefault([vehicle.lon, vehicle.lat])}
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
