import React from "react";
import ReactDOM from "react-dom/client";

let initialized = false;

export async function loadYandexMaps() {
  if (!initialized) {
    const apiKey = import.meta.env.VITE_YANDEX_MAPS_API_KEY;

    if (!apiKey) {
      throw new Error("VITE_YANDEX_MAPS_API_KEY is not set");
    }

    if (!document.querySelector('script[data-yandex-maps]')) {
      const script = document.createElement("script");
      script.src = `https://api-maps.yandex.ru/v3/?apikey=${encodeURIComponent(apiKey)}&lang=ru_RU`;
      script.async = true;
      script.dataset.yandexMaps = "true";
      document.head.appendChild(script);
    }

    await new Promise<void>((resolve, reject) => {
      if (typeof ymaps3 !== "undefined") {
        resolve();
        return;
      }

      const check = window.setInterval(() => {
        if (typeof ymaps3 !== "undefined") {
          window.clearInterval(check);
          resolve();
        }
      }, 50);

      window.setTimeout(() => {
        window.clearInterval(check);
        reject(new Error("Yandex Maps API load timeout"));
      }, 15000);
    });

    await ymaps3.ready;
    initialized = true;
  }

  const ymaps3Reactify = await ymaps3.import("@yandex/ymaps3-reactify");
  const reactify = ymaps3Reactify.reactify.bindTo(React, ReactDOM);
  return {
    ...reactify.module(ymaps3),
    reactify,
  };
}
