import React from "react";
import ReactDOM from "react-dom";

let initialized = false;

export async function loadYandexMaps() {
  if (typeof ymaps3 === "undefined") {
    throw new Error("Yandex Maps API не загрузился. Проверь API key и HTTP Referer.");
  }

  if (!initialized) {
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
