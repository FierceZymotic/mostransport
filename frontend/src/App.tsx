import {
  YMap,
  YMapDefaultSchemeLayer,
  YMapDefaultFeaturesLayer,
} from './lib/ymaps3.ts';

function App() {
  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <YMap
        location={{
          center: [37.6176, 55.7558],
          zoom: 10,
        }}
        mode="vector"
      >
        <YMapDefaultSchemeLayer />
        <YMapDefaultFeaturesLayer />
      </YMap>
    </div>
  );
}

export default App;