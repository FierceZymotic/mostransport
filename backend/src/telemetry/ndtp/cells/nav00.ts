export interface Nav00Data {
  timestamp: number;
  longitude: number;
  latitude: number;

  locationValid: boolean;

  speedAvg: number;
  speedMax: number;

  course: number;
  track: number;

  altitude: number;
  nsat: number;
  pdop: number;
}

const NAV00_OFFSET = 12;
const NAV00_SIZE = 26;

export function parseNav00(payload: Buffer): Nav00Data {
  let offset: number;

  if (payload.length >= 38) {
    // Полный NDTP payload:
    // 12 байт служебных данных + 26 байт NAV00
    offset = 12;
  } else if (payload.length >= 28) {
    // Payload, содержащий cell header + NAV00:
    // 2 байта header + 26 байт NAV00
    offset = 2;
  } else {
    throw new Error(
      `Invalid G6CellNav00 payload: expected at least 28 bytes, got ${payload.length}`,
    );
  }

  const timestamp = payload.readUInt32LE(offset);

  const rawLongitude = payload.readUInt32LE(offset + 4);
  const rawLatitude = payload.readUInt32LE(offset + 8);

  const flags = payload.readUInt8(offset + 12);

  const locationValid = (flags & (1 << 7)) !== 0;
  const longitudePositive = (flags & (1 << 6)) !== 0;
  const latitudePositive = (flags & (1 << 5)) !== 0;

  const longitude =
    (longitudePositive ? 1 : -1) *
    rawLongitude /
    10_000_000;

  const latitude =
    (latitudePositive ? 1 : -1) *
    rawLatitude /
    10_000_000;

  const speedAvg = payload.readUInt16LE(offset + 14);
  const speedMax = payload.readUInt16LE(offset + 16);

  const course = payload.readUInt16LE(offset + 18);
  const track = payload.readUInt16LE(offset + 20);

  const altitude = payload.readInt16LE(offset + 22);

  const nsat = payload.readUInt8(offset + 24);
  const pdop = payload.readUInt8(offset + 25);

  return {
    timestamp,
    longitude,
    latitude,
    locationValid,
    speedAvg,
    speedMax,
    course,
    track,
    altitude,
    nsat,
    pdop,
  };
}