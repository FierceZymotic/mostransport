// NPH / realtime-cell classification (organizer spec §5.2, §5.4, §6.1).
// Only NAVDATA/REALTIME packets whose first cell is G6CellNav00 carry telemetry;
// handshakes (serviceId 0, type 100) and other services must not be parsed as navigation.
export const NPH_SERVICE_NAVDATA = 1;
export const NPH_TYPE_REALTIME = 101;
export const CELL_TYPE_NAV00 = 0;
const NPH_HEADER_SIZE = 10;
const CELL_HEADER_SIZE = 2;
const NAV00_SIZE = 26;

export function isRealtimeNavPayload(payload: Buffer): boolean {
  if (payload.length < NPH_HEADER_SIZE + CELL_HEADER_SIZE + NAV00_SIZE) {
    return false;
  }
  const serviceId = payload.readUInt16LE(0);
  const nphType = payload.readUInt16LE(2);
  const firstCellType = payload.readUInt8(NPH_HEADER_SIZE);
  return serviceId === NPH_SERVICE_NAVDATA && nphType === NPH_TYPE_REALTIME && firstCellType === CELL_TYPE_NAV00;
}
