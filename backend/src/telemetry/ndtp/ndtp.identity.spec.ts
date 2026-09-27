import { describe, expect, it } from 'vitest';

import { TelemetryParser } from '../telemetry.parser.js';
import { NdtpPacketExtractor } from './ndtp.packet-extractor.js';
import { isRealtimeNavPayload } from './ndtp.realtime.js';

// Same real emulator frame as ndtp.packet-extractor.spec.ts; NPL bytes 9..12 = 40 e2 01 00 -> 123456.
const REALTIME_HEX =
  '7e7e26000000c5c40240e201000000010065000100020000000000a463b66a4090781660c63a2180005e0190012823640096000a14';

function nplFrame(unitId: number, nph: Buffer, body: Buffer): Buffer {
  const npl = Buffer.alloc(15);
  npl.writeUInt16LE(0x7e7e, 0);
  npl.writeUInt16LE(nph.length + body.length, 2);
  npl.writeUInt8(2, 8);
  npl.writeUInt32LE(unitId, 9);
  return Buffer.concat([npl, nph, body]);
}

function nph(serviceId: number, type: number): Buffer {
  const header = Buffer.alloc(10);
  header.writeUInt16LE(serviceId, 0);
  header.writeUInt16LE(type, 2);
  header.writeUInt16LE(1, 4);
  header.writeUInt32LE(1, 6);
  return header;
}

function handshakeFrame(unitId: number): Buffer {
  const body = Buffer.alloc(18);
  body.writeUInt16LE(6, 0);
  body.writeUInt16LE(2, 2);
  body.writeUInt32LE(unitId, 6);
  body.writeUInt32LE(65535, 10);
  return nplFrame(unitId, nph(0, 100), body);
}

/** NAVDATA/REALTIME frame with a single G6CellNav00 (organizer spec §5.4, §6.1). */
function realtimeFrame(unitId: number, opts: { lat: number; lon: number; valid: boolean; speed: number; ts: number; serviceId?: number; cellType?: number }): Buffer {
  const cell = Buffer.alloc(28);
  cell.writeUInt8(opts.cellType ?? 0, 0); // cell type
  cell.writeUInt8(0, 1); // cell number
  cell.writeUInt32LE(opts.ts, 2);
  cell.writeUInt32LE(Math.round(Math.abs(opts.lon) * 1e7), 6);
  cell.writeUInt32LE(Math.round(Math.abs(opts.lat) * 1e7), 10);
  cell.writeUInt8((opts.valid ? 0x80 : 0) | (opts.lon >= 0 ? 0x40 : 0) | (opts.lat >= 0 ? 0x20 : 0), 14);
  cell.writeUInt16LE(opts.speed, 16);
  return nplFrame(unitId, nph(opts.serviceId ?? 1, 101), cell);
}

describe('NDTP identity and packet classification', () => {
  it('reads the unit identity as the full u32 peerAddress at NPL offset 9', () => {
    const [packet] = new NdtpPacketExtractor().push(Buffer.from(REALTIME_HEX, 'hex'));
    expect(packet.peerAddress).toBe(123456); // readUInt8(9) would give 64
    expect(packet.unitId).toBe(123456);
  });

  it('keeps high unit ids distinct end-to-end (extractor + parser), incl. a shared low byte and the max id', () => {
    const ids = [1105498, 26, 1166336, 2147483647]; // 1105498 & 0xff === 26
    const extractor = new NdtpPacketExtractor();
    const packets = extractor.push(Buffer.concat(ids.map((id, i) =>
      realtimeFrame(id, { lat: 55.75, lon: 37.62, valid: true, speed: 10 + i, ts: 1767670000 + i }))));
    const parser = new TelemetryParser();
    const states = packets.map((p) => parser.parse(p.payload, p.unitId));
    expect(states.map((s) => s.unitId)).toEqual(ids);
    expect(new Set(states.map((s) => s.unitId)).size).toBe(ids.length);
    expect(states.map((s) => s.speed)).toEqual([10, 11, 12, 13]);
  });

  it('classifies realtime Nav00 payloads and rejects handshakes and other services', () => {
    const [realtime] = new NdtpPacketExtractor().push(Buffer.from(REALTIME_HEX, 'hex'));
    const [handshake] = new NdtpPacketExtractor().push(handshakeFrame(1166336));
    const [otherService] = new NdtpPacketExtractor().push(realtimeFrame(7, { lat: 55.7, lon: 37.6, valid: true, speed: 1, ts: 1, serviceId: 0 }));
    const [otherCell] = new NdtpPacketExtractor().push(realtimeFrame(7, { lat: 55.7, lon: 37.6, valid: true, speed: 1, ts: 1, cellType: 8 }));
    expect(isRealtimeNavPayload(realtime.payload)).toBe(true);
    expect(handshake.payload.length).toBe(28); // passes the old `< 28` length filter
    expect(isRealtimeNavPayload(handshake.payload)).toBe(false);
    expect(isRealtimeNavPayload(otherService.payload)).toBe(false);
    expect(isRealtimeNavPayload(otherCell.payload)).toBe(false);
  });

  it('does not turn synthetic zero or out-of-range navigation into a valid position', () => {
    const parser = new TelemetryParser();
    const parse = (lat: number, lon: number) => {
      const [p] = new NdtpPacketExtractor().push(realtimeFrame(5, { lat, lon, valid: true, speed: 0, ts: 1767670000 }));
      return parser.parse(p.payload, p.unitId);
    };
    expect(parse(0, 0).locationValid).toBe(false); // emulator's synthetic Nav00 (zeros, flag valid)
    expect(parse(95, 37.6).locationValid).toBe(false);
    const ok = parse(55.7551234, 37.617321);
    expect(ok.locationValid).toBe(true);
    expect(ok.latitude).toBeCloseTo(55.7551234, 7);
    expect(ok.longitude).toBeCloseTo(37.617321, 7);
  });
});
