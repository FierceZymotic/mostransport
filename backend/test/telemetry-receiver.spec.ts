import { EventEmitter } from 'node:events';
import { describe, expect, it, vi } from 'vitest';

import { TelemetryParser } from '../src/telemetry/telemetry.parser.js';
import { TelemetryReceiver } from '../src/telemetry/telemetry.receiver.js';
import type { VehicleState } from '../src/telemetry/vehicle-state.js';

function frame(unitId: number, ts: number, speed: number, opts: { handshake?: boolean; truncatedNav?: boolean } = {}): Buffer {
  const nph = Buffer.alloc(10);
  nph.writeUInt16LE(opts.handshake ? 0 : 1, 0);
  nph.writeUInt16LE(opts.handshake ? 100 : 101, 2);
  let body: Buffer;
  if (opts.handshake) {
    body = Buffer.alloc(18);
    body.writeUInt32LE(unitId, 6);
  } else {
    body = Buffer.alloc(28);
    body.writeUInt8(0, 0);
    body.writeUInt32LE(ts, 2);
    body.writeUInt32LE(376173210, 6);
    body.writeUInt32LE(557551234, 10);
    body.writeUInt8(0x80 | 0x40 | 0x20, 14);
    body.writeUInt16LE(speed, 16);
  }
  const npl = Buffer.alloc(15);
  npl.writeUInt16LE(0x7e7e, 0);
  npl.writeUInt16LE(nph.length + body.length, 2);
  npl.writeUInt8(2, 8);
  npl.writeUInt32LE(unitId, 9);
  return Buffer.concat([npl, nph, body]);
}

function setup() {
  const events: string[] = [];
  const saved: VehicleState[] = [];
  const published: VehicleState[] = [];
  let delay = 5;
  const repository = {
    // First writes are slower than later ones: without per-connection ordering the second
    // chunk's rows would be stored first.
    save: vi.fn(async (state: VehicleState) => {
      await new Promise((r) => setTimeout(r, Math.max(0, delay--)));
      saved.push(state);
      events.push(`save:${state.timestamp}`);
    }),
    saveLastState: vi.fn(async () => undefined),
  };
  const stream = { publish: vi.fn((state: VehicleState) => { published.push(state); events.push(`publish:${state.timestamp}`); }) };
  const history = { add: vi.fn() };
  const receiver = new TelemetryReceiver(new TelemetryParser(), history as any, repository as any, stream as any);
  const socket = Object.assign(new EventEmitter(), { remoteAddress: '127.0.0.1', remotePort: 1 });
  (receiver as any).handleConnection(socket);
  return { socket, events, saved, published, repository };
}

describe('TelemetryReceiver ingestion order', () => {
  it('stores packets in arrival order and publishes each only after it is persisted', async () => {
    const { socket, events, saved, published } = setup();
    socket.emit('data', Buffer.concat([frame(1105498, 1767700000, 10), frame(1105498, 1767700000, 20)]));
    socket.emit('data', frame(1105498, 1767700005, 30));
    socket.emit('data', frame(1105498, 1767700010, 40));
    await vi.waitFor(() => expect(published).toHaveLength(4));
    expect(saved.map((s) => s.speed)).toEqual([10, 20, 30, 40]); // equal-time packets both kept, in order
    for (const state of published) {
      expect(events.indexOf(`save:${state.timestamp}`)).toBeLessThan(events.indexOf(`publish:${state.timestamp}`));
    }
    expect(saved.every((s) => s.unitId === 1105498)).toBe(true);
  });

  it('applies the session clock once (identity by default) and never stores handshakes', async () => {
    const { socket, saved, published, repository } = setup();
    socket.emit('data', Buffer.concat([frame(26, 0, 0, { handshake: true }), frame(26, 1767700000, 7)]));
    await vi.waitFor(() => expect(published).toHaveLength(1));
    expect(repository.save).toHaveBeenCalledTimes(1);
    expect(saved[0].timestamp).toBe(1767700000);
    expect(saved[0].unitId).toBe(26);
  });
});
