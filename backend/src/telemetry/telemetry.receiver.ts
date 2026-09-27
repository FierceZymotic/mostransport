import {
  Injectable,
  Logger,
  OnModuleDestroy,
  OnModuleInit,
} from '@nestjs/common';
import { Server, Socket, createServer } from 'node:net';
import { demoClock } from '../common/demo-clock.js';
import { NdtpPacket } from './ndtp/ndtp.packet.js';
import { NdtpPacketExtractor } from './ndtp/ndtp.packet-extractor.js';
import { isRealtimeNavPayload } from './ndtp/ndtp.realtime.js';
import { TelemetryParser } from './telemetry.parser.js';
import { TelemetryHistory } from './telemetry-history.js';
import { TelemetryRepository } from './telemetry.repository.js';
import { TelemetryStreamService } from './telemetry-stream.service.js';
import { VehicleState } from './vehicle-state.js';

@Injectable()
export class TelemetryReceiver implements OnModuleInit, OnModuleDestroy {
  private readonly logger = new Logger(TelemetryReceiver.name);

  private server!: Server;

  private readonly port = Number(
    process.env.TELEMETRY_PORT ?? 9000,
  );

  constructor(
    private readonly parser: TelemetryParser,
    private readonly history: TelemetryHistory,
    private readonly repository: TelemetryRepository,
    private readonly stream: TelemetryStreamService,
  ) {}

  onModuleInit() {
    this.server = createServer((socket) => {
      this.handleConnection(socket);
    });

    this.server.listen(this.port, '0.0.0.0', () => {
      this.logger.log(
        `NDTP TCP receiver listening on :${this.port}`,
      );
    });
  }

  private handleConnection(socket: Socket) {
    const remoteAddress = `${socket.remoteAddress}:${socket.remotePort}`;

    this.logger.log(
      `NDTP connection: ${remoteAddress}`,
    );

    const extractor = new NdtpPacketExtractor();

    // Packets of one connection are ingested strictly in arrival order. The socket does not
    // await 'data' handlers, so without this chain two chunks could interleave their DB
    // writes and the telemetry row id (the tie-break for equal event times) would not
    // follow arrival order.
    let ingestion: Promise<void> = Promise.resolve();

    socket.on('data', (chunk: Buffer) => {
      this.logger.log(
        `Received ${chunk.length} bytes`,
      );

      const packets = extractor.push(chunk);

      ingestion = ingestion
        .then(() => this.ingestPackets(packets))
        .catch((error: unknown) => {
          this.logger.error(
            `NDTP ingestion failed: ${error instanceof Error ? error.message : String(error)}`,
          );
        });
    });

    socket.on('close', () => {
      this.logger.log(
        `NDTP connection closed: ${remoteAddress}`,
      );
    });

    socket.on('error', (error) => {
      this.logger.error(
        `NDTP socket error: ${error.message}`,
      );
    });
  }

  private async ingestPackets(packets: NdtpPacket[]): Promise<void> {
    for (const packet of packets) {
      this.logger.debug(
        `NDTP packet extracted: ${packet.raw.length} bytes`,
      );

      // Only NAVDATA/REALTIME packets whose first cell is G6CellNav00 carry a position.
      // The handshake payload is exactly 28 bytes, so a length check alone lets it through.
      if (!isRealtimeNavPayload(packet.payload)) {
        this.logger.debug(
          `Skipping non-telemetry packet (not NAVDATA/REALTIME with Nav00): payload=${packet.payload.length} bytes`,
        );
        continue;
      }

      let telemetry: VehicleState;
      try {
        telemetry = this.parser.parse(
          packet.payload,
          packet.unitId,
        );
      } catch (error) {
        this.logger.warn(
          `Dropping NDTP packet from unit ${packet.unitId}: ${error instanceof Error ? error.message : String(error)}`,
        );
        continue;
      }

      // The session clock is applied exactly once, here (identity unless DEMO_CLOCK_MODE
      // is set); it is a constant offset, so every interval and the order are preserved.
      const state: VehicleState = {
        ...telemetry,
        timestamp: demoClock.toModelEpochSeconds(telemetry.timestamp),
      };

      this.history.add(state);

      try {
        await this.repository.save(state);
        await this.repository.saveLastState(state);
      } catch (dbError) {
        this.logger.warn(
          `Telemetry persisted locally but DB write failed for unit ${packet.unitId}: ${dbError instanceof Error ? dbError.message : String(dbError)}`,
        );
      }

      // Publish only after persistence: a live prediction triggered by this packet
      // (T = its event time) must find the packet in the stored history.
      this.stream.publish(state);

      this.logger.log(
        `VehicleState: ${JSON.stringify(state)}`,
      );
    }
  }

  onModuleDestroy() {
    this.server?.close();
  }
}