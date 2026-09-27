import {
  Injectable,
  Logger,
  OnModuleDestroy,
  OnModuleInit,
} from '@nestjs/common';
import { Server, Socket, createServer } from 'node:net';
import { normalizeTelemetryTimestamp } from '../prediction/prediction.service.js';
import { NdtpPacketExtractor } from './ndtp/ndtp.packet-extractor.js';
import { TelemetryParser } from './telemetry.parser.js';
import { TelemetryHistory } from './telemetry-history.js';
import { TelemetryRepository } from './telemetry.repository.js';
import { TelemetryStreamService } from './telemetry-stream.service.js';

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

    socket.on('data', async (chunk: Buffer) => {
      this.logger.log(
        `Received ${chunk.length} bytes`,
      );

      const packets = extractor.push(chunk);

      for (const packet of packets) {
        this.logger.debug(
          `NDTP packet extracted: ${packet.raw.length} bytes`,
        );

        if (packet.payload.length < 28) {
          this.logger.debug(
            `Skipping non-telemetry packet: payload=${packet.payload.length} bytes`,
          );
          continue;
        }

        try {
          const telemetry = this.parser.parse(
            packet.payload,
            packet.unitId,
          );

          const normalizedTelemetry = {
            ...telemetry,
            timestamp: normalizeTelemetryTimestamp(telemetry.timestamp),
          };

          this.history.add(normalizedTelemetry);
          this.stream.publish(normalizedTelemetry);

          try {
            await this.repository.save(normalizedTelemetry);
            await this.repository.saveLastState(normalizedTelemetry);
          } catch (dbError) {
            this.logger.warn(
              `Telemetry persisted locally but DB write failed for unit ${packet.unitId}: ${dbError instanceof Error ? dbError.message : String(dbError)}`,
            );
          }

          this.logger.log(
            `VehicleState: ${JSON.stringify(normalizedTelemetry)}`,
          );
        } catch (error) {
          this.logger.warn(
            `Dropping NDTP packet from unit ${packet.unitId}: ${error instanceof Error ? error.message : String(error)}`,
          );
        }
      }
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

  onModuleDestroy() {
    this.server?.close();
  }
}