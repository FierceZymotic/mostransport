import {
  Injectable,
  Logger,
  OnModuleDestroy,
  OnModuleInit,
} from '@nestjs/common';
import { Server, Socket, createServer } from 'node:net';
import { NdtpPacketExtractor } from './ndtp/ndtp.packet-extractor.js';
import { TelemetryParser } from './telemetry.parser.js';
import { TelemetryHistory } from './telemetry-history.js';

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

    socket.on('data', (chunk: Buffer) => {
      this.logger.log(
        `Received ${chunk.length} bytes`,
      );

      const packets = extractor.push(chunk);

      for (const packet of packets) {
  this.logger.debug(
    `NDTP packet extracted: ${packet.raw.length} bytes`,
  );
console.log(
  'Receiver packet.payload length:',
  packet.payload.length,
);

const telemetry = this.parser.parse(
  packet.payload,
  packet.unitId,
);

this.history.add(telemetry);

this.logger.log(
  `VehicleState: ${JSON.stringify(telemetry)}`,
);
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