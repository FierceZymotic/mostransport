import {
  ConnectedSocket,
  MessageBody,
  OnGatewayConnection,
  OnGatewayDisconnect,
  SubscribeMessage,
  WebSocketGateway,
} from '@nestjs/websockets';
import type { WebSocket } from 'ws';
import { TelemetryStreamService } from './telemetry-stream.service.js';

@WebSocketGateway({
  cors: true,
  path: '/live',
})
export class TelemetryGateway implements OnGatewayConnection, OnGatewayDisconnect {
  private readonly unsubscribeByClient = new Map<WebSocket, () => void>();

  constructor(private readonly stream: TelemetryStreamService) {}

  @SubscribeMessage('ping')
  handlePing(@MessageBody() body: unknown, @ConnectedSocket() client: WebSocket) {
    client.send(JSON.stringify({ event: 'pong', payload: body ?? null }));
    return { event: 'pong', payload: body ?? null };
  }

  handleConnection(client: WebSocket) {
    const unsubscribe = this.stream.subscribe({
      send: (message: string) => client.send(message),
      close: () => client.close(),
    });

    this.unsubscribeByClient.set(client, unsubscribe);
  }

  handleDisconnect(client: WebSocket) {
    const unsubscribe = this.unsubscribeByClient.get(client);
    if (unsubscribe) {
      unsubscribe();
      this.unsubscribeByClient.delete(client);
    }
  }
}
