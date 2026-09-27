import { NdtpPacket } from './ndtp.packet.js';

const NPL_HEADER_SIZE = 15;
const NPL_SIGNATURE = 0x7e7e;
const MAX_PACKET_SIZE = 65535;

export class NdtpPacketExtractor {
  private buffer = Buffer.alloc(0);

  push(chunk: Buffer): NdtpPacket[] {
    this.buffer = Buffer.concat([this.buffer, chunk]);

    const packets: NdtpPacket[] = [];

    while (true) {
      const packet = this.extractNextPacket();

      if (!packet) {
        break;
      }

      packets.push(packet);
    }

    return packets;
  }

  private extractNextPacket(): NdtpPacket | null {
    if (this.buffer.length < NPL_HEADER_SIZE) {
      return null;
    }

    const signature = this.buffer.readUInt16LE(0);

    if (signature !== NPL_SIGNATURE) {
      this.buffer = this.buffer.subarray(1);
      return null;
    }

    const dataSize = this.buffer.readUInt16LE(2);

    const packetSize = NPL_HEADER_SIZE + dataSize;

    if (
      packetSize < NPL_HEADER_SIZE ||
      packetSize > MAX_PACKET_SIZE
    ) {
      this.buffer = this.buffer.subarray(2);
      return null;
    }

    if (this.buffer.length < packetSize) {
      return null;
    }

    const raw = this.buffer.subarray(0, packetSize);

    // NPL header (organizer spec §5.1): offset 8 u8 type, offset 9 u32 peerAddress
    // (= emulator unitId, 0…2147483647), offset 13 u16 requestId. The unit identity is the
    // full u32 peerAddress; reading a single byte would merge vehicles that share a low byte.
    const type = this.buffer.readUInt8(8);
    const peerAddress = this.buffer.readUInt32LE(9);
    const unitId = peerAddress;

    const payload = this.buffer.subarray(
      NPL_HEADER_SIZE,
      packetSize,
    );

    console.log(
      'HEADER:',
      this.buffer
        .subarray(0, NPL_HEADER_SIZE)
        .toString('hex'),
    );

    console.log(
      'unitId:',
      unitId,
      'peerAddress:',
      peerAddress,
      'type:',
      type,
    );

    this.buffer = this.buffer.subarray(packetSize);

    return {
      raw,
      dataSize,
      peerAddress,
      type,
      unitId,
      payload,
    };
  }
}