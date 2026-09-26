export interface NdtpPacket {
  raw: Buffer;
  dataSize: number;
  peerAddress: number;
  type: number;
  unitId: number;
  payload: Buffer;
}