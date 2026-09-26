import { NdtpPacketExtractor } from './ndtp.packet-extractor.js';

describe('NdtpPacketExtractor', () => {
  it('should extract a real NDTP packet', () => {
    const extractor = new NdtpPacketExtractor();

    const hex =
      '7e7e26000000c5c40240e201000000010065000100020000000000a463b66a4090781660c63a2180005e0190012823640096000a14';

    const packet = Buffer.from(hex, 'hex');

    const result = extractor.push(packet);

    expect(result).toHaveLength(1);

    expect(result[0].dataSize).toBe(38);
    expect(result[0].raw.length).toBe(53);
    expect(result[0].payload.length).toBe(38);
  });

  it('should wait if packet is incomplete', () => {
    const extractor = new NdtpPacketExtractor();

    const packet = Buffer.from(
      '7e7e26000000c5c40240e201000000010065000100020000000000a463b66a4090781660c63a2180005e0190012823640096000a14',
      'hex',
    );

    const firstPart = packet.subarray(0, 20);
    const secondPart = packet.subarray(20);

    expect(extractor.push(firstPart)).toHaveLength(0);

    const result = extractor.push(secondPart);

    expect(result).toHaveLength(1);
    expect(result[0].raw.length).toBe(53);
  });
});