import { describe, expect, it, vi } from 'vitest';

import { TripMatcherService } from '../src/prediction/trip-matcher.service.js';

describe('TripMatcherService', () => {
  it('excludes synthetic Group-B trip IDs from nearest-neighbor matching', async () => {
    const queryRaw = vi.fn(async (query: unknown) => {
      const sql = String(query);
      expect(sql).toContain("tr_id NOT LIKE '9000%'");
      return [
        {
          trId: '131672',
          targetActionId: '53700172828',
          distanceMeters: 12,
        },
      ];
    });

    const prisma = {
      vehicles: {
        findUnique: vi.fn().mockResolvedValue(null),
      },
      $queryRaw: queryRaw,
    } as any;

    const service = new TripMatcherService(prisma);
    const result = await service.findTrip(55.7512, 37.6184);

    expect(result).toEqual({
      trId: '131672',
      targetActionId: '53700172828',
      distanceMeters: 12,
    });
  });

  it('matches only trips that have a valid target action in the required 10–15 minute horizon', async () => {
    const queryRaw = vi.fn(async (query: unknown) => {
      const sql = String(query);
      expect(sql).toContain('time_begin >');
      expect(sql).toContain('time_begin <=');
      expect(sql).toContain('valid_trips');
      return [
        {
          trId: '131672',
          targetActionId: '53700172828',
          distanceMeters: 12,
        },
      ];
    });

    const prisma = {
      vehicles: {
        findUnique: vi.fn().mockResolvedValue(null),
      },
      $queryRaw: queryRaw,
    } as any;

    const service = new TripMatcherService(prisma);
    const result = await service.findTrip(
      55.7512,
      37.6184,
      undefined,
      new Date('2026-01-06T03:35:00.000Z'),
    );

    expect(result).toEqual({
      trId: '131672',
      targetActionId: '53700172828',
      distanceMeters: 12,
    });
  });

  it('does not fall back to another trip when a preferred vehicle trip is missing or out of range', async () => {
    const queryRaw = vi.fn().mockResolvedValue([]);

    const prisma = {
      vehicles: {
        findUnique: vi.fn().mockResolvedValue({ current_tr_id: '131672' }),
      },
      $queryRaw: queryRaw,
    } as any;

    const service = new TripMatcherService(prisma);
    const result = await service.findTrip(
      55.7512,
      37.6184,
      '233',
      new Date('2026-01-06T03:35:00.000Z'),
    );

    expect(result).toBeNull();
    expect(queryRaw).toHaveBeenCalledTimes(1);
  });
});
