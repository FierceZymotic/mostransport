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
    // The prediction time is explicit (no hidden default instant; see the last test).
    const result = await service.findTrip(55.7512, 37.6184, undefined, new Date('2026-01-06T03:35:00.000Z'));

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

  it('uses the mapped trip first and the adapted prediction time for the horizon', async () => {
    const queryRaw = vi.fn(async (strings: TemplateStringsArray, ...values: unknown[]) => {
      // The mapped-trip restriction is a nested Prisma.sql fragment bound as a value.
      const fragment = values.find((v: any) => v && typeof v === 'object' && Array.isArray(v.strings)) as any;
      expect(fragment.strings.join('?')).toContain('sa.tr_id = ');
      expect(fragment.values).toEqual(['134040']);
      expect(strings.join('?')).toContain("tr_id NOT LIKE '9000%'");
      expect(values).toContainEqual(new Date('2026-01-06T09:52:44.000Z')); // T + 10 min
      expect(values).toContainEqual(new Date('2026-01-06T09:57:44.000Z')); // T + 15 min
      return [{ trId: '134040', targetActionId: '1', distanceMeters: 40 }];
    });
    const prisma = { vehicles: { findUnique: vi.fn().mockResolvedValue({ current_tr_id: '134040' }) }, $queryRaw: queryRaw } as any;
    const result = await new TripMatcherService(prisma).findTrip(55.7, 37.6, '1105498', new Date('2026-01-06T09:42:44.000Z'));
    expect(result?.trId).toBe('134040');
    expect(prisma.vehicles.findUnique).toHaveBeenCalledWith({ where: { unit_id: '1105498' } });
    expect(queryRaw).toHaveBeenCalledTimes(1);
  });

  it('refuses to match without an explicit prediction time (no hidden demo instant)', async () => {
    const prisma = { vehicles: { findUnique: vi.fn() }, $queryRaw: vi.fn() } as any;
    const service = new TripMatcherService(prisma);
    await expect(service.findTrip(55.75, 37.61, undefined, undefined as unknown as Date)).rejects.toThrow(/explicit valid prediction time/);
    expect(prisma.$queryRaw).not.toHaveBeenCalled();
  });
});
