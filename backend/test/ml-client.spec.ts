import { afterEach, describe, expect, it, vi } from 'vitest';

import { MlClientService } from '../src/prediction/ml/ml.client.service.js';

describe('MlClientService availability propagation', () => {
  afterEach(() => vi.restoreAllMocks());

  it('maps an ML 503 (artifact not loaded, /ready = false) to 503, not 500', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('{"detail":"predictor is not ready"}', { status: 503 }));
    await expect(new MlClientService().predict({} as any)).rejects.toMatchObject({ status: 503 });
  });

  it('maps an unreachable ML service to 503', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new TypeError('fetch failed'));
    await expect(new MlClientService().predict({} as any)).rejects.toMatchObject({ status: 503 });
  });

  it('keeps other ML errors (e.g. 422 contract violation) as 500', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('{"detail":"bad"}', { status: 422 }));
    await expect(new MlClientService().predict({} as any)).rejects.toMatchObject({ status: 500 });
  });
});
