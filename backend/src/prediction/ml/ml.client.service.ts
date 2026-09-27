import {
  Injectable,
  InternalServerErrorException,
  ServiceUnavailableException,
} from '@nestjs/common';
import {
  PredictionRequest,
  PredictionResponse,
} from './ml.types.js';

@Injectable()
export class MlClientService {
  private readonly baseUrl =
    process.env.ML_SERVICE_URL ??
    'http://localhost:8000';

  async predict(
  request: PredictionRequest,
): Promise<PredictionResponse> {
  console.log(
    'ML REQUEST:',
    JSON.stringify(request, null, 2),
  );

  let response: Response;
  try {
    response = await fetch(
      `${this.baseUrl}/api/v1/predict`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      },
    );
  } catch (error) {
    // ML container not reachable (e.g. still starting): unavailable, not an internal error.
    throw new ServiceUnavailableException(
      `ML service unreachable: ${error instanceof Error ? error.message : String(error)}`,
    );
  }

  

    if (!response.ok) {
      const body = await response.text();

      if (response.status === 503) {
        // ML is up but has no loaded, compatible artifact (/ready = false).
        throw new ServiceUnavailableException(`ML service not ready: ${body}`);
      }

      throw new InternalServerErrorException(
        `ML service returned ${response.status}: ${body}`,
      );
    }

    return (await response.json()) as PredictionResponse;
  }
}