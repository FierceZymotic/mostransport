import {
  Injectable,
  InternalServerErrorException,
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
    const response = await fetch(
      `${this.baseUrl}/api/v1/predict`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      },
    );

    if (!response.ok) {
      const body = await response.text();

      throw new InternalServerErrorException(
        `ML service returned ${response.status}: ${body}`,
      );
    }

    return (await response.json()) as PredictionResponse;
  }
}