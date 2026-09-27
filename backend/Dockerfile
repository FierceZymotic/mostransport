# Backend (NestJS): NDTP TCP receiver (:9000) + REST/WebSocket API (:3000).
#
#   docker build -t mostransport-backend ./backend
#   docker run --rm -p 3000:3000 -p 9000:9000 \
#     -e ML_SERVICE_URL=http://ml:8000 \
#     mostransport-backend

FROM node:22-alpine AS build
WORKDIR /app

COPY package*.json ./
RUN npm install --ignore-scripts

COPY . .
RUN npx prisma generate
RUN npm run build

FROM node:22-alpine AS runtime
WORKDIR /app
ENV NODE_ENV=production

COPY package*.json ./
RUN npm install --omit=dev --ignore-scripts

COPY --from=build /app/dist ./dist

RUN addgroup -S app && adduser -S app -G app
USER app

# 3000 = HTTP/WebSocket API for frontend, 9000 = NDTP TCP telemetry ingest.
EXPOSE 3000 9000

CMD ["node", "dist/main.js"]
