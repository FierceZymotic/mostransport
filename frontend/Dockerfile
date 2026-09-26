# Frontend (Vite/React) dispatcher dashboard, served as a static build via nginx.
#
# VITE_* variables are baked in at BUILD time (Vite convention), not at
# container start. Pass the real key as a build-arg:
#
#   docker build -t mostransport-frontend ./frontend \
#     --build-arg VITE_YANDEX_MAPS_API_KEY=xxxx
#   docker run --rm -p 8080:80 mostransport-frontend

FROM node:22-alpine AS build
WORKDIR /app

ARG VITE_YANDEX_MAPS_API_KEY
ENV VITE_YANDEX_MAPS_API_KEY=${VITE_YANDEX_MAPS_API_KEY}

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:1.27-alpine AS runtime
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
