# ========= STAGE 1: build =========
FROM node:20-alpine AS build

WORKDIR /app

# Copiamos solo manifests (mejor cache)
COPY package.json package-lock.json ./

# Instalación reproducible
RUN npm ci

# Copiamos el resto
COPY . .

# Build de producción
RUN npm run build


# ========= STAGE 2: runtime =========
FROM nginx:alpine

# Limpiamos config default
RUN rm /etc/nginx/conf.d/default.conf

# Copiamos build
COPY --from=build /app/dist /usr/share/nginx/html

# Config mínima nginx
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
