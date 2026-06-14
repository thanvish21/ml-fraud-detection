FROM node:20-alpine AS build
WORKDIR /app
COPY dashboard/package.json /app/package.json
RUN npm install --no-audit --no-fund
COPY dashboard/ /app/
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY dashboard/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
