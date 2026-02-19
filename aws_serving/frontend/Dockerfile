FROM nginx:alpine

# Remove default nginx page
RUN rm -rf /usr/share/nginx/html/*

# Copy custom nginx config template
COPY nginx.conf /etc/nginx/templates/default.conf.template

# Copy frontend files
COPY index.html /usr/share/nginx/html/

# Environment variables with defaults
ENV BACKEND_HOST=backend
ENV BACKEND_PORT=5000

EXPOSE 80
