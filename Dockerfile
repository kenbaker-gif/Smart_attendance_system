FROM nginx:alpine

COPY index.html /usr/share/nginx/html/index.html
COPY profile.jpg /usr/share/nginx/html/profile.jpg
COPY nginx.conf /etc/nginx/templates/default.conf.template

EXPOSE $PORT