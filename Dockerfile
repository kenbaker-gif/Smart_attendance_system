FROM nginx:alpine

COPY index.html /usr/share/nginx/html/index.html
COPY profile.jpg /usr/share/nginx/html/profile.jpg
COPY favicon.ico /usr/share/nginx/html/favicon.ico
COPY favicon.png /usr/share/nginx/html/favicon.png
COPY nginx.conf /etc/nginx/templates/default.conf.template

EXPOSE $PORT