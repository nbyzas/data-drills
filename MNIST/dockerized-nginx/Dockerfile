FROM nginx:1.23.0

RUN rm /etc/nginx/nginx.conf

COPY nginx.conf /etc/nginx

# RUN mkdir /etc/nginx/sites-enabled # not needed thanks to trailing slash

COPY mlflow.conf /etc/nginx/sites-enabled/

