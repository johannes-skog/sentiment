FROM pytorch/torchserve:latest-gpu 

# copy model archive
COPY model_store model_store

COPY config.properties config.properties

COPY entrypoints.sh /home/entrypoints.sh
# make entrypoint.sh executable before copying it

ENTRYPOINT ["/home/entrypoints.sh"]
CMD ["serve"]