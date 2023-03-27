# base image
# use FROM pytorch/torchserve:latest-gpu for GPU support
FROM pytorch/torchserve:latest-gpu 

# install dependencies
RUN pip3 install transformers

# copy model archive
COPY model_store model_store

COPY config.properties config.properties

COPY entrypoints.sh /home/entrypoints.sh
# make entrypoint.sh executable before copying it

ENTRYPOINT ["/home/entrypoints.sh"]
CMD ["serve"]