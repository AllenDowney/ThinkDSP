FROM jupyter/scipy-notebook:latest

# Launchbot labels
LABEL name.launchbot.io="ThinkDSP"
LABEL workdir.launchbot.io="/usr/workdir"
LABEL 8888.port.launchbot.io="Jupyter Notebook"

# Set the working directory
USER root
ENV WORKDIR /usr/workdir
WORKDIR ${WORKDIR}
COPY code ${WORKDIR}
RUN chown -R ${NB_USER} ${WORKDIR}
USER ${NB_USER}

# Expose the notebook port
EXPOSE 8888

# Start the notebook server
CMD jupyter notebook --no-browser --port 8888 --ip=*
