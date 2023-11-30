# Start from a Jupyter base image with Rust installed
FROM jupyter/base-notebook

USER root 

# Install Rust
RUN apt-get update && apt-get install -y curl build-essential \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="${HOME}/.cargo/bin:${PATH}"

# Install evcxr_jupyter
RUN cargo install evcxr_jupyter
RUN evcxr_jupyter --install

# No need for a token / password when starting jupyter 
RUN echo "c.NotebookApp.token    = ''" >> ${HOME}/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.password = ''" >> ${HOME}/.jupyter/jupyter_notebook_config.py


ADD ./ddo_seminar.ipynb /ddo_seminar/ddo_seminar.ipynb

# Set up the working directory
WORKDIR /ddo_seminar/

# Expose the JupyterLab port
EXPOSE 8888

# Start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]