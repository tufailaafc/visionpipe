FROM nvcr.io/nvidia/deepstream:7.1-gc-triton-devel

# Set working directory
WORKDIR /app

# Required for building custom YOLO parser
RUN apt-get update && \
    apt-get install -y build-essential python3-pip wget git curl && \
    pip3 install --upgrade pip && \
    pip3 install onnx onnxruntime onnx-simplifier

# Clone and install Ultralytics repo
RUN git clone https://github.com/ultralytics/ultralytics.git && \
    pip3 install -e ./ultralytics


# Clone DeepStream-YOLO repo
RUN git clone https://github.com/marcoslucianops/DeepStream-Yolo.git

# Patch Makefile to replace spaces with tabs (for safe compilation)
RUN sed -i 's/^\( \+\)/\t/' DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/Makefile

# Compile the custom parser with CUDA version matching DeepStream 7.1
ENV CUDA_VER=12.6
RUN make -C DeepStream-Yolo/nvdsinfer_custom_impl_Yolo clean CUDA_VER=${CUDA_VER} && \
    make -C DeepStream-Yolo/nvdsinfer_custom_impl_Yolo CUDA_VER=${CUDA_VER}


# Builds packages to help gstream process different media files
RUN /opt/nvidia/deepstream/deepstream/user_additional_install.sh

RUN apt-get -y install --reinstall libflac8 libmp3lame0 libxvidcore4 ffmpeg


RUN apt-get update
RUN apt install -y python3-gi python3-dev python3-gst-1.0 python-gi-dev git meson \
    python3 python3-pip python3.10-dev cmake g++ build-essential libglib2.0-dev \
    libglib2.0-dev-bin libgstreamer1.0-dev libtool m4 autoconf automake libgirepository1.0-dev libcairo2-dev



RUN pip3 install build



# Install CUDA Python bindings
RUN pip install --no-cache-dir nvidia-pyindex && \
    pip install --no-cache-dir cuda-python

RUN /opt/nvidia/deepstream/deepstream/update_rtpmanager.sh

RUN apt-get install libgirepository1.0-dev
RUN apt-get install gobject-introspection gir1.2-gst-rtsp-server-1.0

#Install a manager so we can run both the api and deepstream
RUN apt-get update && apt-get install -y supervisor

# For inspecting traffic inside container for debugging
RUN apt install -y net-tools


# Install pip requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir --default-timeout=100 --retries=10 -r requirements.txt



# Copy files project
COPY . /app/
RUN chmod +x /app/entrypoint.sh


# Expose port (optional for REST interface or API layer)
# 8701 is for the app and 8554 for the rtsp stream
EXPOSE 8701
EXPOSE 8554

# Entry point: Run DeepStream app
# CMD ["deepstream-app", "-c", "deepstream/deepstream_app_config.txt"]

#CMD ["uvicorn", "deepstream:app", "--host", "0.0.0.0", "--port", "8701", "--log-level", "info"]
#CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8701"]
#ENTRYPOINT ["./entrypoint.sh"]
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
ENTRYPOINT ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
