ARG HALO_IMAGE=public.ecr.aws/whitecircle/halo:blackwell-1.0.0
FROM ${HALO_IMAGE}

# Use the upstream CUDA/Python stack, but pin the exact trainer API we integrate.
ARG HALO_REV=0bc3a22a56fb5a7a422b1d4f711d92a978a990bd
RUN git init /opt/halo \
    && git -C /opt/halo remote add origin https://github.com/whitecircle/halo.git \
    && git -C /opt/halo fetch --depth 1 origin "${HALO_REV}" \
    && git -C /opt/halo checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/halo rev-parse HEAD)" = "${HALO_REV}" \
    && python -m pip install --no-deps --no-build-isolation -e /opt/halo

ENV PYTHONPATH=/opt/halo:/app \
    HF_HOME=/cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY . /app
# Catch a stale/incompatible base image before a GPU is reserved for the run.
RUN python -c "from src.configs.offline_grpo_config import OfflineGRPOConfig; from src.trainers.grpo.offline import OfflineGRPOTrainer"
ENTRYPOINT ["python", "-m", "halo_demo"]
CMD ["train"]
