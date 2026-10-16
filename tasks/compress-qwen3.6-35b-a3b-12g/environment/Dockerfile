FROM quantbench-agent-base:local
ENV QUANT_BENCH_DEBUG_9B=1
ENV QUANT_BENCH_CHECKER_MAX_RESPONSE_BYTES=67108864

RUN mkdir -p /app/work /app/submission
COPY inference.py /app/
COPY calibration/ /opt/calibration/
RUN chmod 0555 /app/check_submission.py /app/model_inventory.py \
    && chmod -R a=rX /opt/calibration \
    && python3 /app/model_inventory.py --help >/dev/null
WORKDIR /app

RUN python3 /opt/quantbench-runtime/verify_runtime.py
