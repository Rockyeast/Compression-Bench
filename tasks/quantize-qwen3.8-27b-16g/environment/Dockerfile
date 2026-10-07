FROM quantbench-agent-base:20260930-shared-v1

RUN mkdir -p /app/work /app/submission
COPY check_submission.py model_inventory.py /app/
COPY inference.py submission_guard.py /app/
COPY calibration/ /opt/calibration/
RUN chmod 0555 /app/check_submission.py /app/model_inventory.py \
    && chmod -R a=rX /opt/calibration \
    && python3 /app/model_inventory.py --help >/dev/null
WORKDIR /app
