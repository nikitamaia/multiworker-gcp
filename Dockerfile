# Specifies base image and tag
FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-4
WORKDIR /root

# Copies the trainer code to the docker image.
COPY trainer/ /root/trainer/

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.task"]
