FROM python:3.9-slim

WORKDIR /app

COPY BESSBatteryEnv.py train_ppo.py evaluate_ppo.py Snakefile ppo_pipeline.ipynb ./

RUN pip install --no-cache-dir stable-baselines3 gym pandas numpy matplotlib shimmy mlflow prometheus_client snakemake

CMD ["snakemake", "-s", "Snakefile", "--cores", "1"]