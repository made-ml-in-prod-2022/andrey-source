from datetime import timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

default_args = {
    'owner': 'airflow',
    'email': ['airflow@example.com'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
        '02_train_model',
        default_args=default_args,
        description='DAG пайплайна для обучения модели',
        schedule_interval="@weekly",
        start_date=days_ago(2),
) as dag:

    generator = DockerOperator(
        task_id='Download_data',
        image='airflow-generate-train-data',
        network_mode='bridge',
        do_xcom_push=False,
        mounts=[
            Mount(
                source='/home/andrew/park/ML_prod/airflow/data',
                target='/data',
                type='bind',
            )],
        command='--train_path /data/train_data/{{ ds }}',
    )

    preprocessing = DockerOperator(
        task_id='Preprocessing',
        image='airflow-preprocessing',
        network_mode='bridge',
        do_xcom_push=False,
        mounts=[
            Mount(
                source='/home/andrew/park/ML_prod/airflow/data',
                target='/data',
                type='bind',
            )],
        command='--train_path /data/train_data/{{ ds }}',
    )

    split = DockerOperator(
        task_id='Train_test_split',
        image='airflow-split',
        network_mode="bridge",
        do_xcom_push=False,
        mounts=[
            Mount(
                source='/home/andrew/park/ML_prod/airflow/data',
                target='/data',
                type='bind',
            )],
        command='--train_path /data/train_data/{{ ds }} '
                '--test_size 0.15 '
                '--random_state 42',
    )

    train = DockerOperator(
        task_id='Train_model',
        image='airflow-train',
        network_mode="bridge",
        do_xcom_push=False,
        mounts=[
            Mount(
                source='/home/andrew/park/ML_prod/airflow/data',
                target='/data',
                type='bind',
            )],
        command='--train_path /data/train_data/{{ ds }} '
                '--model_path /data/models/{{ ds }}',
    )

    validation = DockerOperator(
        task_id='Validate_model',
        image='airflow-validation',
        network_mode='bridge',
        do_xcom_push=False,
        mounts=[
            Mount(
                source='/home/andrew/park/ML_prod/airflow/data',
                target='/data',
                type='bind',
            )],
        command='--train_path /data/train_data/{{ ds }} '
                '--metrics_path /data/metrics/{{ ds }} '
                '--model_path /data/models/{{ ds }}',
    )

    generator >> preprocessing >> split >> train >> validation
