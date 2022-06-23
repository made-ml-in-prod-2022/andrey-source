from datetime import timedelta
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

default_args = {
    'owner': 'airflow',
    'email': ['airflow@example.com'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
        dag_id='01_generate_data',
        default_args=default_args,
        description='DAG для создания искусственных данных',
        schedule_interval='@daily',
        start_date=days_ago(1),
) as dag:
    generator = DockerOperator(
        image='airflow-generate-inference-data',
        task_id='Generate_dataset',
        network_mode='bridge',
        do_xcom_push=False,
        mounts=[
            Mount(
                source='/home/andrew/park/ML_prod/airflow/data',
                target='/data',
                type='bind',
            )],
        command='--inference_path /data/inference_data/{{ ds }}',
        mount_tmp_dir=False,
    )

    end_dag = DummyOperator(
        task_id='Finish'
    )

    generator >> end_dag
