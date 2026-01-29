import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class to manage all environment variables"""
    
    # MinIO/S3 Settings
    MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'http://minio:9000')
    MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
    MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin123')
    S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'nyc-taxi')
    
    # Spark Settings
    SPARK_DRIVER_MEMORY = os.getenv('SPARK_DRIVER_MEMORY', '3g')
    SPARK_EXECUTOR_MEMORY = os.getenv('SPARK_EXECUTOR_MEMORY', '3g')
    SPARK_EXECUTOR_INSTANCES= os.getenv('SPARK_EXECUTOR_INSTANCES', 3)
    SPARK_EXECUTOR_CORES = os.getenv('SPARK_EXECUTOR_CORES', 2)

    
    # Logging Settings
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE_NAME = os.getenv('LOG_FILE_NAME', 'eda.log')
    
    # Application Settings
    APP_NAME = os.getenv('APP_NAME', 'NYC Taxi EDA')
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
    
    @classmethod
    def display_config(cls):
        """Display current configuration (hide sensitive info)"""
        print("=" * 50)
        print("Current Configuration:")
        print("=" * 50)
        print(f"App Name: {cls.APP_NAME}")
        print(f"Environment: {cls.ENVIRONMENT}")
        print(f"MinIO Endpoint: {cls.MINIO_ENDPOINT}")
        print(f"S3 Bucket: {cls.S3_BUCKET_NAME}")
        print(f"Spark Driver Memory: {cls.SPARK_DRIVER_MEMORY}")
        print(f"Spark Executor Memory: {cls.SPARK_EXECUTOR_MEMORY}")
        print(f"Spark Executor Instances: {cls.SPARK_EXECUTOR_INSTANCES}")
        print(f"Spark Executor Cores: {cls.SPARK_EXECUTOR_CORES}")
        print(f"Log Level: {cls.LOG_LEVEL}")
        print(f"Log File: {cls.LOG_FILE_NAME}")
        print("=" * 50)
