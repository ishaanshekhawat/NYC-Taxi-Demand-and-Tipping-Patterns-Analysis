# NYC Taxi Demand & Tipping Patterns — Project Plan & Requirements

## 1. Project overview

Analyze NYC Yellow Taxi trip records to understand spatial/temporal demand patterns and tipping behavior. Store raw and processed data in a local S3-compatible bucket (MinIO) inside Docker. Run notebooks and PySpark inside the same container stack to ingest, process, analyze, visualize, and model demand and tip prediction.

## 2. Primary objectives

* Ingest multi-month/year NYC taxi trip data into a local S3 (MinIO).

* Clean & preprocess large Parquet files efficiently with PySpark.

* **Exploratory Data Analysis (EDA):** Daily/weekly/hourly demand, geographic hot-spots, trip distance/time distributions, and tip distributions.

* **Tipping patterns:** Correlate tip amount / tip percentage with fare, payment method, time-of-day, day-of-week, pickup/dropoff location, passenger count.

* **Demand forecasting:** Time-series forecast (per-zone or aggregated) to predict short-term demand.

* **Tip prediction (optional ML):** Predict likelihood and amount/percentage of tipping for a trip.

* **Deliverables:** Notebooks, reproducible Docker setup, sample dashboards/visualizations (Plotly/Matplotlib/nbviews), and documentation.

## 3. Scope (what’s included/optional)

**Included:**

* NYC Yellow taxi dataset ingestion.
* Local S3 (MinIO) for object storage.
* PySpark ETL & analysis in Jupyter notebooks.
* EDA visualizations, basic forecasting, and tip analysis.

**Optional:**

* Production deployment on cloud (S3, EMR).
* Real-time streaming ingestion (can be added later through Apache Kafka).
* Complex deep-learning models (out of scope unless requested).

## 4. High-level architecture

**Docker Compose services:**

* Minio (S3-compatible object store)
* Jupyter (JupyterLab or Notebook with PySpark + required libs)
* (Optional) spark-master / spark-worker if using a standalone Spark cluster inside Docker

**Data flow:**

* Place Parquet files locally or upload to MinIO.
* PySpark in Jupyter reads CSVs from S3 endpoints (MinIO).
* Write processed results back to S3 for persistence.
* Visualizations are generated in notebooks or through Tableau.

## 5. Deliverables

* Dockerized environment (docker-compose.yml, Dockerfile).
* Jupyter notebooks for each process.
* Processed parquet files in S3 layout.
* README with setup instructions and how to run notebooks.
* Short technical report (2–4 pages) summarizing findings, key charts, model performance.
* Optional: small interactive dashboard (Streamlit) reading outputs from S3.
