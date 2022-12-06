## Azure Databricks Online Feature Store
### Sample Implementation

<hr/>

<img src="https://github.com/rafaelvp-db/databricks-online-feature-store/blob/master/img/arch.png?raw=true" />

### Overview

* This repo presents a sample implementation of **Azure Databricks Online Feature Store**, containing two scenarios:
* Model Deployment with [Serverless Realtime Inference](https://docs.databricks.com/mlflow/serverless-real-time-inference.html): **automatic feature lookup**
* Model Deployment without Serverless Realtime Inference (e.g. on AKS): **manual feature lookup** using [Azure CosmosDB Python SDK](https://learn.microsoft.com/nl-nl/azure/cosmos-db/nosql/sdk-python)

### Additional Details

* Terraform is used to automate the process for:
    * Setting up a CosmosDB database;
    * Storing the URI, Read Key and Read Write Keys as secrets in a Databricks backed secret scope


### Reference

* [Databricks Feature Store](https://docs.databricks.com/machine-learning/feature-store/index.html)
* [Work with online stores](https://docs.databricks.com/machine-learning/feature-store/online-feature-stores.html)