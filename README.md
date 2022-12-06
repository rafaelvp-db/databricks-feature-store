## Azure Databricks Online Feature Store
### Sample Implementation

<hr/>

<img src="https://github.com/rafaelvp-db/databricks-online-feature-store/blob/master/img/arch.png?raw=true" />

### Overview

* This repo presents a sample implementation of **Azure Databricks Online Feature Store**, containing two scenarios:
* Model Deployment with Serverless Realtime Inference: automatic feature lookup
* Model Deployment without Serverless Realtime Inference (e.g. on AKS): manual feature lookup using Azure CosmosDB Python SDK

### Additional Details

* Terraform is used to automate the process for:
    * Setting up a CosmosDB database;
    * Storing the URI, Read Key and Read Write Keys as secrets in a Databricks backed secret scope