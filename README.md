## Azure Databricks Online Feature Store
### Sample Implementation

<hr/>

<img src="https://github.com/rafaelvp-db/databricks-online-feature-store/blob/master/img/arch.png?raw=true" />

### Overview

* This repo presents a sample implementation of **Azure Databricks Online Feature Store**, containing two scenarios:
* Model Deployment with [Serverless Realtime Inference](https://docs.databricks.com/mlflow/serverless-real-time-inference.html): **automatic feature lookup**
* Model Deployment without Serverless Realtime Inference (e.g. on AKS): **manual feature lookup** using [Azure CosmosDB Python SDK](https://learn.microsoft.com/nl-nl/azure/cosmos-db/nosql/sdk-python)

#### Example: Manual Feature Lookup

```python
URL = os.environ['ACCOUNT_URI']
KEY = os.environ['ACCOUNT_KEY']
client = CosmosClient(URL, credential=KEY)
DATABASE_NAME = 'wine_db'
database = client.get_database_client(DATABASE_NAME)
CONTAINER_NAME = 'feature_store_online_wine_features'
container = database.get_container_client(CONTAINER_NAME)

# Enumerate the returned items
import json
items = container.query_items(
    query='SELECT TOP 1 * FROM feature_store_online_wine_features',
    enable_cross_partition_query=True
)
```

### Additional Details

* Terraform is used to automate the process for:
    * Setting up a CosmosDB database;
    * Storing the URI, Read Key and Read Write Keys as secrets in a Databricks backed secret scope


### Reference

* [Databricks Feature Store](https://docs.databricks.com/machine-learning/feature-store/index.html)
* [Work with online stores](https://docs.databricks.com/machine-learning/feature-store/online-feature-stores.html)