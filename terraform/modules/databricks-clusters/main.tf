terraform {
  required_providers {
    databricks = {
      source  = "databricks/databricks"
    }
  }
}

data "databricks_current_user" "me" {} 

data "databricks_node_type" "smallest" {
  local_disk = true
}

data "databricks_spark_version" "beta_ml" {
  beta = true
  ml = true
  latest = true
}

resource "databricks_cluster" "fs_cluster" {
  cluster_name            = "fs_test"
  spark_version           = data.databricks_spark_version.beta_ml.id
  node_type_id            = data.databricks_node_type.smallest.id
  autotermination_minutes = 30
  autoscale {
    min_workers = 1
    max_workers = 2
  }
}

resource "databricks_library" "cosmos" {
  cluster_id = databricks_cluster.fs_cluster.id
  maven {
    coordinates = "com.azure.cosmos.spark:azure-cosmos-spark_3-3_2-12:4.17.2"
  }
}