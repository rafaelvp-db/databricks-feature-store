terraform {
  required_providers {
    databricks = {
      source  = "databricks/databricks"
    }
  }
}

data "databricks_current_user" "me" {}

resource "databricks_secret_scope" "cosmos" {
  name = "fs-cosmos-db"
}

resource "databricks_secret" "endpoint" {
  key          = "cosmos-endpoint"
  string_value = var.cosmosdb_endpoint
  scope        = databricks_secret_scope.cosmos.id
}
resource "databricks_secret_scope" "read" {
  name = "fs-read-scope2"
}

resource "databricks_secret" "read" {
  key          = "cosmos-authorization-key"
  string_value = var.cosmosdb_read_primary_key
  scope        = databricks_secret_scope.read.id
}

resource "databricks_secret_scope" "write" {
  name = "fs-write-scope2"
}

resource "databricks_secret" "write" {
  key          = "cosmos-authorization-key"
  string_value = var.cosmosdb_read_write_primary_key
  scope        = databricks_secret_scope.write.id
}