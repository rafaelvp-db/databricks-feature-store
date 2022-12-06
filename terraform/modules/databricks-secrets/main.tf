terraform {
  required_providers {
    databricks = {
      source  = "databricks/databricks"
    }
  }
}

data "databricks_current_user" "me" {}

resource "databricks_secret_scope" "read" {
  name = "fs-read-scope2"
}

resource "databricks_secret" "read" {
  key          = "${var.cosmosdb_account_name}-authorization-key"
  string_value = var.cosmosdb_read_primary_key
  scope        = databricks_secret_scope.read.id
}

resource "databricks_secret_scope" "write" {
  name = "fs-write-scope2"
}

resource "databricks_secret" "write" {
  key          = "${var.cosmosdb_account_name}-authorization-key"
  string_value = var.cosmosdb_read_write_primary_key
  scope        = databricks_secret_scope.write.id
}