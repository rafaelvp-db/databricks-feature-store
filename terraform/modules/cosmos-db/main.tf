resource "azurerm_cosmosdb_account" "fs" {
  name                = var.cosmosdb_account_name
  resource_group_name = var.resource_group_name
  location            = var.location
  offer_type          = "Standard"

  geo_location {
    location          = "westus"
    failover_priority = 0
  }

  consistency_policy {
    consistency_level       = "BoundedStaleness"
    max_interval_in_seconds = 300
    max_staleness_prefix    = 100000
  }
}