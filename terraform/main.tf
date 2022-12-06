provider "azurerm" {
   features {}
}

resource "azurerm_resource_group" "rsg" {
    name     = "rsg-online-fs-example2"
    location = "West Europe"
}

module "cosmosdb" {
    source              = "./modules/cosmos-db"
    depends_on          = [azurerm_resource_group.rsg]
    resource_group_name = azurerm_resource_group.rsg.name
}

module "databricks_secrets" {
    source                          = "./modules/databricks-secrets"
    cosmosdb_account_name           = module.cosmosdb.name
    cosmosdb_read_primary_key       = module.cosmosdb.primary_key
    cosmosdb_read_write_primary_key = module.cosmosdb.primary_key
}