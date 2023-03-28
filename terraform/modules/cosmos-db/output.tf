output "endpoint" {
    value     = azurerm_cosmosdb_account.fs.endpoint
}

output "name" {
    value     = azurerm_cosmosdb_account.fs.name
}

output "primary_key" {
    value     = azurerm_cosmosdb_account.fs.primary_key
    sensitive = true
}