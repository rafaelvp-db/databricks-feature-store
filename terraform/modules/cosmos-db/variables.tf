variable "resource_group_name" {
  type = string
}

variable "cosmosdb_account_name" {
  type = string
  default = "online-fs-cosmosdb-account2"
}

variable "location" {
  type = string
  default = "West Europe"
}