from azure.cosmos import CosmosClient
import os

def test_get_feature():

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
    
    assert len(list(items)) > 0
