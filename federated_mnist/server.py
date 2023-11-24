import flwr as fl

num_clients = 10
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1,  # Sample 10% of available clients for the next round
    min_fit_clients=num_clients,  # Minimum number of clients to be sampled for the next round
    min_available_clients=num_clients,  # Minimum number of clients that need to be connected to the server before a training round can start
    min_eval_clients=num_clients
)

fl.server.start_server(config={"num_rounds": 100}, strategy=strategy)