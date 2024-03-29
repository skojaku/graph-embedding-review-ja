import pandas as pd
import numpy as np
from scipy import sparse
import networkx as nx
import country_converter as coco


def load_airport():
    node_table = pd.read_csv(
        "http://opsahl.co.uk/tnet/datasets/openflights_airports.txt", sep=" ",
    )

    edge_table = pd.read_csv(
        "http://opsahl.co.uk/tnet/datasets/openflights.txt",
        sep=" ",
        header=None,
        names=["source", "target", "weight"],
    )

    regional_code = pd.read_csv(
        "https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv"
    )

    # Country name resolution

    cc = coco.CountryConverter()
    node_table["alpha-2"] = np.nan
    node_table["alpha-3"] = np.nan
    rename_country = {"Perú": "Peru"}
    for i, row in node_table.iterrows():
        country = row["Country"]
        country = rename_country.get(country, country)
        iso2 = cc.convert(names=country, to="ISO2")
        if iso2 == "not found":
            continue

        iso3 = cc.convert(names=country, to="ISO3")

        node_table.loc[i, "alpha-2"] = iso2
        node_table.loc[i, "alpha-3"] = iso3
    node_table = pd.merge(
        node_table, regional_code, left_on="alpha-3", right_on="alpha-3", how="left"
    )
    uids, edges = np.unique(
        edge_table[["source", "target"]].values.reshape(-1), return_inverse=True
    )
    edge_table[["source", "target"]] = edges.reshape((edge_table.shape[0], 2))

    node_table = pd.merge(
        pd.DataFrame({"id": np.arange(uids.size), "Airport ID": uids}),
        node_table,
        left_on="Airport ID",
        right_on="Airport ID",
        how="left",
    )
    num_nodes = node_table.shape[0]
    net = sparse.csr_matrix(
        (edge_table.weight, (edge_table.source, edge_table.target)),
        shape=(num_nodes, num_nodes),
    )
    net = net + net.T
    net.data = np.ones_like(net.data)

    net = nx.from_scipy_sparse_matrix(net)
    largest_cc = max(nx.connected_components(net), key=len)
    s = net.subgraph(largest_cc)
    net = nx.adjacency_matrix(s)

    node_table = node_table.loc[
        np.isin(node_table["id"].values, np.array(list(largest_cc))), :
    ]

    # Add network stats
    node_table["deg"] = np.array(net.sum(axis=0)).reshape(-1)

    return node_table, net
