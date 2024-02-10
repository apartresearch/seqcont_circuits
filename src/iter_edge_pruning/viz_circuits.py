from graphviz import Digraph

def plot_graph_adjacency_qkv(head_to_head_adjList, mlp_to_mlp_adjList, head_to_mlp_adjList,
                             mlp_to_head_adjList, heads_to_resid, mlps_to_resid,
                             filename="circuit_graph", highlighted_nodes=None):
    dot = Digraph()
    dot.attr(ranksep='0.45', nodesep='0.11')  # vert height- ranksep, nodesep- w

    dot.node('resid_post', color="#ffcccb", style='filled')

    for node in mlp_to_mlp_adjList.keys():
        sender_name = "MLP " + str(node)
        dot.node(sender_name, color="#ffcccb", style='filled')

    for node in head_to_head_adjList.keys():
        sender_name = f"{node[0]} , {node[1]} {node[2]}"
        dot.node(sender_name, color="#ffcccb", style='filled')
        sender_name = f"{node[0]} , {node[1]}"
        dot.node(sender_name, color="#ffcccb", style='filled')

    edges_added = []
    # for every q k v node, plot an edge to output node
    for node in head_to_head_adjList.keys():
        sender_name = f"{node[0]} , {node[1]} {node[2]}"
        receiver_name = f"{node[0]} , {node[1]}"
        dot.edge(sender_name, receiver_name, color = 'red')
        edges_added.append((sender_name, receiver_name))

    for node in mlp_to_head_adjList.keys():
        sender_name = f"{node[0]} , {node[1]} {node[2]}"
        dot.node(sender_name, color="#ffcccb", style='filled')
        sender_name = f"{node[0]} , {node[1]}"
        dot.node(sender_name, color="#ffcccb", style='filled')

    # for every q k v node, plot an edge to output node
    for node in mlp_to_head_adjList.keys():
        sender_name = f"{node[0]} , {node[1]} {node[2]}"
        receiver_name = f"{node[0]} , {node[1]}"
        if (sender_name, receiver_name) not in edges_added:
            dot.edge(sender_name, receiver_name, color = 'red')

    def loop_adjList(adjList):
        for end_node, start_nodes_list in adjList.items():
            if isinstance(end_node, int):
                receiver_name = "MLP " + str(end_node)
            elif isinstance(end_node, tuple):
                if len(end_node) == 3:
                    receiver_name = f"{end_node[0]} , {end_node[1]} {end_node[2]}"
                elif len(end_node) == 2:
                    receiver_name = f"{end_node[0]} , {end_node[1]}"
            else:
                receiver_name = 'resid_post'
            for start in start_nodes_list:
                if isinstance(start, int):
                    sender_name = "MLP " + str(start)
                elif isinstance(start, tuple):
                    if len(start) == 3:
                        sender_name = f"{start[0]} , {start[1]} {start[2]}"
                    elif len(start) == 2:
                        sender_name = f"{start[0]} , {start[1]}"
                dot.node(sender_name, color="#ffcccb", style='filled')
                dot.node(receiver_name, color="#ffcccb", style='filled')
                dot.edge(sender_name, receiver_name, color = 'red')

    loop_adjList(head_to_head_adjList)
    loop_adjList(mlp_to_mlp_adjList)
    loop_adjList(head_to_mlp_adjList)
    loop_adjList(mlp_to_head_adjList)
    loop_adjList(heads_to_resid)
    loop_adjList(mlps_to_resid)

    # Save the graph to a file
    dot.format = 'png'  # You can change this to 'pdf', 'png', etc. based on your needs
    dot.render(filename)

def plot_graph_adjacency(head_to_head_adjList, mlp_to_mlp_adjList, head_to_mlp_adjList,
                             mlp_to_head_adjList, heads_to_resid, mlps_to_resid,
                             filename="circuit_graph", highlighted_nodes=None):
    dot = Digraph()
    dot.attr(ranksep='0.45', nodesep='0.11')  # vert height- ranksep, nodesep- w

    def loop_adjList(adjList):
        for end_node, start_nodes_list in adjList.items():
            if isinstance(end_node, int):
                receiver_name = "MLP " + str(end_node)
            elif isinstance(end_node, tuple):
                receiver_name = f"{end_node[0]} , {end_node[1]}"
            else:
                receiver_name = 'resid_post'
            for start in start_nodes_list:
                if isinstance(start, int):
                    sender_name = "MLP " + str(start)
                elif isinstance(start, tuple):
                    sender_name = f"{start[0]} , {start[1]}"
                dot.node(sender_name, color="#ffcccb", style='filled')
                dot.node(receiver_name, color="#ffcccb", style='filled')
                dot.edge(sender_name, receiver_name, color = 'red')

    loop_adjList(head_to_head_adjList)
    loop_adjList(mlp_to_mlp_adjList)
    loop_adjList(head_to_mlp_adjList)
    loop_adjList(mlp_to_head_adjList)
    loop_adjList(heads_to_resid)
    loop_adjList(mlps_to_resid)

    # Save the graph to a file
    dot.format = 'png'  # You can change this to 'pdf', 'png', etc. based on your needs
    dot.render(filename)