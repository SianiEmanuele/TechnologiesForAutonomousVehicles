import math
import os
import osmnx as ox
import random
import heapq

# Constants
EARTH_RADIUS_M = 6371.0
VERBOSE = False # Set to True to enable verbose logging

def style_unvisited_edge(G, edge):
    G.edges[edge]["color"] = "#d36206"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 0.2

def style_visited_edge(G, edge):
    #G.edges[edge]["color"] = "#d36206"
    G.edges[edge]["color"] = "green"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 1

def style_active_edge(G, edge):
    #G.edges[edge]["color"] = '#e8a900'
    G.edges[edge]["color"] = "red"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 1

def style_path_edge(G, edge):
    G.edges[edge]["color"] = "white"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 5

def plot_graph(G):
    ox.plot_graph(
        G,
        node_size =  [ G.nodes[node]["size"] for node in G.nodes ],
        edge_color = [ G.edges[edge]["color"] for edge in G.edges ],
        edge_alpha = [ G.edges[edge]["alpha"] for edge in G.edges ],
        edge_linewidth = [ G.edges[edge]["linewidth"] for edge in G.edges ],
        node_color = "white",
        bgcolor = "#18080e"
    )

def calculate_heuristic(G, node, dest):
    """
    Calculates the heuristic (estimated time) between two nodes using the Haversine formula.

    Args:
        G: The graph object.
        node: The starting node.
        dest: The destination node.

    Returns:
        float: The estimated time in seconds.
    """
    maxspeed = 40 # Average speed in km/h

    # Haversine formula
    phi1 = math.radians(G.nodes[node]["y"])
    phi2 = math.radians(G.nodes[dest]["y"])
    delta_phi = math.radians(G.nodes[dest]["y"] - G.nodes[node]["y"])
    delta_lambda = math.radians(G.nodes[dest]["x"] - G.nodes[node]["x"])
    a = math.sin(delta_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d_m = 6371000.0 * c       # distance in meters
    speed_mps = maxspeed * 1000.0 / 3600.0 # speed in m/s
    return d_m / speed_mps  # time in seconds

def dijkstra(G, orig, dest, plot=False):
    for node in G.nodes:
        G.nodes[node]["dijkstra_visited"] = False
        G.nodes[node]["distance"] = float("inf")
        G.nodes[node]["dijkstra_uses"] = None
        G.nodes[node]["size"] = 0
    for edge in G.edges:
        style_unvisited_edge(G, edge)
    G.nodes[orig]["distance"] = 0
    G.nodes[orig]["size"] = 50
    G.nodes[dest]["size"] = 50
    pq = [(0, orig)]
    step = 0
    while pq:
        _, node = heapq.heappop(pq)
        if node == dest:
            print("Iterations:", step)
            plot_graph(G)
            return step
        if G.nodes[node]["dijkstra_visited"]: continue
        G.nodes[node]["dijkstra_visited"] = True
        for edge in G.out_edges(node):
            style_visited_edge(G, (edge[0], edge[1], 0))
            neighbor = edge[1]
            weight = G.edges[(edge[0], edge[1], 0)]["weight"]
            if G.nodes[neighbor]["distance"] > G.nodes[node]["distance"] + weight:
                G.nodes[neighbor]["distance"] = G.nodes[node]["distance"] + weight
                G.nodes[neighbor]["dijkstra_uses"] = node
                heapq.heappush(pq, (G.nodes[neighbor]["distance"], neighbor))
                for edge2 in G.out_edges(neighbor):
                    style_active_edge(G, (edge2[0], edge2[1], 0))
        step += 1

def reconstruct_path(G, orig, dest, plot=False, algorithm=None):
    for edge in G.edges:
        style_unvisited_edge(G, edge)
    dist = 0
    speeds = []
    curr = dest
    total_weight = 0
    while curr != orig:
        prev = G.nodes[curr][f"{algorithm}_uses"]
        total_weight += G.edges[(prev, curr, 0)]["weight"]
        dist += G.edges[(prev, curr, 0)]["length"]
        speeds.append(G.edges[(prev, curr, 0)]["maxspeed"])
        style_path_edge(G, (prev, curr, 0))
        if algorithm:
            G.edges[(prev, curr, 0)][f"{algorithm}_uses"] = G.edges[(prev, curr, 0)].get(f"{algorithm}_uses", 0) + 1
        curr = prev
    dist /= 1000
    avg_speed = sum(speeds) / len(speeds)
    if VERBOSE:
        print("Distance: ", dist, "km")
        print("Average speed: ", avg_speed, "km/h")
        print("Total weight: ", total_weight, "s")

    return total_weight

def plot_heatmap(G, algorithm):
    edge_colors = ox.plot.get_edge_colors_by_attr(G, f"{algorithm}_uses", cmap="hot")
    fig, _ = ox.plot_graph(
        G,
        node_size =  [ G.nodes[node]["size"] for node in G.nodes ],
        # edge_color = edge_colors,
        edge_color = [ G.edges[edge]["color"] for edge in G.edges ],
        edge_alpha = [ G.edges[edge]["alpha"] for edge in G.edges ],
        edge_linewidth = [ G.edges[edge]["linewidth"] for edge in G.edges ],
        bgcolor = "#18080e"
    )
def a_star(G, orig, dest):
    """
    Implements the A* algorithm to find the shortest path.

    Args:
        G: The graph object.
        orig: The starting node.
        dest: The destination node.

    Returns:
        int: The number of iterations performed.
    """
    # Initialize the graph
    for node in G.nodes:
        G.nodes[node]["astar_visited"] = False
        G.nodes[node]["gScore"] = float("inf")
        G.nodes[node]["fScore"] = float("inf")
        G.nodes[node]["astar_uses"] = None
        G.nodes[node]["size"] = 0

    for edge in G.edges:
        style_unvisited_edge(G, edge)

    # Initialize the starting node
    G.nodes[orig]["gScore"] = 0
    G.nodes[orig]["fScore"] = calculate_heuristic(G, orig, dest)
    G.nodes[orig]["size"] = 50
    G.nodes[dest]["size"] = 50

    # Heap queue for open set
    open_set = [(0, orig)]

    # empty map for visited nodes
    came_from = {}
    step = 0
    # while there are nodes to visit
    while open_set:
        _, current = heapq.heappop(open_set)

        # exit condition
        if current == dest:
            print("Iterations:", step)
            plot_graph(G)
            return step

        # skip if already visited
        if G.nodes[current]["astar_visited"]: continue

        # visit the node
        else:
            # mark as visited
            G.nodes[current]["astar_visited"] = True

            # evaluate neighbors
            for edge in G.out_edges(current):
                style_visited_edge(G, (edge[0], edge[1], 0))
                neighbor = edge[1]
                tentative_gScore = G.nodes[current]["gScore"] + G.edges[(edge[0], edge[1], 0)]["weight"]

                # if the new path estimated time is less than the previous one
                if tentative_gScore < G.nodes[neighbor]["gScore"]:
                    came_from[neighbor] = current
                    G.nodes[neighbor]["gScore"] = tentative_gScore
                    h = calculate_heuristic(G, neighbor, dest)
                    G.nodes[neighbor]["fScore"] = tentative_gScore + h
                    G.nodes[neighbor]["astar_uses"] = current
                    heapq.heappush(open_set, (G.nodes[neighbor]["fScore"], neighbor))

                    # style the edge
                    for edge2 in G.out_edges(neighbor):
                        style_active_edge(G, (edge2[0], edge2[1], 0))
            step += 1
    return step

def execution_loop(place_name, times):
    """
    Executes the pathfinding algorithms multiple times for a given place.

    Args:
        place_name: The name of the place to generate the graph.
        times: The number of iterations to run.

    Saves:
        CSV files with raw and average results.
    """
    distances = []
    dijkstra_weights = []
    dijkstra_iterations = []
    a_star_weights = []
    a_star_iterations = []


    for i in range(times):
        G = ox.graph_from_place(place_name, network_type="drive")

        for edge in G.edges:
            # Cleaning the "maxspeed" attribute, some values are lists, some are strings, some are None
            maxspeed = 40
            if "maxspeed" in G.edges[edge]:
                maxspeed = G.edges[edge]["maxspeed"]
                if type(maxspeed) == list:
                    speeds = [int(speed) for speed in maxspeed]
                    maxspeed = min(speeds)
                elif type(maxspeed) == str:
                    maxspeed = maxspeed.strip(" mph")
                    maxspeed = int(maxspeed)
            G.edges[edge]["maxspeed"] = maxspeed / 3.6  # Convert to m/s
            # Adding the "weight" attribute (time = distance / speed)
            G.edges[edge]["weight"] = G.edges[edge]["length"] / (maxspeed / 3.6)  # Convert to m/s

        for edge in G.edges:
            G.edges[edge]["dijkstra_uses"] = 0
            G.edges[edge]["a_star_uses"] = 0

        start = random.choice(list(G.nodes))
        end = random.choice(list(G.nodes))

        # calculate harvesin distance
        dist = calculate_heuristic(G, start, end)
        distances.append(dist)

        print("Running Dijkstra | ", i + 1)
        dijkstra_steps = dijkstra(G, start, end, plot=True)
        dijkstra_iterations.append(dijkstra_steps)
        dijk_weight = reconstruct_path(G, start, end, algorithm="dijkstra", plot=True)
        dijkstra_weights.append(dijk_weight)
        plot_heatmap(G, "dijkstra")
        print("Done\n")

        print("Running a_star | ", i + 1)
        as_steps = a_star(G, start, end)
        a_star_iterations.append(as_steps)
        as_weight = reconstruct_path(G, start, end, algorithm="astar", plot=True)
        a_star_weights.append(as_weight)
        plot_heatmap(G, "astar")
        print("Done\n")

        # save the results into csv file
        output_dir = 'results'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # raw results
        output_file = os.path.join(output_dir, f"raw_results_{place_name.replace(', ', '_')}.csv")
        with open(output_file, 'a') as f:
            #header
            if i == 0:
                f.write("iteration, place_name, distance, dijkstra_steps, dijkstra_weight, a_star_steps, a_star_weight\n")
            f.write(f"{i},{place_name},{dist},{dijkstra_steps},{dijk_weight},{as_steps},{as_weight}\n")


    # average results
    avg_dist = sum(distances) / len(distances)
    avg_dijkstra_steps = sum(dijkstra_iterations) / len(dijkstra_iterations)
    avg_dijkstra_weight = sum(dijkstra_weights) / len(dijkstra_weights)
    avg_a_star_steps = sum(a_star_iterations) / len(a_star_iterations)
    avg_a_star_weight = sum(a_star_weights) / len(a_star_weights)
    print("Place name:", place_name)
    print(f"Average distance: {avg_dist}")
    print(f"Average Dijkstra steps: {avg_dijkstra_steps}")
    print(f"Average Dijkstra weight: {avg_dijkstra_weight}")
    print(f"Average A* steps: {avg_a_star_steps}")
    print(f"Average A* weight: {avg_a_star_weight}")

    # save the results into csv file
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # average results
    output_file = os.path.join(output_dir, f"average_results_{place_name.replace(', ', '_')}.csv")
    with open(output_file, 'a') as f:
        f.write("place_name, distance, dijkstra_steps, dijkstra_weight, a_star_steps, a_star_weight\n")
        f.write(f"{place_name},{avg_dist},{avg_dijkstra_steps},{avg_dijkstra_weight},{avg_a_star_steps},{avg_a_star_weight}\n")



if __name__ == "__main__":
    # Turin | 10 iterations
    place_name = "Turin, Piedmont, Italy"
    execution_loop(place_name, 10)

    # Piedmont | 10 iterations
    place_name = "Piedmont, California, USA"
    execution_loop(place_name, 10)











