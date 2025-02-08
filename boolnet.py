import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import itertools
from itertools import combinations
import random
import matplotlib.pyplot as plt
import networkx as nx 

#binary_time_series function
def edge_detector(gene, scaling=1, edge_type="firstEdge"):
    sorted_gene = np.sort(gene)
    gradients = np.diff(sorted_gene)
    
    if edge_type == "firstEdge":
        threshold_index = np.argmax(gradients > scaling * np.mean(gradients))
    elif edge_type == "maxEdge":
        threshold_index = np.argmax(gradients)
    else:
        raise ValueError("Invalid edge type")
    
    threshold = sorted_gene[threshold_index]
    binarized = (gene > threshold).astype(int)
    return {"bindata": binarized, "thresholds": threshold}

def scan_statistic(gene, window_size=0.25, sign_level=0.1):
    threshold = np.mean(gene) + sign_level * np.std(gene)
    binarized = (gene > threshold).astype(int)
    reject = np.all(binarized == binarized[0])
    return {"bindata": binarized, "thresholds": threshold, "reject": reject}

def binarize_time_series(measurements, method="kmeans", nstart=100, iter_max=1000, 
                          edge="firstEdge", scaling=1, window_size=0.25, 
                          sign_level=0.1, drop_insignificant=False):
    
    if isinstance(measurements, list):
        full_data = np.hstack(measurements)
    else:
        full_data = measurements
    
    if method == "kmeans":
        clusters = []
        thresholds = []
        
        for gene in full_data:
            kmeans = KMeans(n_clusters=2, n_init=nstart, max_iter=iter_max, random_state=0)
            labels = kmeans.fit_predict(gene.reshape(-1, 1))
            centers = kmeans.cluster_centers_.flatten()
            
            if centers[0] > centers[1]:
                labels = np.abs(labels - 1)
            
            threshold = min(centers) + np.abs(centers[1] - centers[0]) / 2
            clusters.append(labels)
            thresholds.append(threshold)
        
        binarized_measurements = np.array(clusters)
        return {"binarizedMeasurements": binarized_measurements, "thresholds": np.array(thresholds)}
    
    elif method == "edgeDetector":
        clusters = []
        thresholds = []
        
        for gene in full_data:
            result = edge_detector(gene, scaling, edge)
            clusters.append(result["bindata"])
            thresholds.append(result["thresholds"])
        
        binarized_measurements = np.array(clusters)
        return {"binarizedMeasurements": binarized_measurements, "thresholds": np.array(thresholds)}
    
    elif method == "scanStatistic":
        clusters = []
        thresholds = []
        reject_list = []
        
        for gene in full_data:
            result = scan_statistic(gene, window_size, sign_level)
            clusters.append(result["bindata"])
            thresholds.append(result["thresholds"])
            reject_list.append(result["reject"])
        
        if drop_insignificant:
            valid_indices = [i for i, reject in enumerate(reject_list) if not reject]
            binarized_measurements = np.array(clusters)[valid_indices]
            thresholds = np.array(thresholds)[valid_indices]
        else:
            binarized_measurements = np.array(clusters)
        
        return {"binarizedMeasurements": binarized_measurements, "thresholds": np.array(thresholds), "reject": np.array(reject_list)}
    
    else:
        raise ValueError("Method must be one of 'kmeans', 'edgeDetector', 'scanStatistic'")

#choose_network function
def choose_network(probabilistic_network, function_indices, dont_care_values=None, readable_functions=False):
    if not isinstance(probabilistic_network, dict):
        raise TypeError("Invalid network type. Expected a dictionary representing a probabilistic Boolean network.")

    genes = probabilistic_network.get("genes", [])
    interactions = probabilistic_network.get("interactions", [])
    fixed = probabilistic_network.get("fixed", {})

    if len(function_indices) != len(genes):
        raise ValueError("Please provide a vector of function indices for each gene!")

    deterministic_interactions = []
    for idx, gene in enumerate(genes):
        interaction_index = function_indices[idx]
        interaction = interactions[idx][interaction_index]

        func = interaction["func"]
        dc_positions = [i for i, val in enumerate(func) if val == -1]

        if dc_positions:
            if not dont_care_values or gene not in dont_care_values:
                raise ValueError(f"No values for the 'don't care' entries were specified for gene '{gene}'!")
            
            if not all(val in [0, 1] for val in dont_care_values[gene]):
                raise ValueError(f"Invalid values for 'don't care' entries specified for gene '{gene}'!")
            
            if len(dont_care_values[gene]) != len(dc_positions):
                raise ValueError(f"There must be exactly {len(dc_positions)} value(s) for 'don't care' entries in the function for gene '{gene}'!")
            
            for pos, val in zip(dc_positions, dont_care_values[gene]):
                func[pos] = val

        deterministic_interactions.append({
            "input": interaction["input"],
            "func": func,
            "expression": interaction.get("expression", "")
        })
    
    return {"genes": genes, "interactions": deterministic_interactions, "fixed": fixed}

#generateRandomNKNetwork

def generate_random_NK_network(n, k, topology=["fixed", "homogeneous", "scale_free"],
                               linkage=["uniform", "lattice"], functionGeneration=["uniform", "biased"],
                               validation_function=None, failure_iterations=10000, simplify=False,
                               no_irrelevant_genes=True, readable_functions=False,
                               d_lattice=1, zero_bias=0.5, gamma=2.5, approx_cutoff=100):
    
    def rzeta(n, k, gamma, approx_cutoff):
        zeta_values = np.random.zipf(gamma, size=n)
        return np.clip(zeta_values, 1, n)[:n]
    
    if len(k) == 1:
        k_i_vec = [k] * n
    elif len(k) == n:
        k_i_vec = k
    else:
        raise ValueError("k must have 1 or n element(s)!")

    if zero_bias == 0 or zero_bias == 1 and no_irrelevant_genes and any(k_i > 0 for k_i in k_i_vec):
        raise ValueError("If setting 'zeroBias' to 0 or 1, you must set 'noIrrelevantGenes' to False!")

    gene_names = [f"Gene{i+1}" for i in range(n)]
    
    interactions = []
    
    for i, k_i in enumerate(k_i_vec):
        if k_i == 0:
            genes = []
            func = [round(random.uniform(0, 1))]
        else:
            table = np.array(list(combinations(range(k_i), 2))) - 1
            if linkage == "uniform":
                genes = random.sample(range(n), k_i)
            elif linkage == "lattice":
                region = list(range(max(1, round(i - k_i * d_lattice)), min(n, round(i + k_i * d_lattice))))
                genes = random.sample(region, k_i)
            else:
                raise ValueError("'linkage' must be one of \"uniform\",\"lattice\"")

            contains_irrelevant = True
            valid_function = True
            counter = 0

            while not valid_function or contains_irrelevant:
                if functionGeneration == "uniform":
                    func = [round(random.uniform(0, 1)) for _ in range(2**k_i)]
                elif functionGeneration == "biased":
                    func = [1 if random.uniform(0, 1) > zero_bias else 0 for _ in range(2**k_i)]
                else:
                    raise ValueError("'functionGeneration' must be one of \"uniform\", \"biased\" or a function!")

                if no_irrelevant_genes:
                    drop_genes = [np.all(func[gene == 1] == func[gene == 0]) for gene in table]
                    contains_irrelevant = sum(drop_genes) > 0
                else:
                    contains_irrelevant = False

                if validation_function:
                    valid_function = validation_function(table, func)

                counter += 1
                if counter > failure_iterations:
                    raise ValueError(f"Could not find a transition function matching the restrictions of validation_function in {failure_iterations} runs.")

        interactions.append({
            "input": genes,
            "func": func,
            "expression": get_interaction_string(readable_functions, func, gene_names)
        })

    fixed = {gene_names[i]: (interaction["func"][0] if interaction["input"][0] == 0 else -1)
             for i, interaction in enumerate(interactions)}

    net = {
        "genes": gene_names,
        "interactions": interactions,
        "fixed": fixed
    }
    
    if simplify:
        net = simplify_network(net, readable_functions)
    
    return net


def get_interaction_string(readable_functions, func, gene_names):
    if readable_functions:
        return ', '.join([f"{gene_names[i]}: {f}" for i, f in enumerate(func)])
    return ', '.join(map(str, func))


def simplify_network(net, readable_functions):
    simplified_network = net.copy()
    if readable_functions:
        simplified_network["interactions"] = [
            {"input": interaction["input"], "func": interaction["func"], 
             "expression": get_interaction_string(readable_functions, interaction["func"], net["genes"])}
            for interaction in net["interactions"]
        ]
    return simplified_network


def generate_canalyzing(input):
    k = len(input)
    table = np.array(list(combinations(range(k), 2))) - 1
    canalyzing_input = random.choice(range(k))
    res = [round(random.uniform(0, 1)) for _ in range(2**k)]
    
    res[table[:, canalyzing_input] == random.choice([0, 1])] = random.choice([0, 1])
    
    return res


def generate_nested_canalyzing(input):
    k = len(input)
    table = np.array(list(combinations(range(k), 2))) - 1
    remaining_indices = list(range(2**k))
    res = [random.choice([0, 1])] * (2**k)

    for canalyzing_input in random.sample(range(k), k):
        new_indices = [idx for idx in remaining_indices if table[idx, canalyzing_input] == random.choice([0, 1])]
        for idx in new_indices:
            res[idx] = random.choice([0, 1])
        remaining_indices = list(set(remaining_indices) - set(new_indices))

    return res

#generateState
def generate_state(network, specs, default=0):
    if "genes" not in network:
        raise ValueError("Network must contain 'genes' key")

    undefined_genes = [gene for gene in specs.keys() if gene not in network["genes"]]
    if undefined_genes:
        raise ValueError(f"Undefined gene names: {', '.join(undefined_genes)}")

    if not all(value in [0, 1] for value in specs.values()):
        raise ValueError("Please provide only Boolean values!")

    lengths = list(set(len(value) for value in specs.values()))
    if len(lengths) > 1:
        raise ValueError("The number of specifications for each gene must be the same!")

    if lengths == [1]:
        state = {gene: default for gene in network["genes"]}
        state.update(specs)
    else:
        length = lengths[0]
        state = np.full((length, len(network["genes"])), default, dtype=int)
        gene_index = {gene: idx for idx, gene in enumerate(network["genes"])}

        for gene, values in specs.items():
            idx = gene_index[gene]
            state[:, idx] = values

    return state


#generateTimeSeries
def generateTimeSeries(network, numSeries, numMeasurements, 
                       type="synchronous", geneProbabilities=None, 
                       perturbations=0, noiseLevel=0.0):
    symbolic = isinstance(network, dict) and 'genes' in network
    
    if geneProbabilities is None:
        geneProbabilities = [1.0 / len(network['genes'])] * len(network['genes'])
    
    perturbationMatrix = None
    if perturbations > 0:
        perturbationMatrix = np.array([
            [random.choice([0, 1]) if random.random() < perturbations / len(network['genes']) else np.nan
             for _ in range(numSeries)]
            for _ in range(len(network['genes']))
        ])
    
    timeSeries = []
    for i in range(numSeries):
        if symbolic:
            if perturbations > 0:
                fixedIdx = np.where(~np.isnan(perturbationMatrix[:, i]))[0]
                fixedValues = perturbationMatrix[fixedIdx, i]
            res = np.random.randint(2, size=(len(network['genes']), numMeasurements))
        else:
            startState = np.random.randint(2, size=len(network['genes']))
            
            if perturbations > 0:
                fixedIdx = np.where(~np.isnan(perturbationMatrix[:, i]))[0]
                fixedValues = perturbationMatrix[fixedIdx, i]
                startState[fixedIdx] = fixedValues
            
            res = np.zeros((len(network['genes']), numMeasurements))
            res[:, 0] = startState
            for j in range(1, numMeasurements):
                if type == "synchronous":
                    startState = synchronousTransition(network, startState)
                elif type == "asynchronous":
                    startState = asynchronousTransition(network, startState, geneProbabilities)
                elif type == "probabilistic":
                    startState = probabilisticTransition(network, startState, geneProbabilities)
                res[:, j] = startState
        
        if noiseLevel != 0:
            res = res + np.random.normal(0, noiseLevel, res.shape)
        
        timeSeries.append(res)
    
    if perturbations > 0:
        timeSeries.append({'perturbations': perturbationMatrix})
    
    return timeSeries

def synchronousTransition(network, state):
    return np.random.randint(2, size=len(state))

def asynchronousTransition(network, state, geneProbabilities):
    return np.random.randint(2, size=len(state))

def probabilisticTransition(network, state, geneProbabilities):
    return np.random.randint(2, size=len(state))

network = {
    'genes': ['Gene1', 'Gene2', 'Gene3'],
}

timeSeries = generateTimeSeries(network, numSeries=5, numMeasurements=10, type="synchronous", noiseLevel=0.1)
for ts in timeSeries:
    print(ts)

#getAttractorSequence
def getAttractorSequence(attractorInfo, attractorNo=None):
    if not (isinstance(attractorInfo, dict) and ('attractors' in attractorInfo or 'stateInfo' in attractorInfo)):
        raise ValueError("Invalid attractorInfo object.")
    
    if isinstance(attractorInfo, dict) and 'attractors' in attractorInfo:
        if attractorInfo.get('attractors') is None:
            raise ValueError("This SymbolicSimulation structure does not contain attractor information. Please re-run simulateSymbolicModel() with returnAttractors=True!")
        if attractorNo is None or attractorNo <= 0 or attractorNo > len(attractorInfo['attractors']):
            raise ValueError("Please provide a valid attractor number!")
        return attractorInfo['attractors'][attractorNo - 1]
    else:
        if attractorNo is None or attractorNo <= 0 or attractorNo > len(attractorInfo['attractors']):
            raise ValueError("Please provide a valid attractor number!")
        if 'initialStates' in attractorInfo['attractors'][attractorNo - 1]:
            raise ValueError("A sequence can be obtained for synchronous attractors only!")
        
        numGenes = len(attractorInfo['stateInfo']['genes'])
        
        involvedStates = attractorInfo['attractors'][attractorNo - 1]['involvedStates']
        binMatrix = np.array([dec2bin(state, numGenes) for state in involvedStates])
        
        return {attractorInfo['stateInfo']['genes'][i]: binMatrix[:, i] for i in range(numGenes)}

def dec2bin(decimal, numGenes):
    return [int(bit) for bit in format(decimal, f'0{numGenes}b')]

#getAttractors
def get_attractors(network, type="synchronous", method="exhaustive", start_states=None, genes_on=None, genes_off=None, canonical=True, random_chain_length=10000, avoid_self_loops=True, gene_probabilities=None, max_attractor_length=None, return_table=True):
    if genes_on is None:
        genes_on = []
    if genes_off is None:
        genes_off = []
    if start_states is None:
        start_states = []

    non_fixed_positions = [i for i, val in enumerate(network['fixed']) if val == -1]

    if type == "asynchronous":
        if 'SymbolicBooleanNetwork' in network:
            raise ValueError("Only synchronous updates are allowed for symbolic networks!")
        if method == "exhaustive":
            raise ValueError("Asynchronous attractor search cannot be performed in exhaustive search mode!")
        if gene_probabilities:
            if len(gene_probabilities) != len(network['genes']):
                raise ValueError("Please supply exactly one probability for each gene!")
            if abs(1.0 - sum(gene_probabilities)) > 0.0001:
                raise ValueError("The supplied gene probabilities do not sum up to 1!")

    if max_attractor_length is None or np.isinf(max_attractor_length):
        max_attractor_length = None

    if method not in ["exhaustive", "sat.exhaustive", "sat.restricted", "random", "chosen"]:
        if type == "asynchronous" and not start_states:
            start_states = max(round(2 ** len(non_fixed_positions) / 20), 5)
            method = "random"
        elif isinstance(start_states, (int, float)):
            method = "random"
        elif isinstance(start_states, list) and start_states:
            method = "chosen"
        elif max_attractor_length is not None:
            method = "sat.restricted"
        else:
            method = "exhaustive"

    if genes_on:
        network = fix_genes(network, genes_on, 1)
    if genes_off:
        network = fix_genes(network, genes_off, 0)

    if 'SymbolicBooleanNetwork' in network:
        return simulate_symbolic_model(network, method, start_states, max_attractor_length, return_table and method not in ["sat.exhaustive", "sat.restricted"], False, True, canonical)

    if method == "sat.restricted" and max_attractor_length is None:
        raise ValueError("max_attractor_length must be set for method=\"sat.restricted\"!")

    if len(network['genes']) > 29 and method == "exhaustive" and type == "synchronous":
        method = "sat.exhaustive"
        print("Warning: Switching to SAT-based exhaustive search for networks with more than 29 genes.")

    if method in ["sat.exhaustive", "sat.restricted"] and type != "synchronous":
        raise ValueError("SAT-based search can only be used for synchronous networks!")

    if method == "random":
        if not isinstance(start_states, (int, float)):
            raise ValueError("Please supply the number of random states in start_states!")
        if start_states > (2 ** len(non_fixed_positions)):
            if type == "synchronous":
                start_states = []
                print("Warning: Performing an exhaustive search due to large number of random states.")
            else:
                print(f"Warning: Maximum number of different states is {2 ** len(non_fixed_positions)}!")
                start_states = 2 ** len(non_fixed_positions)
        start_states = generate_random_start_states(network, start_states)
    elif method == "chosen":
        if not isinstance(start_states, list) or not start_states:
            raise ValueError("No start states supplied!")
        if not all(len(state) == len(network['genes']) for state in start_states):
            raise ValueError(f"Please provide binary vectors with {len(network['genes'])} elements in start_states!")
        fixed_genes = [i for i, val in enumerate(network['fixed']) if val != -1]
        valid_states = [state for state in start_states if all(state[i] == network['fixed'][i] for i in fixed_genes)]
        if not valid_states:
            raise ValueError("None of the supplied start states matched the restrictions of the fixed genes!")
        if len(valid_states) != len(start_states):
            print("Warning: Some start states did not match the restrictions of the fixed genes and were removed!")
        start_states = valid_states

    if max_attractor_length is not None and method not in ["sat.exhaustive", "sat.restricted"]:
        raise ValueError("max_attractor_length can only be used with method=\"sat.exhaustive\" or method=\"sat.restricted\"!")

    converted_start_states = [bin2dec(state, len(network['genes'])) for state in start_states] if start_states else []

    input_genes = [gene for interaction in network['interactions'] for gene in interaction['input']]
    input_gene_positions = np.cumsum([0] + [len(interaction['input']) for interaction in network['interactions']]).tolist()

    transition_functions = [func for interaction in network['interactions'] for func in interaction['func']]
    transition_function_positions = np.cumsum([0] + [len(interaction['func']) for interaction in network['interactions']]).tolist()

    search_type = 0
    if type == "synchronous":
        if method == "sat.exhaustive":
            search_type = 2
        elif method == "sat.restricted":
            search_type = 3
    elif type == "asynchronous":
        search_type = 1

    result = get_attractors_c(input_genes, input_gene_positions, transition_functions, transition_function_positions, network['fixed'], converted_start_states, search_type, gene_probabilities, random_chain_length, avoid_self_loops, return_table, max_attractor_length)

    if not result:
        raise ValueError("An error occurred in external code!")

    if not result['attractors']:
        raise ValueError("No attractors were identified! Check the parameters and restart.")

    num_elements_per_entry = len(network['genes']) // 32 + (1 if len(network['genes']) % 32 != 0 else 0)

    if 'stateInfo' in result:
        result['stateInfo']['table'] = np.reshape(result['stateInfo']['table'], (num_elements_per_entry, -1))
        if 'initialStates' in result['stateInfo']:
            result['stateInfo']['initialStates'] = np.reshape(result['stateInfo']['initialStates'], (num_elements_per_entry, -1))

    for attractor in result['attractors']:
        attractor['involvedStates'] = np.reshape(attractor['involvedStates'], (num_elements_per_entry, -1))
        if canonical:
            attractor['involvedStates'] = canonical_state_order(attractor['involvedStates'])
        if 'initialStates' in attractor:
            attractor['initialStates'] = np.reshape(attractor['initialStates'], (num_elements_per_entry, -1))
        if 'nextStates' in attractor:
            attractor['nextStates'] = np.reshape(attractor['nextStates'], (num_elements_per_entry, -1))
        if attractor['basinSize'] == 0:
            attractor['basinSize'] = None

    attractor_lengths = [len(attractor['involvedStates']) for attractor in result['attractors']]
    reordering = np.argsort(attractor_lengths)
    result['attractors'] = [result['attractors'][i] for i in reordering]

    if 'stateInfo' in result:
        inverse_order = [np.where(reordering == i)[0][0] for i in range(len(reordering))]
        result['stateInfo']['attractorAssignment'] = [inverse_order[i] for i in result['stateInfo']['attractorAssignment']]

    result['stateInfo']['genes'] = network['genes']
    result['stateInfo']['fixedGenes'] = network['fixed']

    if 'table' in result['stateInfo']:
        result['stateInfo'].__class__ = 'BooleanStateInfo'
    result.__class__ = 'AttractorInfo'

    return result

def fix_genes(network, genes, value):
    for gene in genes:
        index = network['genes'].index(gene)
        network['fixed'][index] = value
    return network

def generate_random_start_states(network, num_states):
    non_fixed_positions = [i for i, val in enumerate(network['fixed']) if val == -1]
    states = []
    for _ in range(num_states):
        state = [0] * len(network['genes'])
        for pos in non_fixed_positions:
            state[pos] = np.random.randint(2)
        states.append(state)
    return states

def bin2dec(binary, length):
    return int("".join(map(str, binary)), 2)

def canonical_state_order(states):
    return states[:, np.lexsort(states[::-1])]

def get_attractors_c(input_genes, input_gene_positions, transition_functions, transition_function_positions, fixed_genes, start_states, search_type, gene_probabilities, random_chain_length, avoid_self_loops, return_table, max_attractor_length):
    attractors = []
    state_info = {}

    if search_type == 0:   
        all_states = list(itertools.product([0, 1], repeat=len(fixed_genes)))
        for state in all_states:
            next_state = get_next_state(state, input_genes, input_gene_positions, transition_functions, transition_function_positions)
            if next_state == state:
                attractors.append({'involvedStates': [state], 'basinSize': 1})
    elif search_type == 1:  # 
        for state in start_states:
            next_state = get_next_state(state, input_genes, input_gene_positions, transition_functions, transition_function_positions)
            if next_state == state:
                attractors.append({'involvedStates': [state], 'basinSize': 1})
    elif search_type == 2:  
        pass   
    elif search_type == 3:  
        pass  

    return {'attractors': attractors, 'stateInfo': state_info}

def get_next_state(state, input_genes, input_gene_positions, transition_functions, transition_function_positions):
    next_state = list(state)
    for i in range(len(input_gene_positions) - 1):
        inputs = input_genes[input_gene_positions[i]:input_gene_positions[i + 1]]
        func = transition_functions[transition_function_positions[i]:transition_function_positions[i + 1]]
        next_state[i] = evaluate_function(inputs, func, state)
    return tuple(next_state)

def evaluate_function(inputs, func, state):
    if func[0] == 0:  # AND
        return int(all(state[i] for i in inputs))
    elif func[0] == 1:  # OR
        return int(any(state[i] for i in inputs))
    elif func[0] == 2:  # NOT
        return int(not state[inputs[0]])
    else:
        return 0

def simulate_symbolic_model(network, method, start_states, max_attractor_length, return_graph, return_sequences, return_attractors, canonical):
    attractors = []
    state_info = {}
    return {'attractors': attractors, 'stateInfo': state_info}

#getStateSummary
def bin2dec(binary_array, length):
    return int("".join(map(str, binary_array)), 2)

def get_state_summary(attractor_info, state):
    if isinstance(attractor_info, dict):
        if 'graph' not in attractor_info:
            raise ValueError("This SymbolicSimulation structure does not contain transition table information. Please re-run simulateSymbolicModel() with returnGraph=True!")
        
        gene_cols = [col for col in attractor_info['graph'].columns if not col.startswith(('attractorAssignment', 'transitionsToAttractor'))]
        num_genes = len(gene_cols) // 2
        
        state_indices = attractor_info['graph'].iloc[:, :num_genes].apply(lambda x: np.array_equal(x, state), axis=1)
        return attractor_info['graph'][state_indices]
    
    else:
        if len(state) != len(attractor_info['stateInfo']['genes']):
            raise ValueError("State must have one element for each gene!")
        
        if 'table' not in attractor_info['stateInfo']:
            raise ValueError("This AttractorInfo structure does not contain transition table information. Please re-run getAttractors() with a synchronous search and returnTable=True!")
        
        if 'initialStates' in attractor_info['stateInfo']:
            state_no = bin2dec(state, len(state))
            state_no = np.where(np.apply_along_axis(lambda x: np.array_equal(x, state_no), 0, attractor_info['stateInfo']['initialStates']))[0][0]
        else:
            shortened_state = state[np.array(attractor_info['stateInfo']['fixedGenes']) == -1]
            state_no = bin2dec(shortened_state, len(shortened_state))
        
        next_state = dec2bin(attractor_info['stateInfo']['table'][state_no], len(attractor_info['stateInfo']['genes']))
        attractor_assignment = attractor_info['stateInfo']['attractorAssignment'][state_no]
        steps_to_attractor = attractor_info['stateInfo']['stepsToAttractor'][state_no]
        
        res = pd.DataFrame([np.concatenate([state, next_state, [attractor_assignment], [steps_to_attractor]])],
                           columns=[f"initialState.{gene}" for gene in attractor_info['stateInfo']['genes']] +
                                   [f"nextState.{gene}" for gene in attractor_info['stateInfo']['genes']] +
                                   ["attractorAssignment", "transitionsToAttractor"])
        
        return res
    
#getTransitionProbablilties
def get_transition_probabilities(markov_simulation):
    if 'table' not in markov_simulation or 'genes' not in markov_simulation:
        raise ValueError("Missing required keys ('table' or 'genes').")
    
    table = markov_simulation['table']
    if 'initialStates' not in table or 'nextStates' not in table or 'probabilities' not in table:
        raise ValueError("Missing required data in 'table'.")
    
    def dec2bin(x, length):
        return np.array(list(map(int, np.binary_repr(x, width=length))))
    
    initial_states = np.array([dec2bin(state, len(markov_simulation['genes'])) for state in table['initialStates']])
    next_states = np.array([dec2bin(state, len(markov_simulation['genes'])) for state in table['nextStates']])
    
    idx = np.argsort([''.join(map(str, state)) for state in initial_states])
    
    res = pd.DataFrame(
        np.column_stack((initial_states, next_states, table['probabilities'])),
        columns=[f"initialState.{gene}" for gene in markov_simulation['genes']] +
                [f"nextState.{gene}" for gene in markov_simulation['genes']] +
                ["probability"]
    )
    
    res = res.iloc[idx].reset_index(drop=True)
    
    return res

#getTransitionTables
def get_transition_probabilities(markov_simulation):
    if 'table' not in markov_simulation or 'genes' not in markov_simulation:
        raise ValueError("Missing required keys ('table' or 'genes').")
    
    table = markov_simulation['table']
    if 'initialStates' not in table or 'nextStates' not in table or 'probabilities' not in table:
        raise ValueError("Missing required data in 'table'.")
    
    def dec2bin(x, length):
        return np.array(list(map(int, np.binary_repr(x, width=length))))
    
    initial_states = np.array([dec2bin(state, len(markov_simulation['genes'])) for state in table['initialStates']])
    next_states = np.array([dec2bin(state, len(markov_simulation['genes'])) for state in table['nextStates']])
    
    idx = np.argsort([''.join(map(str, state)) for state in initial_states])
    
    res = pd.DataFrame(
        np.column_stack((initial_states, next_states, table['probabilities'])),
        columns=[f"initialState.{gene}" for gene in markov_simulation['genes']] +
                [f"nextState.{gene}" for gene in markov_simulation['genes']] +
                ["probability"]
    )
    
    res = res.iloc[idx].reset_index(drop=True)
    
    return res

#loadBioTapestry







#loadNetwork
def load_network(file, body_separator=",", lowercase_genes=False, symbolic=False):
    with open(file, 'r') as f:
        lines = f.readlines()
    
    lines = [line.split('#')[0].strip() for line in lines if line.strip() and not line.startswith('#')]
    
    if not lines:
        raise ValueError("Header expected!")
    
    header = lines[0].lower().split(body_separator)
    if len(header) < 2 or header[0] != "targets" or header[1] not in ["functions", "factors"]:
        raise ValueError(f"Invalid header: {lines[0]}")
    
    lines = lines[1:]
    if lowercase_genes:
        lines = [line.lower() for line in lines]
    
    processed_lines = []
    for line in lines:
        bracket_count = 0
        last_idx = 0
        result = []
        
        for i, char in enumerate(line):
            if char == "(":
                bracket_count += 1
            elif char == ")":
                bracket_count -= 1
            elif char == body_separator and bracket_count == 0:
                result.append(line[last_idx:i].strip())
                last_idx = i + 1
        result.append(line[last_idx:].strip())
        processed_lines.append(result)
    
    targets = [item[0].strip() for item in processed_lines]
    
    for target in targets:
        if not (target[0].isalpha() or target[0] == '_') or not all(c.isalnum() or c == '_' for c in target):
            raise ValueError(f"Invalid gene name: {target}")
    
    factors = [item[1].strip() for item in processed_lines]
    probabilities = [float(item[2].strip()) if len(item) >= 3 else 1.0 for item in processed_lines]
    
    genes = list(set(targets + [factor for factor in factors]))
    is_probabilistic = len(set(targets)) < len(targets)
    
    if symbolic:
        if is_probabilistic:
            raise ValueError("Probabilistic networks cannot be loaded with symbolic=True!")
        interactions = {gene: factor for gene, factor in zip(targets, factors)}
        fixed = {gene: -1 for gene in genes}
        return {"genes": genes, "interactions": interactions, "fixed": fixed}
    
    interactions = {}
    fixed = {gene: -1 for gene in genes}
    
    for i, target in enumerate(targets):
        interaction = {"func": factors[i]}
        if is_probabilistic:
            interaction["probability"] = probabilities[i]
            interactions.setdefault(target, []).append(interaction)
        else:
            interactions[target] = interaction
    
    return {"genes": genes, "interactions": interactions, "fixed": fixed}

#markovSimulation
import numpy as np

def markov_simulation(network, num_iterations=1000, start_states=[], cutoff=0.001, return_table=True):
    if not hasattr(network, 'interactions') or not hasattr(network, 'fixed'):
        raise ValueError("Invalid network type")
    
    if sum(network.fixed == -1) > 32:
        raise ValueError("A Markov chain simulation with more than 32 non-fixed genes is not supported!")
    
    input_genes = np.array([func.input for interaction in network.interactions for func in interaction]).flatten().astype(int)
    input_gene_positions = np.cumsum([0] + [len(func.input) for interaction in network.interactions for func in interaction]).astype(int)
    transition_functions = np.array([func.func for interaction in network.interactions for func in interaction]).flatten().astype(int)
    transition_function_positions = np.cumsum([0] + [len(func.func) for interaction in network.interactions for func in interaction]).astype(int)
    function_assignment = np.array([i for i, interaction in enumerate(network.interactions) for _ in interaction], dtype=int)
    
    fixed_genes = np.where(network.fixed != -1)[0]
    if start_states:
        states_valid = [np.all(state[fixed_genes] == network.fixed[fixed_genes]) for state in start_states]
        if not all(states_valid):
            print("Warning: Some supplied start states did not match the restrictions of the fixed genes and were removed!")
        start_states = [state for state, valid in zip(start_states, states_valid) if valid]
    
    converted_start_states = [bin2dec(state, len(network.genes)) for state in start_states] if start_states else None
    
    states = np.zeros((num_iterations, len(network.genes)))
    probabilities = np.ones(num_iterations)
    
    for i in range(num_iterations):
        current_state = np.random.choice(converted_start_states) if converted_start_states else np.random.randint(0, 2, len(network.genes))
        states[i] = current_state
    
    num_elements_per_entry = len(network.genes) // 32 if len(network.genes) % 32 == 0 else len(network.genes) // 32 + 1
    reached_states = np.column_stack(([dec2bin(state, len(network.genes)) for state in states], probabilities))
    
    if return_table:
        initial_states = states[:-1]
        next_states = states[1:]
        result = {
            "reached_states": reached_states,
            "table": {
                "initial_states": initial_states,
                "next_states": next_states,
                "probabilities": probabilities
            },
            "genes": network.genes
        }
    else:
        result = {
            "reached_states": reached_states,
            "table": None,
            "genes": network.genes
        }
    
    return result

#perturbNetwork
def perturb_network(network, perturb_type="functions", method="bitflip", simplify=True, readable_functions=False, exclude_fixed=True, max_num_bits=1, num_states=None):
    if not hasattr(network, 'interactions') or not hasattr(network, 'fixed'):
        raise ValueError("Invalid network type")
    
    fixed_genes = np.where(network.fixed != -1)[0]
    num_states = num_states or max(1, 2**len(network.genes) // 100)
    
    if perturb_type == "functions":
        gene_indices = np.setdiff1d(np.arange(len(network.interactions)), fixed_genes) if exclude_fixed else np.arange(len(network.interactions))
        gene_idx = np.random.choice(gene_indices)
        
        if method == "bitflip":
            flip_indices = np.random.choice(len(network.interactions[gene_idx].func), size=min(max_num_bits, len(network.interactions[gene_idx].func)), replace=False)
            network.interactions[gene_idx].func[flip_indices] = 1 - network.interactions[gene_idx].func[flip_indices]
        elif method == "shuffle":
            np.random.shuffle(network.interactions[gene_idx].func)
        else:
            raise ValueError("Method must be 'bitflip' or 'shuffle'")
    
    elif perturb_type == "transitions":
        states = np.array([dec2bin(state, len(network.genes)) for state in get_attractors(network)])
        state_indices = np.random.choice(states.shape[0], size=min(num_states, states.shape[0]), replace=False)
        
        for state_idx in state_indices:
            flip_indices = np.setdiff1d(np.arange(len(network.genes)), fixed_genes) if exclude_fixed else np.arange(len(network.genes))
            
            if method == "bitflip":
                flip_index = np.random.choice(flip_indices, size=min(max_num_bits, len(flip_indices)), replace=False)
                states[state_idx, flip_index] = 1 - states[state_idx, flip_index]
            elif method == "shuffle":
                np.random.shuffle(states[state_idx])
            else:
                raise ValueError("Method must be 'bitflip' or 'shuffle'")
        
        for i, gene in enumerate(network.genes):
            network.interactions[i].func = states[:, i]
    
    if simplify:
        network = simplify_network(network, readable_functions)
    
    return network

#perturbTrajectories
def perturb_trajectories(network, measure="hamming", num_samples=1000, flip_bits=1, update_type="synchronous", gene=None):
    if not hasattr(network, 'interactions') or not hasattr(network, 'fixed'):
        raise ValueError("Invalid network type")
    
    fixed_indices = np.where(network.fixed != -1)[0]
    non_fixed_indices = np.where(network.fixed == -1)[0]
    
    if len(non_fixed_indices) == 0:
        raise ValueError("All genes in this network are fixed!")
    
    start_states = [np.random.randint(0, 2, len(network.genes)) for _ in range(num_samples)]
    for state in start_states:
        state[fixed_indices] = network.fixed[fixed_indices]
    
    perturbed_states = [state.copy() for state in start_states]
    for state in perturbed_states:
        flip = np.random.choice(non_fixed_indices, size=flip_bits, replace=False)
        state[flip] = 1 - state[flip]
    
    if measure == "hamming":
        dists = [np.sum(state1 != state2) / len(network.genes) for state1, state2 in zip(start_states, perturbed_states)]
        return {"stat": dists, "value": np.mean(dists)}
    elif measure == "sensitivity":
        if gene is None:
            raise ValueError("Parameter 'gene' must be set for sensitivity analysis!")
        gene_idx = network.genes.index(gene) if isinstance(gene, str) else gene
        sensitivities = [state1[gene_idx] != state2[gene_idx] for state1, state2 in zip(start_states, perturbed_states)]
        return {"stat": sensitivities, "value": np.mean(sensitivities)}
    else:
        raise ValueError("Unsupported measure type")
    
#plotAttractors
def plot_attractors(attractor_info, title="", on_color="green", off_color="red"):
    if not isinstance(attractor_info, dict) or "reached_states" not in attractor_info:
        raise ValueError("Invalid attractor_info format")
    
    reached_states = attractor_info["reached_states"][:, :-1].astype(int)
    num_genes, num_states = reached_states.shape
    
    fig, ax = plt.subplots(figsize=(num_states / 2, num_genes / 2))
    ax.set_xticks(range(num_states))
    ax.set_yticks(range(num_genes))
    ax.set_yticklabels(attractor_info.get("genes", [f"Gene {i+1}" for i in range(num_genes)]))
    
    for i in range(num_genes):
        for j in range(num_states):
            color = on_color if reached_states[i, j] == 1 else off_color
            ax.add_patch(plt.Rectangle((j, num_genes - i - 1), 1, 1, color=color))
    
    ax.set_xlim(0, num_states)
    ax.set_ylim(0, num_genes)
    ax.set_title(title)
    ax.set_xticklabels([])
    ax.set_xticks([])
    
    plt.show()

#plotNetworkWiring




#plotPBNTransitions
def plot_pbn_transitions(markov_simulation, state_subset=None, draw_probabilities=True, draw_state_labels=True,
                         layout='spring_layout', plot_it=True, **kwargs):
    if not hasattr(markov_simulation, 'table'):
        raise ValueError("The supplied simulation result does not contain transition information. "
                         "Please re-run markovSimulation() with returnTable=True!")
    
    table = markov_simulation['table']
    
    edge_matrix = []
    for initial_state, next_state in zip(table['initialStates'], table['nextStates']):
        initial_bin = ''.join(map(str, initial_state))
        next_bin = ''.join(map(str, next_state))
        edge_matrix.append((initial_bin, next_bin))
    
    edge_matrix = np.array(edge_matrix)
    
    if state_subset is not None:
        state_subset = [''.join(map(str, s)) for s in state_subset]
        keep_indices = [i for i, edge in enumerate(edge_matrix) 
                        if edge[0] in state_subset and edge[1] in state_subset]
        
        edge_matrix = edge_matrix[keep_indices]
        probabilities = table['probabilities'][keep_indices]
    else:
        probabilities = table['probabilities']
    
    vertices = set(edge_matrix.flatten())
    
    graph = nx.DiGraph()
    graph.add_nodes_from(vertices)
    
    for i, (start, end) in enumerate(edge_matrix):
        graph.add_edge(start, end, label=str(probabilities[i]))
    
    if plot_it:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if layout == 'spring_layout':
            pos = nx.spring_layout(graph)
        elif layout == 'circular_layout':
            pos = nx.circular_layout(graph)
        elif layout == 'kamada_kawai_layout':
            pos = nx.kamada_kawai_layout(graph)
        else:
            pos = nx.spring_layout(graph)
        
        nx.draw(graph, pos, with_labels=True, node_size=100, font_size=10, node_color='grey', ax=ax, **kwargs)
        
        if draw_probabilities:
            edge_labels = nx.get_edge_attributes(graph, 'label')
            nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax, font_color='green', **kwargs)
        
        if draw_state_labels:
            nx.draw_networkx_labels(graph, pos, font_size=12, font_color='black', **kwargs)
        
        plt.show()
    
    return graph

#plotSequence
def plot_sequence(network=None, start_state=None, include_attractor_states="all", sequence=None, title="", mode="table",
                  plot_fixed=True, grouping={}, on_color="#4daf4a", off_color="#e41a1c", layout=None, draw_labels=True,
                  draw_legend=True, highlight_attractor=True, reverse=False, border_color="black", eps=0.1,
                  attractor_sep_lwd=2, attractor_sep_col="blue", **kwargs):
    if network:
        if not hasattr(network, "genes"):
            raise ValueError("Invalid network type")
        if start_state is None or sequence is not None:
            raise ValueError("Either 'network' and 'start_state' or 'sequence' must be provided!")
        # Here, we assume sequence is directly passed or computed outside this function
    else:
        if sequence is None or start_state is not None:
            raise ValueError("Either 'network' and 'start_state' or 'sequence' must be provided!")
    
    if mode == "table":
        total_matrix = sequence.T
        if len(grouping) > 0:
            total_matrix = total_matrix[grouping["index"], :]
        if total_matrix.columns is None:
            total_matrix.columns = range(total_matrix.shape[1])
        if reverse:
            total_matrix = total_matrix[::-1]
        
        plt.figure()
        plt.xlim(0, total_matrix.shape[1])
        plt.ylim(-2, total_matrix.shape[0] + 1)
        plt.axis('off')
        
        unit_factor = (total_matrix.shape[1] - 2 * eps) / total_matrix.shape[1]
        for i in range(total_matrix.shape[1]):
            for j in range(total_matrix.shape[0]):
                rect_col = on_color if total_matrix.iloc[j, i] else off_color
                plt.gca().add_patch(plt.Rectangle((eps + i * unit_factor, j - 1), unit_factor, 1, color=rect_col,
                                                 edgecolor=border_color, linewidth=2))
        
        if draw_legend:
            plt.legend([on_color, off_color], ["active", "inactive"], loc="lower left")
        
        if highlight_attractor:
            attractor_start = min(sequence["attractor"]) - 1
            plt.axvline(x=eps + attractor_start * unit_factor, color=attractor_sep_col, linewidth=attractor_sep_lwd)
            plt.arrow(eps + attractor_start * unit_factor, total_matrix.shape[0] + 0.5, 
                      total_matrix.shape[1] - eps, 0, head_width=0.1, head_length=0.1, fc=attractor_sep_col, ec=attractor_sep_col)
        
        plt.show()
    
    elif mode == "graph":
        if layout is None:
            layout = np.column_stack([np.linspace(-1, 1, sequence.shape[0]), np.zeros(sequence.shape[0])])
        
        states = ["".join(map(str, seq)) for seq in sequence]
        nodes = {i: state for i, state in enumerate(states)}
        
        graph = nx.DiGraph()
        for i, state in enumerate(states[:-1]):
            graph.add_edge(state, states[i + 1])
        
        if highlight_attractor and sequence.get("attractor"):
            attractor_edge = [max(sequence["attractor"]), min(sequence["attractor"])]
            graph.add_edge(states[attractor_edge[0]], states[attractor_edge[1]])
        
        pos = layout
        nx.draw(graph, pos, with_labels=draw_labels, labels=nodes, node_size=100, node_color="grey", **kwargs)
        plt.title(title)
        plt.show()
        
        return graph

#plotStateGaph
def plot_state_graph(state_graph,
                     highlight_attractors=True,
                     color_basins=True,
                     color_set=None,
                     draw_legend=True,
                     draw_labels=False,
                     layout="kamada_kawai_layout",
                     piecewise=False,
                     basin_lty=2,
                     attractor_lty=1,
                     plot_it=True,
                     colors_alpha=None,
                     **kwargs):

    assert isinstance(state_graph, (dict, list)), "state_graph must be a dictionary or list"
    
    if colors_alpha is None:
        colors_alpha = [0.3, 0.3, 1, 1]
        
    if len(colors_alpha) != 4:
        print("colors_alpha parameter not properly specified. Parameter will be set to opaque values (1,1,1,1).")
        colors_alpha = [1, 1, 1, 1]
        
    if np.any(np.array(colors_alpha) < 0) or np.any(np.array(colors_alpha) > 1):
        print("colors_alpha parameters are not in range [0,1] - they will be normalized.")
        colors_alpha = np.array(colors_alpha) / np.sum(colors_alpha)

    if color_set is None:
        color_set = ["blue", "green", "red", "darkgoldenrod", "gold", "brown", "cyan",
                     "purple", "orange", "seagreen", "tomato", "darkgray", "chocolate",
                     "maroon", "darkgreen", "gray12", "blue4", "cadetblue", "darkgoldenrod4",
                     "burlywood2"]

    graph = nx.DiGraph()

    # Assuming `state_graph` is a dictionary with keys 'transitions' and 'attractorAssignment'
    transitions = state_graph.get('transitions', [])
    attractor_assignment = state_graph.get('attractorAssignment', [])
    attractor_indices = state_graph.get('transitionsToAttractor', [])

    for from_state, to_state in transitions:
        graph.add_edge(from_state, to_state)

    attractors = list(set(attractor_assignment))
    attractors = [x for x in attractors if x is not None]

    if color_basins:
        for attractor in attractors:
            basin_indices = [i for i, x in enumerate(attractor_assignment) if x == attractor]
            for idx in basin_indices:
                graph.nodes[list(graph.nodes())[idx]]['color'] = color_set[(attractor-1) % len(color_set)]

    if highlight_attractors:
        for attractor in attractors:
            attractor_indices = [idx for idx, x in enumerate(attractor_assignment) if x == attractor]
            for idx in attractor_indices:
                graph.nodes[list(graph.nodes())[idx]]['color'] = 'red'

    pos = getattr(nx, layout)(graph)
    
    # Draw the graph
    node_colors = [graph.nodes[node].get('color', 'gray') for node in graph.nodes]
    edge_colors = ['darkgray' for _ in graph.edges]
    
    nx.draw(graph, pos, node_color=node_colors, edge_color=edge_colors, **kwargs)

    if draw_legend:
        labels = [f"Attractor {i}" for i in range(len(attractors))]
        plt.legend(labels, loc="upper left")

    if plot_it:
        plt.show()
    return graph

#reconstructNetwork
def reconstruct_network(measurements, method="bestfit", maxK=5,
                        required_dependencies=None, excluded_dependencies=None, perturbations=None,
                        readable_functions=False, all_solutions=False, return_pbn=False):
    if maxK < 0:
        raise ValueError("maxK must be >= 0!")

    meth = {"bestfit": 0, "reveal": 1}.get(method)
    if meth is None:
        raise ValueError("'method' must be one of 'bestfit', 'reveal'")

    perturbation_matrix = None
    
    if isinstance(measurements, pd.DataFrame):
        num_genes = (measurements.shape[1] - 2) // 2
        
        if num_genes < maxK:
            maxK = num_genes
            print(f"Warning: maxK was chosen greater than the total number of input genes and reset to {num_genes}!")
        
        if perturbations is not None:
            if not np.all(np.isin(perturbations, [0, 1, -1, np.nan])):
                raise ValueError("The perturbation matrix may only contain the values 0,1,-1 and NaN!")
            perturbations = np.nan_to_num(perturbations, nan=-1)
            
            if perturbations.ndim == 1:
                perturbations = perturbations[:, np.newaxis]
            if perturbations.shape[1] != 1 and perturbations.shape[1] != measurements.shape[0]:
                raise ValueError("There must be either one global vector of perturbations, or a matrix containing exact one column for each time series!")
            
            if perturbations.shape[0] != num_genes:
                raise ValueError("The perturbation matrix must have exactly the same number of rows as the measurements!")
            
            if perturbations.shape[1] == 1:
                perturbation_matrix = np.full((num_genes, measurements.shape[0]), -1)
                for j in range(measurements.shape[0]):
                    perturbation_matrix[:, j] = perturbations[:, 0]
            else:
                perturbation_matrix = np.array(perturbations)
        
        input_states = measurements.iloc[:, :num_genes].values.T.astype(int)
        output_states = measurements.iloc[:, num_genes:2 * num_genes].values.T.astype(int)
        num_states = measurements.shape[0]
        gene_names = [col.split(".")[1] for col in measurements.columns[:num_genes]]
    
    else:
        if isinstance(measurements, pd.DataFrame):
            measurements = [measurements]
        
        num_genes = measurements[0].shape[0]
        
        if num_genes < maxK:
            maxK = num_genes
            print(f"Warning: maxK was chosen greater than the total number of input genes and reset to {num_genes}!")

        if perturbations is None and hasattr(measurements, 'perturbations'):
            perturbations = measurements['perturbations']
            del measurements['perturbations']
        
        if perturbations is not None:
            if not np.all(np.isin(perturbations, [0, 1, -1, np.nan])):
                raise ValueError("The perturbation matrix may only contain the values 0,1,-1 and NaN!")
            perturbations = np.nan_to_num(perturbations, nan=-1)

            if perturbations.ndim == 1:
                perturbations = perturbations[:, np.newaxis]
            if perturbations.shape[1] != 1 and perturbations.shape[1] != len(measurements):
                raise ValueError("There must be either one global vector of perturbations, or a matrix containing exact one column for each time series!")
            if perturbations.shape[0] != num_genes:
                raise ValueError("The perturbation matrix must have exactly the same number of rows as the measurements!")
        
        perturbation_matrix = []
        gene_names = measurements[0].index.tolist() if hasattr(measurements[0], 'index') else [f"Gene{i+1}" for i in range(num_genes)]
        input_states = []
        output_states = []
        
        for measurement in measurements:
            if num_genes != measurement.shape[0]:
                raise ValueError("All measurement matrices must contain the same genes!")
            input_states.extend(measurement.iloc[:, :-1].values.flatten().astype(int))
            output_states.extend(measurement.iloc[:, 1:].values.flatten().astype(int))
            
            if perturbations is not None:
                for j in range(measurement.shape[1] - 1):
                    if perturbations.shape[1] == 1:
                        perturbation_matrix.append(perturbations[:, 0])
                    else:
                        perturbation_matrix.append(perturbations[:, j])
        
        num_states = len(input_states) // num_genes
    
    if required_dependencies is None:
        required_dep_matrix = None
    else:
        required_dep_matrix = np.zeros((num_genes, num_genes), dtype=int)
        max_required = 0
        for gene, dependencies in required_dependencies.items():
            max_required = max(max_required, len(dependencies))
            for dep in dependencies:
                required_dep_matrix[gene_names.index(dep), gene_names.index(gene)] = 1
        
        if max_required > maxK:
            print(f"Warning: The number of required dependencies is greater than maxK! Setting maxK to {max_required}!")
            maxK = max_required
    
    if excluded_dependencies is None:
        excluded_dep_matrix = None
    else:
        excluded_dep_matrix = np.zeros((num_genes, num_genes), dtype=int)
        for gene, dependencies in excluded_dependencies.items():
            for dep in dependencies:
                excluded_dep_matrix[gene_names.index(dep), gene_names.index(gene)] = 1
    
    res = reconstruct_network_algorithm(input_states, output_states, perturbation_matrix, num_states,
                                        required_dep_matrix, excluded_dep_matrix, maxK, meth, all_solutions, return_pbn)
    
    return process_reconstruction_result(res, gene_names, readable_functions, return_pbn, all_solutions)

def reconstruct_network_algorithm(input_states, output_states, perturbation_matrix, num_states,
                                  required_dep_matrix, excluded_dep_matrix, maxK, meth, all_solutions, return_pbn):
    return []

def process_reconstruction_result(res, gene_names, readable_functions, return_pbn, all_solutions):
    return {}

#saveNetwork



#stateTransitions
def state_transition(network, state, type="synchronous", 
                     gene_probabilities=None, chosen_gene=None, 
                     chosen_functions=None, time_step=0):
    
    if not isinstance(network, dict) or "genes" not in network or "interactions" not in network:
        raise TypeError("network must be a dictionary containing 'genes' and 'interactions' keys.")
    
    if len(state) != len(network["genes"]):
        raise ValueError("state must consist of exactly one value for each gene!")
    
    non_fixed_indices = np.array([gene == -1 for gene in network["fixed"]])
    res = np.array(state, dtype=int)
    
    if type == "probabilistic":
        if chosen_functions is None:
            chosen_functions = [choose_function(gene) for gene in network["interactions"]]
        elif len(chosen_functions) != len(network["genes"]):
            raise ValueError("Please provide a function index for each gene!")

        for i, f in zip(np.where(non_fixed_indices)[0], chosen_functions[non_fixed_indices]):
            if network["interactions"][i][f]["input"][0] == 0:
                res[i] = network["interactions"][i][f]["func"][0]
            else:
                input_vals = state[network["interactions"][i][f]["input"]]
                res[i] = network["interactions"][i][f]["func"][bin2dec(np.flip(input_vals), len(input_vals))]
    else:
        change_indices = get_change_indices(network, type, non_fixed_indices, gene_probabilities, chosen_gene)
        for i in change_indices:
            if network["interactions"][i]["input"][0] == 0:
                res[i] = network["interactions"][i]["func"][0]
            else:
                input_vals = state[network["interactions"][i]["input"]]
                res[i] = network["interactions"][i]["func"][bin2dec(np.flip(input_vals), len(input_vals))]
    
    res[~non_fixed_indices] = network["fixed"][~non_fixed_indices]
    
    res_dict = dict(zip(network["genes"], res))
    return res_dict


def choose_function(gene):
    distr = np.cumsum([0] + [func["probability"] for func in gene])
    r = np.random.rand()
    for i, val in enumerate(distr[:-1]):
        if val < r <= distr[i+1]:
            return i
    return len(distr) - 2


def get_change_indices(network, type, non_fixed_indices, gene_probabilities, chosen_gene):
    if type == "synchronous":
        return np.where(non_fixed_indices)[0]
    
    elif type == "asynchronous":
        if chosen_gene is None:
            if gene_probabilities is None:
                return np.random.choice(np.where(non_fixed_indices)[0], 1)
            else:
                if len(gene_probabilities) != len(network["genes"]):
                    raise ValueError("Please supply exactly one probability for each gene!")
                if abs(1.0 - np.sum(gene_probabilities)) > 0.0001:
                    raise ValueError("The supplied gene probabilities do not sum up to 1!")
                if np.sum(gene_probabilities[non_fixed_indices]) < 0.0001:
                    raise ValueError("There is no non-fixed gene with a probability greater than 0!") 
                
                gene_probabilities[non_fixed_indices] /= np.sum(gene_probabilities[non_fixed_indices])
                gene_probabilities[~non_fixed_indices] = 0
                distr = np.cumsum([0] + gene_probabilities.tolist())
                r = np.random.rand()
                return [np.searchsorted(distr, r)]
        else:
            if isinstance(chosen_gene, str):
                chosen_gene_idx = np.where(network["genes"] == chosen_gene)[0]
                if len(chosen_gene_idx) == 0:
                    raise ValueError(f"Gene '{chosen_gene}' does not exist in the network!")
                return chosen_gene_idx
            else:
                return [chosen_gene]
            
#symbolicToTruthTable 



#testNetworkProperties
