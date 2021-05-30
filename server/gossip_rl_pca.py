import client_gossip
import load_data
import RL
import logging
import numpy as np
import pickle
import random
import sys
from threading import Thread
import torch
import utils.dists as dists  # pylint: disable=no-name-in-module
import ast
import copy
import time
import os
import math
from sklearn.decomposition import PCA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Gossip_RL_pca(object):
    """Basic federated learning server."""

    def __init__(self, config):
        self.config = config
        # maximum available degree of each node
        # can modify according to number of nodes, network condiction and computation capability
        # e.g. for a 32 nodes [6, 8, 6, 4, 7, 8, 7, 5, 6, 7, 5, 4, 5, 5, 5, 7, 4, 8, 4, 7, 4, 5, 5, 10, 5, 4, 6, 8, 7, 8, 4, 7]
        self.band_degree = \
        [6, 8, 6, 4, 7, 8, 7, 5, 6, 7, 5, 4, 5, 5, 5, 7, 4, 8, 4, 7, 4, 5, 5, 10, 5, 4, 6, 8, 7, 8, 4, 7]
        self.data_dist = {} 
        self.mean_loss = []
        self.BATCH_SIZE = 128
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.NUM_EPISODE = 200
        self.dim = len(self.band_degree)
        self.pca = PCA(n_components=self.dim)
    # Set up server
    def rl(self):
        n_clients = self.config.clients.total
        D = self.config.degree
        self.steps_done = 0
        self.invalid_action = []
        self.network_limit_action = []
        self.reward_history = []

        self.num_state = self.dim * n_clients
        self.num_action = ((n_clients*(n_clients-1))//2 - D*n_clients) * 2 + 1
        self.policy_network = RL.DQN(self.num_state, self.num_action)
        self.target_network = RL.DQN(self.num_state, self.num_action)
        

        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), amsgrad=True)
        self.memory = RL.ReplayBuffer(10000)
        for i_episode in range(self.NUM_EPISODE):
            self.i_episode = i_episode
            self.use_action = []
            self.reward_sum = 0
            self.boot()
            self.run()
            torch.save(self.policy_network, 'RL_model')
            if i_episode > 0 and i_episode % 5 == 0:
                self.update_target_weight()
    
    
    def check_band_degree(self, action):
        if action == 0:
            return True
        vertex = []
        # check the action table
        if action < self.action_remove:
            for i in range(len(self.current_matrix[0])):
                for j in range(len(self.current_matrix[0])):
                    if self.action_table[i][j] == action:
                        vertex.append([i,j])
                        break
        else:
            return True
        ###########################
        if self.current_matrix[vertex[0][0]][vertex[0][1]] and self.current_matrix[vertex[0][1]][vertex[0][0]]:
            return True

        degree_1 = self.count_degree_at_target_node(self.current_matrix, vertex[0][0]) + 1
        degree_2 = self.count_degree_at_target_node(self.current_matrix, vertex[0][1]) + 1
        if (degree_1 <= self.band_degree[vertex[0][0]]) and (degree_2 <= self.band_degree[vertex[0][1]]):
            return True
        else:
            return False

    def boot(self):
        logging.info('Booting {} server...'.format(self.config.server))

        model_path = self.config.paths.model
        total_clients = self.config.clients.total
        # Add fl_model to import path
        sys.path.append(model_path)

        # Set up simulated server
        self.load_data()
        self.load_model()
        self.make_clients(total_clients)

    def load_data(self):
        import fl_model  # pylint: disable=import-error

        # Extract config for loaders
        config = self.config
        # Set up data generator
        generator = fl_model.Generator()

        # Generate data
        data_path = self.config.paths.data
        data = generator.generate(data_path)
        labels = generator.labels

        logging.info('Dataset size: {}'.format(
            sum([len(x) for x in [data[label] for label in labels]])))
        logging.info('Labels ({}): {}'.format(
            len(labels), labels))

        # Set up data loader
        self.loader = {
            'basic': load_data.Loader(config, generator),
            'bias': load_data.BiasLoader(config, generator),
            'shard': load_data.ShardLoader(config, generator)
        }[self.config.loader]

        logging.info('Loader: {}, IID: {}'.format(
            self.config.loader, self.config.data.IID))

    def load_model(self):
        import fl_model  # pylint: disable=import-error

        model_path = self.config.paths.model
        model_type = self.config.model

        logging.info('Model: {}'.format(model_type))

        # Set up global model
        self.model = fl_model.Net()
        self.save_model(self.model, model_path, '')

        # Extract flattened weights (if applicable)
        if self.config.paths.reports:
            self.saved_reports = {}
            self.save_reports(0, [])  # Save initial model

    def update_parameters(self):
        
        if len(self.memory) < self.BATCH_SIZE:
            return 
        
        loss_fn = torch.nn.MSELoss(reduction = 'mean') #define loss function mean square error
        
        batch = self.memory.sample(self.BATCH_SIZE)
        states , actions , rewards , next_states = batch
        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions).view(-1,1)
        rewards = torch.from_numpy(rewards)
        next_states = torch.from_numpy(next_states)
        actions = actions.long()
        
        pred_q = self.policy_network(states).gather(1 , actions).view(-1) 
        #see the action we take that time
        
        next_state_value = torch.zeros(self.BATCH_SIZE).detach()
        expected_q = (next_state_value + rewards).detach()
        
        
        loss = loss_fn(pred_q , expected_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_weight(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def make_clients(self, num_clients):
        IID = self.config.data.IID
        labels = self.loader.labels
        loader = self.config.loader
        loading = self.config.data.loading

        if not IID:  # Create distribution for label preferences if non-IID
            dist = {
                "uniform": dists.uniform(num_clients, len(labels)),
                "normal": dists.normal(num_clients, len(labels))
            }[self.config.clients.label_distribution]
            random.shuffle(dist)  # Shuffle distribution

        # Make simulated clients
        clients = []
        label_count = 0
        for client_id in range(num_clients):

            # Create new client
            new_client = client_gossip.Client(client_id)

            if not IID:  # Configure clients for non-IID data
                if self.config.data.bias:
                    # Bias data partitions
                    bias = self.config.data.bias
                    # Choose weighted random preference
                    pref = random.choices(labels, dist)[0]
                    new_client.set_bias(pref, bias)
                elif self.config.data.shard:
                    # Shard data partitions
                    shard = self.config.data.shard
                    # Assign shard config
                    new_client.set_shard(shard)

            clients.append(new_client)

        if loader == 'bias':
            logging.info('Label distribution: {}'.format(
                [[client.pref for client in clients].count(label) for label in labels]))

        if loading == 'static':
            if loader == 'shard':  # Create data shards
                self.loader.create_shards()

            # Send data partition to all clients
            [self.set_client_data(client) for client in clients]

        self.clients = clients

    # Run federated learning
    def run(self):
        rounds = self.config.fl.rounds
        target_accuracy = self.config.fl.target_accuracy
        reports_path = self.config.paths.reports
        if target_accuracy:
            logging.info('Training: {} rounds or {}% accuracy\n'.format(
                rounds, 100 * target_accuracy))
        else:
            logging.info('Training: {} rounds\n'.format(rounds))
        # Perform rounds of federated learning

        for round in range(1, rounds + 1):
            self.round_num = round
            logging.info('**** Round {}/{} ****'.format(round, rounds))

            # Run the federated learning round
            accuracy = self.round(round)
            # Break loop when target accuracy is met
            if target_accuracy and (accuracy >= target_accuracy):
                logging.info('Target accuracy reached.')
                break

        if reports_path:
            with open(reports_path, 'wb') as f:
                pickle.dump(self.saved_reports, f)
            logging.info('Saved reports: {}'.format(reports_path))
        

    def round(self, num_round):
        import fl_model  # pylint: disable=import-error
        self.num_round = num_round # calculate number of round
        N = self.config.clients.total
        if self.config.data.IID:
            D = "IID"
        else:
            D = self.config.data.bias['primary']
        R = self.config.degree
        S = self.config.selectneighbor
        # Select clients to participate in the round
        sample_clients = self.selection()
        # Configure sample clients
        self.configuration(sample_clients, num_round)

        # Run clients using multithreading for better parallelism
        threads = [Thread(target=client.run) for client in sample_clients]
        [t.start() for t in threads]
        [t.join() for t in threads]

        # Recieve client updates
        reports = self.reporting(sample_clients)
        # Perform weight aggregation

        logging.info('Aggregating updates')
        new_weight = self.aggregation(reports)
        #######new_weight
        # Load updated weights
        testset = self.loader.get_testset()
        batch_size = self.config.fl.batch_size
        testloader = fl_model.get_testloader(testset, batch_size)
        accuracy_list = []
        rsample = random.sample(range(0, N), 8)
        for key in new_weight:
            fl_model.load_weights(self.model, new_weight[key])
            self.save_model(self.model, self.config.paths.model, key)
            if self.action == 0:
                if key in rsample: # random sample node to test accuracy
                    self.model.to(device)
                    self.model.eval()
                    accuracy = fl_model.test(self.model, testloader)
                    accuracy_list.append(accuracy)
                    logging.info('Node #{} Average accuracy: {:.2f}%\n'.format(key, 100 * accuracy))
            else:
                if key in self.action_node:
                    self.model.to(device)
                    self.model.eval()
                    accuracy = fl_model.test(self.model, testloader)
                    accuracy_list.append(accuracy)
                    logging.info('Node #{} Average accuracy: {:.2f}%\n'.format(key, 100 * accuracy))
        mean_accuracy = sum(accuracy_list)/len(accuracy_list)
        accuracy_list.sort()
        ########## get the reward and store in replay memory #############
        if (num_round > 1):
            self.p_reward = copy.deepcopy(self.reward)
        self.reward = self.reward_fun(mean_accuracy, self.config.fl.target_accuracy)
        self.reward_sum = self.reward_sum + self.reward
        if (num_round > 1):
            self.memory.add(self.p_state, self.p_action, self.p_reward, self.state)
            self.update_parameters()
        ############### record average loss ################
        avg_loss = []
        for report in reports:
            avg_loss.append(report.loss)
        self.mean_loss.append(sum(avg_loss) / len(avg_loss))
        return mean_accuracy

    # Federated learning phases

    def selection(self):
        # Select devices to participate in round
        clients_per_round = self.config.clients.per_round
        # print("clients per round: ", clients_per_round)
        # Select clients randomly
        sample_clients = [client for client in random.sample(
            self.clients, clients_per_round)]

        return sample_clients

    def configuration(self, sample_clients, num_round):
        loader_type = self.config.loader
        loading = self.config.data.loading

        if loading == 'dynamic':
            # Create shards if applicable
            if loader_type == 'shard':
                self.loader.create_shards()

        # Configure selected clients for federated learning task
        for client in sample_clients:
            if loading == 'dynamic':
                self.set_client_data(client)  # Send data partition to client

            # Extract config for client
            config = self.config

            # Continue configuraion on client
            client.configure(config, num_round)

    def reporting(self, sample_clients):
        # Recieve reports from sample clients
        reports = [client.get_report() for client in sample_clients]
        logging.info('Reports recieved: {}'.format(len(reports)))
        assert len(reports) == len(sample_clients)

        return reports

    def aggregation(self, reports):
        return self.federated_averaging(reports)

    # Report aggregation
    def extract_client_updates(self, reports):
        import fl_model  # pylint: disable=import-error

        # Extract baseline model weights
        baseline_weights = fl_model.extract_weights(self.model)

        # Extract weights from reports
        weights = [report.weights for report in reports]

        # Calculate updates from weights
        updates = []
        for weight in weights:
            update = []
            for i, (name, weight) in enumerate(weight):
                bl_name, baseline = baseline_weights[i]

                # Ensure correct weight is being updated
                assert name == bl_name

                # Calculate update
                delta = weight - baseline
                update.append((name, delta))
            updates.append(update)

        return updates

    def env_graph(self, current_gm, action):
        # use action to generate graph_matrix
        # search the action from the action table
        n_clients = self.config.clients.total
        next_gm = copy.deepcopy(current_gm)
        if action != 0:
            if action < self.action_remove:
                for i in range(len(self.action_table[0])):
                    for j in range(len(self.action_table[0])):
                        if (self.action_table[i][j] == action):
                            next_gm[i][j] = 1
                            next_gm[j][i] = 1
                            self.action_node = [i,j]
            else:
                for i in range(len(self.action_table_2[0])):
                    for j in range(len(self.action_table_2[0])):
                        if (self.action_table_2[i][j] == action):
                            next_gm[i][j] = 0
                            next_gm[j][i] = 0
                            self.action_node = [i,j]
        HT = self.hitting_time(self.generate_comm_matrix(next_gm))
        self.HT = []
        self.state_bias =[]
        for i in range(1, n_clients+1):
            self.HT.append(max(HT[(i-1)*n_clients:i*n_clients-1])[0])
        for i in range(n_clients):
            self.state_bias.append(self.get_group_bias(next_gm, i))
        next_state = torch.Tensor(self.HT + self.state_bias)
        return next_state, next_gm

    def read_graph(self):
        s_neighbor = self.config.selectneighbor
        D = self.config.degree
        n_clients = self.config.clients.total
        connect_list = []
        
        ######## action ################
        if (self.num_round == 1): # number of round start from 1
            self.construct_data_counterbalance_regular_graph(n_clients) # create initial graph
            path = './graph/D_{}_C_{}_CONT_{}.txt'.format(D, n_clients, s_neighbor)
            self.g_original_matrix = self.read_graph_as_matrix(path)
            self.current_matrix = copy.deepcopy(self.g_original_matrix)
            self.create_action_table()
            HT = self.hitting_time(self.generate_comm_matrix(self.current_matrix))
            self.state = self.pca_state
            
            self.action = self.take_action(self.state)

            if self.action != 0:
                self.invalid_action.append(self.action)
            self.next_HT_Bias, self.next_graph = self.env_graph(self.current_matrix, self.action) # input an action and return a state (Bias + HT 先 HT 後 Bias)
        
        else:
            ########### action here ###################
            self.current_matrix = self.next_graph
            self.p_state = copy.deepcopy(self.state)
            self.p_action = copy.deepcopy(self.action)
            ######################################
            self.state = self.pca_state
            self.action = self.take_action(self.state)
            if self.action != 0:
                self.invalid_action.append(self.action)
            self.next_HT_Bias, self.next_graph = self.env_graph(self.current_matrix, self.action)
            ###########################################
        self.create_graph_by_matrix(self.current_matrix)
        rgraph = './graph/RL_graph/EP_{}/EP_{}_Round_{}.txt'\
                    .format(self.i_episode, self.i_episode, self.round_num)

        with open(rgraph, 'r') as g:
            n = g.readline()
            num_nodes = int(n[:-1])
            while True:
                line = g.readline()
                if not line:
                    break
                a = line[:-1].split()
                connect_list.append([int(a[0]), int(a[1])])
            g.close()
        return num_nodes, connect_list
    
    def create_action_table(self):
        n_clients = self.config.clients.total
        o_graph_matrix = copy.deepcopy(self.g_original_matrix)
        o_graph_matrix_2 = copy.deepcopy(self.g_original_matrix)
        action_matrix = copy.deepcopy(self.g_original_matrix) 
        action_matrix_2 = copy.deepcopy(self.g_original_matrix)
        ###### initialize action matrix #########
        for i in range(n_clients):
            for j in range(n_clients):
                if ((action_matrix[i][j] == 1) or (i == j)):
                    action_matrix[i][j] = -1
                    action_matrix_2[i][j] = -1
        #########################################
        give_num = 1
        for i in range(n_clients):
            for j in range(n_clients):
                if ((i != j) and (o_graph_matrix[i][j] != 1)):
                    o_graph_matrix[i][j] = 1
                    o_graph_matrix[j][i] = 1
                    action_matrix[i][j] = give_num
                    action_matrix[j][i] = give_num
                    give_num = give_num + 1
        self.action_remove = copy.deepcopy(give_num)
        for i in range(n_clients):
            for j in range(n_clients):
                if ((i != j) and (o_graph_matrix_2[i][j] != 1)):
                    o_graph_matrix_2[i][j] = 1
                    o_graph_matrix_2[j][i] = 1
                    action_matrix_2[i][j] = give_num
                    action_matrix_2[j][i] = give_num
                    give_num = give_num + 1
        self.action_table = action_matrix
        self.action_table_2 = action_matrix_2
        
    def construct_standard_graph(self, n_clients):
        s_neighbor = self.config.selectneighbor
        D = self.config.degree
        if not os.path.isdir("./graph"):
            os.mkdir("./graph/")
        with open('./graph/D_{}_C_{}_CONT_{}.txt'.format(D, n_clients, s_neighbor), 'w+') as g:
            g.write(str(n_clients) + '\n')
            for j in range(n_clients):
                for i in range(1,(D//2)+1):
                    g.write(str(j) + ' ' + str((j+n_clients-i)%n_clients) + '\n')
                    g.write(str(j) + ' ' + str((j+n_clients+i)%n_clients) + '\n')
            g.close()
    
    def construct_data_counterbalance_regular_graph(self, n_clients):
        s_neighbor = self.config.selectneighbor
        D = self.config.degree
        c_graph = self.complete_graph()
        neighbor_list = self.neighbor_selection(c_graph)
        if not os.path.isdir("./graph"):
            os.mkdir("./graph/")
        with open('./graph/D_{}_C_{}_CONT_{}.txt'.format(D, n_clients, s_neighbor), 'w+') as g:
            g.write(str(n_clients) + '\n')
            for i in neighbor_list:
                for j in range(1, (D//2) + 1):
                    g.write(str(i) + ' ' + str(neighbor_list[(neighbor_list.index(i) + n_clients + j) % n_clients]) + '\n')
                    g.write(str(i) + ' ' + str(neighbor_list[(neighbor_list.index(i) + n_clients - j) % n_clients]) + '\n')
            g.close()

    def bias_value(self, data_set):
        # data_set : {ID1:[#datapoint1,#datapoint2...], ID2....}
        L = len(self.loader.labels)
        dist_list = np.zeros(L)
        for key in data_set:
            dist_list = dist_list + np.array(data_set[key])
        mean_value = np.mean(dist_list)
        b_value = 0
        for val in dist_list:
            b_value += (val - mean_value) ** 2
        return b_value
    
    def complete_graph(self):
        data_dist = self.data_dist # data distribution {}
        n_clients = self.config.clients.total
        column, row = n_clients, n_clients
        c_graph = np.zeros((column, row)) # n_client * n_client 的矩陣存 bias value
        pair_list = []
        for i in range(n_clients):
            for j in range(i+1, n_clients):
                pair_list.append([i, j])
        for i in range(n_clients):
            c_graph[i][i] = float("inf")
        for pair in pair_list:
            data_set = {}
            data_set[pair[0]] = data_dist[pair[0]]
            data_set[pair[1]] = data_dist[pair[1]]
            val = self.bias_value(data_set)
            c_graph[pair[0]][pair[1]] = val
            c_graph[pair[1]][pair[0]] = val
        return c_graph

    def construct_regular_graph_matrix(self, target_list):
        n_clients = self.config.clients.total
        D = self.config.degree
        column, row = n_clients, n_clients
        graph = np.zeros((column, row))
        for i in target_list:
            for j in range(1, (D//2) + 1):
                graph[i][target_list[(target_list.index(i) + n_clients + j) % n_clients]] = 1
                graph[target_list[(target_list.index(i) + n_clients + j) % n_clients]][i] = 1
                graph[i][target_list[(target_list.index(i) + n_clients - j) % n_clients]] = 1
                graph[target_list[(target_list.index(i) + n_clients - j) % n_clients]][i] = 1
        return graph
    
    def neighbor_selection(self, c_graph):
        D = self.config.degree
        n_clients = self.config.clients.total
        assert D%2 == 0 # D must be even number
        D = D//2
        n_list = [] # neighbor list
        r_list = [] # rest node list
        for i in range(1, n_clients): # initialize rest node list
            r_list.append(i)
        start_node = 0
        n_list.append(start_node)
        index = self.find_n_minimal(c_graph[0], D)
        n_list.extend(index)
        # remove nodes in rest node list
        for node in n_list:
            if node in r_list:
                r_list.remove(node)
        # select neighbor
        while(len(r_list) != 0):
            templist = []
            for rest_node in r_list:
                temp = 0
                for sel_node in n_list[-D:]:
                    temp = temp + c_graph[sel_node][rest_node]
                templist.append(temp)
            selected_node = r_list[templist.index(min(templist))]
            n_list.append(selected_node)
            r_list.remove(selected_node)
        return n_list
    
    def find_n_minimal(self, nplist, n):
        lest_n_min = []
        lest_n_min_list = []
        coplist = copy.deepcopy(nplist)
        for i in range(n):
            minimal = np.amin(nplist)
            lest_n_min.append(minimal)
            b = np.where(nplist == minimal)
            nplist = np.delete(nplist, b[0][0])
        for m in range(len(lest_n_min)):
            if ((m == 0) or ((m!=0) and lest_n_min[m]!=lest_n_min[m-1])):
                ac = 0
                lest_n_min_list.append((np.where(coplist == lest_n_min[m])[0][ac]))
            else:
                ac += 1
                lest_n_min_list.append((np.where(coplist == lest_n_min[m])[0][ac]))
        return lest_n_min_list

    def create_graph_by_matrix(self, graph_matrix):
        s_neighbor = self.config.selectneighbor
        D = self.config.degree
        n_clients = self.config.clients.total
        if (self.config.server == "gossip_rl_pca"):
            if not os.path.isdir("./graph/RL_graph"):
                os.mkdir("./graph/RL_graph")
            if not os.path.isdir("./graph/RL_graph/EP_{}".format(self.i_episode)):
                os.mkdir("./graph/RL_graph/EP_{}".format(self.i_episode))
            with open('./graph/RL_graph/EP_{}/EP_{}_Round_{}.txt'\
.format(self.i_episode, self.i_episode, self.round_num), 'w+') as gg:
                gg.write(str(n_clients) + '\n')
                for i in range(n_clients):
                    for j in range(n_clients):
                        if (graph_matrix[i][j] == 1):
                            gg.write(str(i) + ' ' + str(j) + '\n')
                gg.close()
    
    def get_group_bias(self, graph_matrix, target_node):
        neighbor_group = {}
        data_dist = self.data_dist
        neighbor_group[target_node] = data_dist[target_node] 
        for i in range(len(graph_matrix[target_node])):
            if graph_matrix[target_node][i] == 1:
                neighbor_group[i] = data_dist[i]
        val = self.bias_value(neighbor_group)
        return val

    def count_degree(self, num_node, connect_list):
        degree = {}
        for i in range(num_node):
            degree[i] = 0
        for pair in connect_list:
            degree[pair[0]] += 1
        return degree

    def get_neighbor(self, connect_graph):
        neighbor = {}
        for pair in connect_graph:
            if pair[0] in neighbor:
                neighbor[pair[0]].append(pair[1])
            else:
                neighbor[pair[0]] = [pair[1]]
        return neighbor

    def read_graph_as_matrix(self, path):
        connect_list = []
        with open(path, 'r') as g:
            n = g.readline()
            graph_matrix = np.zeros((int(n),int(n)))
            num_nodes = int(n[:-1])
            while True:
                line = g.readline()
                if not line:
                    break
                a = line[:-1].split()
                graph_matrix[int(a[0])][int(a[1])] = 1
            g.close()
        return graph_matrix
    def federated_averaging(self, reports):
        import fl_model  # pylint: disable=import-error
        n_clients = self.config.clients.total
        weight_dict = {} # {ID: weights}
        weight_delta = {} # {Node ID: weights sum}
        for report in reports:
            weight_dict[report.client_id] = report.weights
        ############# set pca state ########################
        self.pca_state = self.pca_env(weight_dict)
        
        ####################################################
        num_node, connect_graph = self.read_graph() # [[1, 3], [2, 4], [3, 2], [4, 1]]
        degree = self.count_degree(num_node, connect_graph)
        neighbor = self.get_neighbor(connect_graph) # {0: [1, 2, 3, 4], 1: [2, 3], 2: [1], 3: [1, 2], 4: [2], 5: [3], 6: [4], 7: [0, 3]}
        
        for i in range(num_node):
            delta = {}
            for n in neighbor[i]:
                for j, (name, weight) in enumerate(weight_dict[n]):
                    b_name, b_weight = weight_dict[i][j]
            ############ matropolic update ################
                    assert name == b_name
                    if name in delta:
                        delta[name].append((1/max(degree[i], degree[n])) * (weight - b_weight))
                    else:
                        delta[name] = [(1/max(degree[i], degree[n])) * (weight - b_weight)]
            ########## sum the value in delta ############
            update = []
            for key in delta:
                item = delta[key][0]
                for c in range(1, len(delta[key])):
                    item = item + delta[key][c] # += ??????
                update.append((key, item))
            weight_delta[i] = update
        new_weight = {}
        ########## update new weight each node ##############
        for i in range(num_node):
            up_weight = []
            for j, (name, base) in enumerate(weight_dict[i]):
                nei_name, nei_weight = weight_delta[i][j]
                up_weight.append((name, base + nei_weight))
            new_weight[i] = up_weight

        return new_weight
    def pca_env(self, dict):
        state = []
        n_clients = self.config.clients.total
        for i in range(n_clients):
            fw = self.flatten_weights(dict[i])
            state.append(fw)
        s = self.pca.fit_transform(state)
        flatten_state = []
        for p in s:
            flatten_state.extend(p)
        return torch.Tensor(flatten_state)
    def reward_fun(self, avg_accu, tar_accu):
        CONSTANT = 64
        reward = (CONSTANT**(avg_accu - tar_accu)) - 1
        return reward
    
    def generate_comm_matrix(self, graph):
        comm_matrix = np.zeros((len(graph[0]), len(graph[0])))
        for i in range(len(graph[0])):
            a = self.count_degree_at_target_node(graph, i)
            for j in range(len(graph[0])):
                if ((i != j) and (graph[i][j] != 0)):
                    b = self.count_degree_at_target_node(graph, j)
                    m = max(a,b)
                    comm_matrix[i][j] = 1/(2*m)
            comm_matrix[i][i] = 1 - sum(comm_matrix[i])
        return comm_matrix

    def count_degree_at_target_node(self, graph, vertex):
        degree = 0
        for i in range(len(graph[0])):
            if (graph[vertex][i] == 1):
                degree = degree + 1
        return degree
    

    def count_degree(self, num_node, connect_list):
        degree = {}
        for i in range(num_node):
            degree[i] = 0
        for pair in connect_list:
            degree[pair[0]] += 1
        return degree
    

    def hitting_time(self, comm_matrix):
        n = self.config.clients.total * self.config.clients.total
        LHS = np.zeros((n, n))
        RHS = np.zeros((n, 1))
        for row in range(len(comm_matrix)):
            for col in range(len(comm_matrix)):
                if (row == col):
                    LHS[row*(1+len(comm_matrix))][col*(1+len(comm_matrix))] = 1
                    RHS[col*(1+len(comm_matrix))][0] = 0
                else:
                    for i in range(len(comm_matrix[row])):
                        if (comm_matrix[row][i] != 0) and (i != col):
                            LHS[row*(len(comm_matrix))+col][i*len(comm_matrix)+col] = comm_matrix[row][i]
                            if (i*len(comm_matrix)+col) == (row*(len(comm_matrix))+col):
                                LHS[row*(len(comm_matrix))+col][i*len(comm_matrix)+col] = LHS[row*(len(comm_matrix))+col][i*len(comm_matrix)+col]-1
                    RHS[row*(len(comm_matrix))+col][0] = -1
        LHS_inv = np.linalg.inv(LHS)
        ans = LHS_inv.dot(RHS)
        return ans

    def take_action(self, state, is_testing = False):
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        rand_val = np.random.uniform()
        if ((rand_val > eps_threshold) or (is_testing == True)):
            val = self.policy_network(state)
            action_list = torch.topk(val, len(val))[1]
            satisfy = False
            for a in action_list:
                if (a.item() not in self.network_limit_action):
                    action = a.item()
                    if action != 0:
                        satisfy = self.check_band_degree(action)
                        if satisfy == False:
                            self.network_limit_action.append(action)
                            assert action not in self.use_action

                    if (action == 0) or (satisfy == True):
                        break
            ################################################################
        else:
            while(True):
                action = np.random.\
                    choice([i for i in range(0, self.num_action) if i not in self.network_limit_action])
                if action == 0:
                    break
                elif self.check_band_degree(action) == False:
                    self.invalid_action.append(action)
                    self.network_limit_action.append(action)
                    assert action not in self.use_action
                else:
                    break
        if is_testing == False:
            self.steps_done += 1
        if action not in self.use_action:
            self.use_action.append(action)
        return action

    # Server operations
    @staticmethod
    def flatten_weights(weights):
        # Flatten weights into vectors
        weight_vecs = []
        for _, weight in weights:
            weight_vecs.extend(weight.flatten().tolist())

        return np.array(weight_vecs)

    def set_client_data(self, client):
        loader = self.config.loader

        # Get data partition size
        if loader != 'shard':
            if self.config.data.partition.get('size'):
                partition_size = self.config.data.partition.get('size')
            elif self.config.data.partition.get('range'):
                start, stop = self.config.data.partition.get('range')
                partition_size = random.randint(start, stop)

        # Extract data partition for client
        if loader == 'basic':
            data, distr = self.loader.get_partition(partition_size)
        elif loader == 'bias':
            data, distr = self.loader.get_partition(partition_size, client.pref)
        elif loader == 'shard':
            data = self.loader.get_partition()
        else:
            logging.critical('Unknown data loader type')

        # Send data to client
        self.data_dist[client.client_id] = distr
        client.set_data(data, self.config)

    def save_model(self, model, path, ID):
        path += '/global{}'.format(ID)
        torch.save(model.state_dict(), path)
        logging.info('Saved global model: {}'.format(path))

    def save_reports(self, round, reports):
        import fl_model  # pylint: disable=import-error

        if reports:
            self.saved_reports['round{}'.format(round)] = [(report.client_id, self.flatten_weights(
                report.weights)) for report in reports]

        # Extract global weights
        self.saved_reports['w{}'.format(round)] = self.flatten_weights(
            fl_model.extract_weights(self.model))
