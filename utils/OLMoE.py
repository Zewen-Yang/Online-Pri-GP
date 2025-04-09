# Copyright (c) by Zewen Yang under GNU General Public License v3 (GPLv3)
# Last modified: Zewen Yang 11/2024

import math
import numpy as np
from scipy.stats import norm
from utils.GPmodel import GPmodel 
from utils.common import * 


class OLMoE:
    def __init__(self, indivDataThersh, x_dim, y_dim, 
                 sigmaN, sigmaF, sigmaL, 
                 priorFuncList, agentQuantity, Graph):
        self.indivDataThersh = indivDataThersh
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.sigmaN = sigmaN
        self.sigmaF = sigmaF
        self.sigmaL = sigmaL
        self.priorFunc_list = priorFuncList
        self.agentQuantity = agentQuantity
        self.agents = []
        self.Graph = Graph

        self.error_list = [[[] for _ in range(agentQuantity)] for _ in range(agentQuantity)]
        self.neighbors_list = [[] for _ in range(agentQuantity)]
        self.requestNieghbors_list = [[] for _ in range(agentQuantity)]
        self.aggWieghts_list = [[] for _ in range(agentQuantity)]
        self.ordered_error_list = [[] for _ in range(agentQuantity)]
        self.largsetIndices_list = [[] for _ in range(agentQuantity)]
        self.MASpredictTimes_list = [[0] for _ in range(agentQuantity)]
        self.deleteNieghborsQuantity_list = [1 for _ in range(agentQuantity)]


        for i in range(agentQuantity):
            self.agents.append(GPmodel(indivDataThersh, x_dim, y_dim, 
                                      sigmaN, sigmaF, sigmaL, 
                                      priorFuncList[i], agentIndex=i))
            self.neighbors_list[i] = list(self.Graph.neighbors(i))

    def agentUpdateOnce(self, agentIndex, x, y):
        self.agents[agentIndex].addDataOnce(x, y)
        self.agents[agentIndex].errorRecord(x, y)
        self.agents[agentIndex].updateKmatOnce()

    def agentUpdateEntire(self, agentIndex, x, y):
        self.agents[agentIndex].addDataEntire(x, y)
        self.agents[agentIndex].updateKmatEntire()

    def requestWhichNeighbors_initial(self, agentIndex):
        temp_neighbors_list = self.neighbors_list[agentIndex]
        temp_weights = equalProportions(len(self.neighbors_list[agentIndex]))
        self.requestNieghbors_list[agentIndex] = temp_neighbors_list
        self.aggWieghts_list[agentIndex] = temp_weights

    def requestWhichNeighbors(self, agentIndex):
        temp_neighbors_list = self.neighbors_list[agentIndex]
        for agentIndex_neighbor in temp_neighbors_list:
            neighbor_totalError = self.agents[agentIndex_neighbor].priorError_list
            neighbor_error = np.sum(neighbor_totalError)/len(neighbor_totalError)
            self.error_list[agentIndex][agentIndex_neighbor] = neighbor_error

        temp_errorlist =  self.error_list[agentIndex] 
        non_empty_values = [item for item in temp_errorlist  if not isinstance(item, list) or np.size(item) > 0]
        non_empty_values = minmaxScaling(non_empty_values)
        self.ordered_error_list[agentIndex] = non_empty_values
        deleteNieghborsQuantity = self.deleteNieghborsQuantity_list[agentIndex]
        sorted_indices = np.argsort(-np.array(non_empty_values))
        largset_indices = sorted_indices[:deleteNieghborsQuantity].flatten()
        self.largsetIndices_list[agentIndex] = largset_indices

        sort_agentList = np.sort(temp_neighbors_list)
        requestNieghborsList = np.delete(sort_agentList,largset_indices, axis=0)
        self.requestNieghbors_list[agentIndex] = requestNieghborsList
        mean = non_empty_values[largset_indices.item()]
        std_dev = 0.25
        temp_weights = 1/norm.pdf(non_empty_values, loc=mean, scale=std_dev)
        temp_weights = np.delete(temp_weights, largset_indices, axis=0)
        self.aggWieghts_list[agentIndex] = temp_weights

    def predict_Pri_initial(self, agentIndex, x):
            temp_agents_list = self.requestNieghbors_list[agentIndex]
            temp_agents_list = np.sort(temp_agents_list)
            temp_weights = self.aggWieghts_list[agentIndex]
            temp_weights = getProportions(temp_weights)
            temp_mu_list = []
            temp_var_list = []
            weight_list = []
            for i in range(len(temp_agents_list)):
                act_agent = temp_agents_list[i]
                weight = temp_weights[i]
                mu, var = self.agents[act_agent].predict_initial(x)
                weight_list.append(weight)
                temp_mu_list.append(np.squeeze(mu))
                temp_var_list.append(np.squeeze(var))
            mu = np.dot(weight_list, temp_mu_list).reshape(1,-1)
            self.MASpredictTimes_list[agentIndex][0] += 1
            return mu

    def predict_Pri(self, agentIndex, x):
            temp_agents_list = self.requestNieghbors_list[agentIndex]
            temp_agents_list = np.sort(temp_agents_list)
            temp_weights = self.aggWieghts_list[agentIndex]
            temp_weights = getProportions(temp_weights)
            temp_mu_list = []
            temp_var_list = []
            weight_list = []
            for i in range(len(temp_agents_list)):
                act_agent = temp_agents_list[i]
                weight = temp_weights[i]
                mu, var = self.agents[act_agent].predict(x)
                weight_list.append(weight)
                temp_mu_list.append(np.squeeze(mu))
                temp_var_list.append(np.squeeze(var))
            mu = np.dot(weight_list, temp_mu_list).reshape(1,-1)
            self.MASpredictTimes_list[agentIndex][0] += 1
            return mu
    
    def predict_Pri_withVar(self, agentIndex, x):
        temp_agents_list = self.requestNieghbors_list[agentIndex]
        temp_agents_list = np.sort(temp_agents_list)
        temp_weights = self.aggWieghts_list[agentIndex]
        temp_mu_list = []
        weight_list = []
        for i in range(len(temp_agents_list)):
            act_agent = temp_agents_list[i]
            mu, var = self.agents[act_agent].predict(x)
            temp_mu_list.append(np.squeeze(mu))
            weight = ((temp_weights[i])**(1/2)) * ((1/var)**(1/2))
            weight_list.append(np.squeeze(weight))
        weights = getProportions(weight_list)
        mu = np.dot(weights, temp_mu_list).reshape(1,-1)
        self.MASpredictTimes_list[agentIndex][0] += 1
        return mu

    # MoE
    def predict_MoE_initial(self, i_agent, x):
        temp_agents_list = self.neighbors_list[i_agent]
        temp_agents_list = np.sort(temp_agents_list)
        temp_weights = equalProportions(len(temp_agents_list))
        temp_mu_list = []
        for i in range(len(temp_agents_list)):
            act_agent = temp_agents_list[i]
            mu, var = self.agents[act_agent].predict_initial(x)
            temp_mu_list.append(np.squeeze(mu))
        mu = np.dot(temp_mu_list, temp_weights)
        return mu
    
    def predict_MoE(self, i_agent, x):
        temp_agents_list = self.neighbors_list[i_agent]
        temp_agents_list = np.sort(temp_agents_list)
        temp_weights = equalProportions(len(temp_agents_list))
        temp_mu_list = []
        for i in range(len(temp_agents_list)):
            act_agent = temp_agents_list[i]
            mu, var = self.agents[act_agent].predict(x)
            temp_mu_list.append(np.squeeze(mu))
        mu = np.dot(temp_mu_list, temp_weights)
        return mu

    # PoE
    def predict_PoE_initial(self, i_agent, x):
        temp_agents_list = self.neighbors_list[i_agent]
        temp_agents_list = np.sort(temp_agents_list)
        temp_mu_list = []
        temp_var_list = []
        for i in range(len(temp_agents_list)):
            act_agent = temp_agents_list[i]
            mu, var = self.agents[act_agent].predict_initial(x)
            temp_mu_list.append(np.squeeze(mu))
            temp_var_list.append(np.squeeze(1/var))
        mu = np.dot(temp_mu_list, temp_var_list)
        mu = mu/np.sum(temp_var_list)
        return mu
    
    def predict_PoE(self, i_agent, x):
        temp_agents_list = self.neighbors_list[i_agent]
        temp_agents_list = np.sort(temp_agents_list)
        temp_mu_list = []
        temp_var_list = []
        for i in range(len(temp_agents_list)):
            act_agent = temp_agents_list[i]
            mu, var = self.agents[act_agent].predict(x)
            temp_mu_list.append(np.squeeze(mu))
            temp_var_list.append(np.squeeze(1/var))
        mu = np.dot(temp_mu_list, temp_var_list)
        mu = mu/np.sum(temp_var_list)
        return mu

    # gPoE
    def predict_gPoE_initial(self, i_agent, x):
        temp_agents_list = self.neighbors_list[i_agent]
        temp_agents_list = np.sort(temp_agents_list)
        temp_mu_list = []
        temp_var_list = []
        for i in range(len(temp_agents_list)):
            act_agent = temp_agents_list[i]
            mu, var = self.agents[act_agent].predict_initial(x)
            beta_k = math.log((self.sigmaF**2 + 1e-5)/var) 
            temp_mu_list.append(np.squeeze(mu))
            temp_var_list.append(np.squeeze(beta_k/(var)))
        mu = np.dot(temp_mu_list, temp_var_list)
        mu = mu/np.sum(temp_var_list)
        return mu

    def predict_gPoE(self, i_agent, x):
        temp_agents_list = self.neighbors_list[i_agent]
        temp_agents_list = np.sort(temp_agents_list)
        temp_mu_list = []
        temp_var_list = []
        for i in range(len(temp_agents_list)):
            act_agent = temp_agents_list[i]
            mu, var = self.agents[act_agent].predict(x)
            beta_k = math.log((self.sigmaF**2 + 1e-5)/var) 
            temp_mu_list.append(np.squeeze(mu))
            temp_var_list.append(np.squeeze(beta_k/(var)))
        mu = np.dot(temp_mu_list, temp_var_list)
        mu = mu/np.sum(temp_var_list)
        return mu
    
    # BCM
    def predict_BCM_initial(self, i_agent, x):
        temp_agents_list = self.neighbors_list[i_agent]
        temp_mu_list = []
        temp_var_list = []
        for i in range(len(temp_agents_list)):
            act_agent = temp_agents_list[i]
            mu, var = self.agents[act_agent].predict_initial(x)
            temp_mu_list.append(np.squeeze(mu))
            temp_var_list.append(np.squeeze(1/var))
        prior_var = (1-len(temp_agents_list))/(self.sigmaF**2+ self.sigmaN**2)
        var_BCM = np.sum(temp_var_list)+ prior_var.ravel()
        weights = temp_var_list/var_BCM
        mu_BCM = np.dot(temp_mu_list, weights)
        return mu_BCM
    
    def predict_BCM(self, i_agent, x):
        temp_agents_list = self.neighbors_list[i_agent]
        temp_mu_list = []
        temp_var_list = []
        for i in range(len(temp_agents_list)):
            act_agent = temp_agents_list[i]
            mu, var = self.agents[act_agent].predict(x)
            temp_mu_list.append(np.squeeze(mu))
            temp_var_list.append(np.squeeze(1/var))
        prior_var = (1-len(temp_agents_list))/(self.sigmaF**2+ self.sigmaN**2)
        var_BCM = np.sum(temp_var_list)+ prior_var.ravel()
        weights = temp_var_list/var_BCM
        mu_BCM = np.dot(temp_mu_list, weights)
        return mu_BCM
    
    # rBCM
    def predict_rBCM_initial(self, i_agent, x):
        temp_agents_list = self.neighbors_list[i_agent]
        temp_mu_list = []
        temp_var_list = []
        beta_k_list = []
        for i in range(len(temp_agents_list)):
            act_agent = temp_agents_list[i]
            mu, var = self.agents[act_agent].predict_initial(x)
            beta_k = math.log((self.sigmaF**2 + self.sigmaN**2)/var) 
            temp_mu_list.append(np.squeeze(mu))
            temp_var_list.append(np.squeeze(beta_k/var))
            beta_k_list.append(np.squeeze(beta_k))
        bcm_var = (1-np.sum(beta_k_list))/(self.sigmaF**2+ self.sigmaN**2)
        var_RBCM = np.sum(temp_var_list )+ bcm_var.ravel()
        weights = temp_var_list/var_RBCM
        mu_RBCM = np.dot(temp_mu_list, weights)
        return mu_RBCM
    
    def predict_rBCM(self, i_agent, x):
        temp_agents_list = self.neighbors_list[i_agent]
        temp_mu_list = []
        temp_var_list = []
        beta_k_list = []
        for i in range(len(temp_agents_list)):
            act_agent = temp_agents_list[i]
            mu, var = self.agents[act_agent].predict(x)
            beta_k = math.log((self.sigmaF**2 + self.sigmaN**2)/var) 
            temp_mu_list.append(np.squeeze(mu))
            temp_var_list.append(np.squeeze(beta_k/var))
            beta_k_list.append(np.squeeze(beta_k))
        bcm_var = (1-np.sum(beta_k_list))/(self.sigmaF**2+ self.sigmaN**2)
        var_RBCM = np.sum(temp_var_list )+ bcm_var.ravel()
        weights = temp_var_list/var_RBCM
        mu_RBCM = np.dot(temp_mu_list, weights)
        return mu_RBCM