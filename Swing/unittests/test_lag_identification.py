import unittest
import Swing.util.lag_identification as lag_id
from Swing.util.Evaluator import Evaluator
import numpy as np
from random import randint
import numpy.testing as npt
import random
import pdb
import itertools
import pandas as pd
import pickle
import h5py
from scipy.stats import pearsonr

import networkx as nx

"""this test generally checks to see if aupr calculations are robust and if trivial cases evaluate as expected"""

class TestLagIdentification(unittest.TestCase):

    def setUp(self):
        self.experiments = lag_id.get_experiment_list("../../data/invitro/cantone_switchon_interpolated_timeseries.tsv", timepoints=18, perturbs=5)

        gold_standard_file = "../../data/invitro/cantone_switchon_interpolated_goldstandard.tsv"

        self.evaluator = Evaluator(gold_standard_file, sep = '\t')
        self.genes = list(self.experiments[0].columns.values)
        

        self.experiments2 = lag_id.get_experiment_list('../../data/invitro/omranian_parsed_timeseries.tsv', 5,26)
        self.genes2 = list(self.experiments2[0].columns.values)

        self.signed_edge_list = pd.read_csv('../../data/invitro/omranian_signed_parsed_goldstandard.tsv',sep='\t',header=None)
        self.signed_edge_list.columns=['regulator', 'target', 'signs']
        self.signed_edge_list['regulator-target'] = tuple(zip(self.signed_edge_list['regulator'], self.signed_edge_list['target']))

    def get_xcorr_indices(self,diff_ts, lag, tolerance):
        pair_list = []
        # get all pairs
        targets = np.array(np.where((diff_ts >= lag-tolerance ) & (diff_ts <= lag+tolerance)))
        n_ind = targets.shape[1]
        pair_list = [tuple(targets[:,x]) for x in range(n_ind)]
        # only keep tuples where the parent index is greater than the child
        if lag != 0:
            pair_list = [ x for x in pair_list if x[1] < x[2]]
        p_pair_list = [(x[0],x[1]) for x in pair_list]
        c_pair_list = [(x[0],x[2]) for x in pair_list]

        return(p_pair_list,c_pair_list)

        
    def get_pairwise_xcorr(self,parent,child,experiment,time_map,lag,tolerance,rc = (26,5)):         
        ts_shape = time_map.shape[1]-1
        ts = time_map.iloc[:,:ts_shape]
        ts = ts.values
            
        all_ps_values = np.zeros(rc)
        all_cs_values = np.zeros(rc)

        # make an array of differences

        diff_ts = np.abs(ts[:,:,None] - ts[:,None,:])
        # get all indices with the same difference
        ps_values = np.zeros(rc)
        cs_values = np.zeros(rc)
        ps = [x[parent].values for x in experiment]
        cs = [x[child].values for x in experiment]
        all_ps_values = np.vstack(ps)
        all_cs_values = np.vstack(cs)

        p_idx,c_idx = self.get_xcorr_indices(diff_ts, lag, tolerance)
        
        ps_values = [all_ps_values[x] for x in p_idx]
        cs_values = [all_cs_values[x] for x in c_idx]
        
        rsq, pval = pearsonr(ps_values,cs_values)
        
        return(rsq,pval)
        
    def test_calculate_edge_lag(self):
        flat = False
        timestep = 1
        h5f = h5py.File('/projects/p20519/roller_output/xcorr_omranian.h5','r')
        xcorr = h5f['dataset1'][:]
        h5f.close()
        # experiments, parent, child, time
        e, p, c, t = xcorr.shape

        
        # load the interval file
        edges = self.signed_edge_list['regulator-target']
        lag_estimate = np.zeros((p,c))

        #initialize dataframe to return
        col, row = np.meshgrid(range(len(self.genes2)), range(len(self.genes2)))
        edge_lag = pd.DataFrame()
        edge_lag['Parent'] = np.array(self.genes2)[row.flatten()]
        edge_lag['Child'] = np.array(self.genes2)[col.flatten()]
        edge_lag['Edge'] = list(zip(edge_lag['Parent'], edge_lag['Child']))

        lag_results = []
        time_map = pd.read_csv('../../data/invitro/omranian_timesteps.tsv', sep='\t')
        time_steps = time_map['Timestep'].tolist()
        
        G=nx.Graph()
        G.add_edges_from(edges)

        fp_edges = edge_lag['Edge'].iloc[:1000000].tolist()
        fp_edges = [x for x in fp_edges if x not in edges]
        parsed_fp_edges = []
        for fp_edge in fp_edges:
            if (fp_edge[0] in G.nodes()) and (fp_edge[1] in G.nodes()):
                if nx.has_path(G,fp_edge[0],fp_edge[1]):
                    continue
                else:
                    #pl = nx.shortest_path_length(G,fp_edge[0], fp_edge[1])
                    parsed_fp_edges.append(fp_edge)

        pdb.set_trace()
        #parsed_fp_edges = parsed_fp_edges[:2870]


        for edge in parsed_fp_edges:
            # Ignore self edges
            if edge[0] == edge[1]:
                continue
    
            lags = [0,10,20,30,60,90]
            tolerance = 8
            c_list = []
            for lag in lags:
                r,p = self.get_pairwise_xcorr(edge[0],edge[1],self.experiments2,time_map,lag,8,(26,5))
                c_list.append((lag,r,p))

            sign = self.signed_edge_list[self.signed_edge_list['regulator-target'] == edge]['signs'].tolist()
        
            best_lag = min(c_list, key = lambda x: x[2])
            if best_lag[2] > 0.05/len(parsed_fp_edges):
                true_lag = np.nan
            else:
                true_lag = best_lag[0]

            lag_results.append({'Edge':edge, 'Lag':true_lag, 'Sign': sign, 'Lag_list': c_list})
        
        lag_results = pd.DataFrame(lag_results)
        pdb.set_trace()
        edge_lag = pd.merge(edge_lag, lag_results, how='outer', on='Edge')
        pdb.set_trace()

"""
    def test_calculate_edge_lag2(self):

        flat = False
        timestep = 1
        h5f = h5py.File('/projects/p20519/roller_output/xcorr_omranian.h5','r')
        xcorr = h5f['dataset1'][:]
        h5f.close()
        # experiments, parent, child, time
        e, p, c, t = xcorr.shape

        
        # load the interval file
        edges = self.signed_edge_list['regulator-target']
        lag_estimate = np.zeros((p,c))

        #initialize dataframe to return
        col, row = np.meshgrid(range(len(self.genes2)), range(len(self.genes2)))
        edge_lag = pd.DataFrame()
        edge_lag['Parent'] = np.array(self.genes2)[row.flatten()]
        edge_lag['Child'] = np.array(self.genes2)[col.flatten()]
        edge_lag['Edge'] = list(zip(edge_lag['Parent'], edge_lag['Child']))

        lag_results = []
        time_map = pd.read_csv('../../data/invitro/omranian_timesteps.tsv', sep='\t')
        time_steps = time_map['Timestep'].tolist()

        for edge in edges:
            # Ignore self edges
            if edge[0] == edge[1]:
                continue
            p_idx = self.genes2.index(edge[0])
            c_idx = self.genes2.index(edge[1])
            
            parent_values = [(lambda x: x[edge[0]].values) (x) for x in self.experiments2]
            parent_values = np.array(parent_values).flatten()
            child_values = [(lambda x: x[edge[1]].values) (x) for x in self.experiments2]
            child_values = np.array(child_values).flatten()
             
            all_ps_values = np.zeros((26,5))
            all_cs_values = np.zeros((26,5))
            pdb.set_trace()
            for idx,(time,exp) in enumerate(zip(time_steps, self.experiments2)):
                    # get ps_values, get cs_values
                    all_ps_values[idx] = exp[edge[0]].values
                    all_cs_values[idx] = exp[edge[1]].values

            pdb.set_trace()


            # get rid of rows with all 0s
            all_ps_values = all_ps_values[~np.all(all_ps_values == 0, axis=1)]
            all_cs_values = all_cs_values[~np.all(all_cs_values == 0, axis=1)]
            all_ps_values[np.isnan(all_cs_values)] = np.nan
            all_ps_values = all_ps_values.flatten()
            all_cs_values = all_cs_values.flatten()
            all_ps_values = all_ps_values[~np.isnan(all_ps_values)]
            all_cs_values = all_cs_values[~np.isnan(all_cs_values)]
            lagged = pearsonr(all_ps_values,all_cs_values)

            # remove nan
            
            if self.signed_edge_list is not None:
                sign = self.signed_edge_list[self.signed_edge_list['regulator-target'] == edge]['signs'].tolist()[0]

            reverse = xcorr[:, c_idx, p_idx]
            ar = time_map[[0,1,2,3,4]].values
            aggr_cor = []
            for lg in np.unique(ar):      
                (x,y) = np.where(ar == lg)
                coords = list(zip(x,y))
                cor = [reverse[coord] for coord in coords]
                aggr_cor.append((lg,np.mean(cor)))
            
            filtered = reverse
            #filtered = filter_ccfs(reverse, sc_thresh, min_ccf)
            if filtered.shape[0] > 0:
                # f, axarr = plt.subplots(1,2)
                # axarr[0].plot(reverse.T)
                # axarr[1].plot(filtered.T)
                # plt.show()

                # default setting
                if flat:
                    if self.signed_edge_list is None:
                        lag_estimate[p_idx, c_idx] = float(np.mean(np.argmax(np.abs(filtered), axis=1)))*timestep
                    elif sign == '+':
                        lag_estimate[p_idx, c_idx] = float(np.mean(np.argmax(filtered, axis=1)))*timestep
                    elif sign == '-':
                        lag_estimate[p_idx, c_idx] = float(np.mean(np.argmin(filtered, axis=1)))*timestep
                    elif sign == '+-':
                        lag_estimate[p_idx, c_idx] = float(np.mean(np.argmax(np.abs(filtered), axis=1)))*timestep
                    edge_lag['Lag'] = lag_estimate.flatten()

                elif not flat:

                    if sign == '+':
                        lag = [float(x) for x in np.argmax(filtered, axis=1)]*timestep
                    elif sign == '-':
                        lag = [float(x) for x in np.argmin(filtered, axis=1)]*timestep
                    elif sign == '+-':
                        lag = [float(x) for x in np.argmax(np.abs(filtered), axis=1)]*timestep

                    corr, sig = pearsonr(parent_values,child_values)
                    lcorr, lsig = lagged
                    lag_results.append({'Edge':edge, 'Lag':lag, 'Raw_XC': reverse, 'sign': sign, 'corr': aggr_cor, 't0_corr': corr, 't0_sig': sig, 't1_corr':lcorr, 't1_sig':lsig})

        if not flat:
            lag_results = pd.DataFrame(lag_results)
            edge_lag = pd.merge(edge_lag, lag_results, how='outer', on='Edge')
        
        from operator import itemgetter
        positive = lag_results[(lag_results['sign']=="+")]
        negative = lag_results[(lag_results['sign']=="-")]
        posneg = lag_results[(lag_results['sign'] == "+-")]

        positive['max_corr'] = [max(x,key=itemgetter(1)) for x in positive['corr'].values.tolist()]
        negative['max_corr'] = [min(x,key=itemgetter(1)) for x in negative['corr'].values.tolist()]
        posneg['max_corr'] = [max(np.abs(x),key=itemgetter(1)) for x in posneg['corr'].values.tolist()]
        positive['filtered_XC'] = [(lambda x: x[(np.max(x,axis=1) >= 0.6)])(x) for x in positive['Raw_XC'].tolist()]
        t_map = time_map['Timestep'].values 
        positive['filtered_timestep'] = [(lambda x: t_map[(np.max(x,axis=1) >= 0.6)])(x) for x in positive['Raw_XC'].tolist()]
        positive['lag_estimate'] = [(lambda x: np.argmax(x, axis=1))(x) for x in positive['filtered_XC'].tolist()]*positive['filtered_timestep']
        positive['mean_lag'] = [(lambda x: x.mean())(x) for x in positive['lag_estimate'].tolist()]
        positive['filtered_counts'] = [len(x) for x in positive['filtered_timestep'].tolist()]
        
        negative['filtered_XC'] = [(lambda x: x[(np.max(x,axis=1) >= 0.6)])(x) for x in negative['Raw_XC'].tolist()]
        
        negative['filtered_timestep'] = [(lambda x: t_map[(np.max(x,axis=1) >= 0.6)])(x) for x in negative['Raw_XC'].tolist()]
        negative['lag_estimate'] = [(lambda x: np.argmax(x, axis=1))(x) for x in negative['filtered_XC'].tolist()]*negative['filtered_timestep']
        negative['mean_lag'] = [(lambda x: x.mean())(x) for x in negative['lag_estimate'].tolist()]
        negative['filtered_counts'] = [len(x) for x in negative['filtered_timestep'].tolist()]

        negative['filtered'] = [(np.abs(x) >= 0.6) for x in negative['t0_corr'].tolist()]
                # print(edge, np.argmax(filtered, axis=0), np.mean(np.argmax(filtered, axis=0)))
        
        posneg['filtered_XC'] = [(lambda x: x[(np.max(x,axis=1) >= 0.6)])(x) for x in posneg['Raw_XC'].tolist()]
        
        posneg['filtered_timestep'] = [(lambda x: t_map[(np.max(x,axis=1) >= 0.6)])(x) for x in posneg['Raw_XC'].tolist()]
        posneg['lag_estimate'] = [(lambda x: np.argmax(x, axis=1))(x) for x in posneg['filtered_XC'].tolist()]*posneg['filtered_timestep']
        posneg['mean_lag'] = [(lambda x: x.mean())(x) for x in posneg['lag_estimate'].tolist()]
        posneg['filtered_counts'] = [len(x) for x in posneg['filtered_timestep'].tolist()]

        posneg['filtered'] = [(np.abs(x) >= 0.6) for x in posneg['t0_corr'].tolist()]
        pdb.set_trace()
        
        edges = set(edges.tolist())
        all_edges = set(itertools.product(self.genes2, self.genes2))
        not_edges = list(all_edges-edges)
        not_edges = not_edges[:10000]
        not_lag_results = []
        print("done!")
        for edge in not_edges:
            # Ignore self edges
            if edge[0] == edge[1]:
                continue
            p_idx = self.genes2.index(edge[0])
            c_idx = self.genes2.index(edge[1])
            
            parent_values = [(lambda x: x[edge[0]].values) (x) for x in self.experiments2]
            parent_values = np.array(parent_values).flatten()
            child_values = [(lambda x: x[edge[1]].values) (x) for x in self.experiments2]
            child_values = np.array(child_values).flatten()
            
            all_ps_values = np.zeros((26,5))
            all_cs_values = np.zeros((26,5))
            for idx,(time,exp) in enumerate(zip(time_steps, self.experiments2)):
                if time < 20:
                    # get ps_values, get cs_values
                    all_ps_values[idx] = exp[edge[0]].values
                    all_cs_values[idx] = exp[edge[1]].shift(1).values

            # get rid of rows with all 0s
            all_ps_values = all_ps_values[~np.all(all_ps_values == 0, axis=1)]
            all_cs_values = all_cs_values[~np.all(all_cs_values == 0, axis=1)]
            all_ps_values[np.isnan(all_cs_values)] = np.nan
            all_ps_values = all_ps_values.flatten()
            all_cs_values = all_cs_values.flatten()
            all_ps_values = all_ps_values[~np.isnan(all_ps_values)]
            all_cs_values = all_cs_values[~np.isnan(all_cs_values)]
            lagged = pearsonr(all_ps_values,all_cs_values)

            reverse = xcorr[:, c_idx, p_idx]
            ar = time_map[[0,1,2,3,4]].values
            aggr_cor = []
            for lg in np.unique(ar):      
                (x,y) = np.where(ar == lg)
                coords = list(zip(x,y))
                cor = [reverse[coord] for coord in coords]
                aggr_cor.append((lg,np.mean(cor)))
            
            filtered = reverse
            #filtered = filter_ccfs(reverse, sc_thresh, min_ccf)
            if filtered.shape[0] > 0:
                # f, axarr = plt.subplots(1,2)
                # axarr[0].plot(reverse.T)
                # axarr[1].plot(filtered.T)
                # plt.show()

                # default setting
                if flat:
                    if self.signed_edge_list is None:
                        lag_estimate[p_idx, c_idx] = float(np.mean(np.argmax(np.abs(filtered), axis=1)))*timestep
                    elif sign == '+':
                        lag_estimate[p_idx, c_idx] = float(np.mean(np.argmax(filtered, axis=1)))*timestep
                    elif sign == '-':
                        lag_estimate[p_idx, c_idx] = float(np.mean(np.argmin(filtered, axis=1)))*timestep
                    elif sign == '+-':
                        lag_estimate[p_idx, c_idx] = float(np.mean(np.argmax(np.abs(filtered), axis=1)))*timestep
                    edge_lag['Lag'] = lag_estimate.flatten()

                elif not flat:
                    sign = '+-'
                    if sign == '+':
                        lag = [float(x) for x in np.argmax(filtered, axis=1)]*timestep
                    elif sign == '-':
                        lag = [float(x) for x in np.argmin(filtered, axis=1)]*timestep
                    elif sign == '+-':
                        lag = [float(x) for x in np.argmax(np.abs(filtered), axis=1)]*timestep
                    not_lag_results.append({'Edge':edge, 'Lag':lag, 'Raw_XC': reverse, 'sign': sign, 'corr': aggr_cor,'raw_corr': pearsonr(parent_values, child_values), 'lagged_corr': lagged})

        if not flat:
            not_lag_results = pd.DataFrame(not_lag_results)
            edge_lag = pd.merge(edge_lag, not_lag_results, how='outer', on='Edge')
        
        posneg = not_lag_results[(not_lag_results['sign'] == "+-")]

        posneg['filtered_XC'] = [(lambda x: x[(np.max(x,axis=1) >= 0.9)])(x) for x in posneg['Raw_XC'].tolist()]
        t_map = time_map['Timestep'].values 
        posneg['filtered_timestep'] = [(lambda x: t_map[(np.max(x,axis=1) >= 0.9)])(x) for x in posneg['Raw_XC'].tolist()]
        posneg['lag_estimate'] = [(lambda x: np.argmax(x, axis=1))(x) for x in posneg['filtered_XC'].tolist()]*posneg['filtered_timestep']
        posneg['mean_lag'] = [(lambda x: x.mean())(x) for x in posneg['lag_estimate'].tolist()]
        posneg['filtered_counts'] = [len(x) for x in posneg['filtered_timestep'].tolist()]
        posneg['filtered'] = [(np.abs(x[0]) > 0.5) for x in posneg['raw_corr'].tolist()]
        negcorr = posneg['corr'].tolist()
        poscorr = positive['corr'].tolist()
        
        for thres in [0.1, 0.05, 0.01,0.001,0.0001,0.00001,0.000001]:
        
            posneg['t0'] = [x[1] < thres for x in posneg['raw_corr'].tolist()]
            fp0 = len(posneg[posneg['t0']])/len(posneg)
            positive['t0'] = [x[1] < thres for x in positive['raw_corr'].tolist()]
            tp0 = len(positive[positive['t0']])/len(positive)
        

            posneg['t'] = [x[1] < thres for x in posneg['lagged_corr'].tolist()]
            fp = len(posneg[posneg['t']])/len(posneg)
            positive['t'] = [x[1] < thres for x in positive['lagged_corr'].tolist()]
            tp = len(positive[positive['t']])/len(positive)
            print(tp0,fp0, tp0-fp0, tp,fp, tp-fp)
        pdb.set_trace()
    def test_calculate_edge_lag(self):
        exp_xcor = lag_id.xcorr_experiments(self.experiments, 1)
        edges = itertools.product(self.genes,self.genes)
        signed_edge_list = []
        for edge in edges:
            signed_edge_list.append(edge)
        signed_edge_df = pd.DataFrame({'regulator-target':signed_edge_list})
        signs = ['+' if bool(x<13) else '-' for x in range(25)]
        signed_edge_df['signs'] = signs


        filtered_lags=lag_id.calc_edge_lag(exp_xcor, self.genes, 0.1, 0.8, timestep=1, signed_edge_list = signed_edge_df, flat=False)
        print(filtered_lags)
"""
if __name__ == '__main__':
    unittest.main()
