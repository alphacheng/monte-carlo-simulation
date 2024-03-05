# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:10:14 2024

@author: 2182
"""

import numpy as np


# FEMA P-58-1 G: Generation of Simulated Demands G-13


num_realization=10000

# case 1
# full rank
EDPs= np.array([[1.26,1.45,1.71,0.54,0.87,0.88,0.65],
            [1.41,2.05,2.43,0.55,0.87,0.77,0.78],
            [1.37,1.96,2.63,0.75,1.04,0.89,0.81],
            [0.97,1.87,2.74,0.55,0.92,1.12,0.75],
            [0.94,1.80,2.02,0.40,0.77,0.74,0.64],
            [1.73,2.55,2.46,0.45,0.57,0.45,0.59],
            [1.05,2.15,2.26,0.38,0.59,0.49,0.52],
            [1.40,1.67,2.10,0.73,1.50,1.34,0.83],
            [1.59,1.76,2.01,0.59,0.94,0.81,0.72],
            [0.83,1.68,2.25,0.53,1.00,0.90,0.74],
            [0.96,1.83,2.25,0.49,0.90,0.81,0.64]])
lnEDPs=np.log(EDPs)

# case 2
# non full rank
# lnEDPs= np.array([[0.231112,0.371564,0.536493,-0.616186,-0.139262,-0.127833,-0.430783],
# [0.34359,0.71784,0.887891,-0.597837,-0.139262,-0.261365,-0.248461],
# [0.314811,0.672944,0.966984,-0.287682,0.0392207,-0.116534,-0.210721],
# [-0.0304592,0.625938,1.00796,-0.597837,-0.0833816,0.113329,-0.287682],
# [-0.0618754,0.587787,0.703098,-0.916291,-0.261365,-0.301105,-0.446287]])


num_rec,num_var=lnEDPs.shape

lnEDPs_mean=np.average(lnEDPs,axis=0)
lnEDPs_mean=lnEDPs_mean.reshape(lnEDPs_mean.size,1)

lnEDPs_cov=np.cov(lnEDPs.T)
lnEDPs_cov_rank= np.linalg.matrix_rank(lnEDPs_cov)

sigma=np.sqrt(np.diag(lnEDPs_cov))
sigmap2=sigma*sigma

R=lnEDPs_cov/(sigma*sigma.T)

D,L=np.linalg.eigh(lnEDPs_cov)

if lnEDPs_cov_rank>=num_var:
    L_use=L
    D2_use=D
    U = np.random.randn(num_realization, num_var)
else:
    L_use=L[:,num_var-lnEDPs_cov_rank:num_var]
    D2_use=D[num_var-lnEDPs_cov_rank:num_var]
    U = np.random.randn(num_realization, lnEDPs_cov_rank)
    
U=U.T
D_use=np.diag(D2_use**0.5)

Lambda=L_use.dot(D_use)
Z=Lambda.dot(U)+lnEDPs_mean.dot(np.ones((1,num_realization)))

lnEDPs_sim_mean=np.average(Z,axis=1)
lnEDPs_sim_mean=lnEDPs_sim_mean.reshape(lnEDPs_sim_mean.size,1)

lnEDPs_sim_cov=np.cov(Z)

A=lnEDPs_sim_mean/lnEDPs_mean
B=lnEDPs_sim_cov/lnEDPs_cov

W=np.exp(Z)


#matlab version
'''
Example Matlab Code:
% Develop underlying statistics of the response history analysis
clear;
num_realization = 10000;
% loading information: EDP, epistemic variability, 
EDPs =load('EDP.txt');
B=load(‘beta.txt’);%matrix for dispersions β m and β gm , 
% taking natural logarithm of the EDPs. Calling it lnEDPs
lnEDPs=log(EDPs); % Table G-2, or Table G-9 
[num_rec num_var]=size(lnEDPs);
% finding the mean matrix of lnEDPs. Calling it lnEDPs_mean
lnEDPs_mean=mean(lnEDPs); % last row in Table G-2, or Table G-9 
lnEDPs_mean=lnEDPs_mean';
 
% finding the covariance matrix of lnEDPs. Calling it lnEDPs_cov
lnEDPs_cov=cov(lnEDPs); % Table G-3, or Table G-10
 
% finding the rank of covariance matrix of lnEDPs. Calling it 
% lnEDPs_cov_rank
lnEDPs_cov_rank=rank(lnEDPs_cov);
% inflating the variances with epistemic variability 
sigma = sqrt(diag(lnEDPs_cov)); % sqrt first to avoid under/overflow
sigmap2 = sigma.*sigma;
R = lnEDPs_cov ./ (sigma*sigma');
B=B’;
sigmap2=sigmap2+(B(:,1).* B(:,1)); % Inflating variance for β m
sigmap2=sigmap2+(B(:,2).* B(:,2)); % Inflating variance for β gm
sigma=sqrt(sigmap2);
sigma2=sigma*sigma';
lnEDPs_cov_inflated=R.*sigma2;
% finding the eigenvalues eigenvectors of the covariance matrix. 
Calling
% them D2_total and L_total
[L_total D2_total]=eig(lnEDPs_cov_inflated); % Table G-5, Table G-13
 D2_total=eig(lnEDPs_cov_inflated); 
 
% Partition L_total to L_use. L_use is the part of eigenvector matrix
% L_total that corresponds to positive eigenvalues
 if lnEDPs_cov_rank >= num_var
 L_use =L_total; % Table G-5 
else
 L_use =(L_total(:,num_var- lnEDPs_cov_rank+1:num_var));
%Table G-13
 end
% Partition the D2_total to D2_use. D2_use is the part of eigenvalue 
%vector D2_total that corresponds to positive eigenvalues
if lnEDPs_cov_rank >= num_var
 D2_use =D2_total; 
else
 D2_use =D2_total(num_var- lnEDPs_cov_rank+1:num_var); 
end
 
% Find the square root of D2_use and call is D_use. 
 D_use =diag((D2_use).^0.5); %Table G-4, or Table G-12
 
% Generate Standard random numbers
if lnEDPs_cov_rank >= num_var
 U = randn(num_realization,num_var) ;
else
 U = randn(num_realization, lnEDPs_cov_rank) ;
end
U = U' ;
% Create Lambda = D_use . L_use
Lambda = L_use * D_use ;
 
% Create realizations matrix 
Z = Lambda * U + lnEDPs_mean * ones(1,num_realization) ;
 
lnEDPs_sim_mean=mean(Z');
lnEDPs_sim_cov=cov(Z');
 
A=lnEDPs_sim_mean./lnEDPs_mean'; %Table G-7, or Table G-16
B=lnEDPs_sim_cov./lnEDPs_cov; %Table G-8, or Table G-17
 
 W=exp(Z);
%end
'''

