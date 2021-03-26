import sys, os
import time
import pickle
import sqlite3
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import scipy.sparse as sp

import ltr_tensor_solver_actxu_v_cls_010 as tensor_cls
## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
## ncreg parameter is included for the regularization constant 
def main(CV_indices_list, n_order, n_rank, n_repeat, lossdegree=0, \
                     n_sigma=0.01, ncreg=0.0000001):

    time00 = time.time()
    nfold = 5  ## number of folds in the cross validation

    ## (((((((((((((((((((((((((((((((((((((((((((
    ## Parameters to ltr learner
    # n_d=5    ## maximum power(order)
    n_d = n_order
    order = n_d
    # rank=20    ## number of ranks, n_t in the paper
    rank = n_rank
    ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    ## projection dimension to reduce the total size of the parameters
    rankuv=20        ## projection dimension
    ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # sigma=0.002     ## learning step size
    sigma = n_sigma
    learning_rate = sigma
    ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    sigmadscale=4    ## convergence control
    ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    nsigma = 1  ## speed correction interval
    gamma = 0.999999  ## discount factor of the  parameter update
    gammanag = 0.95  ## discount for the ADAM gradient update
    gammanag2 = 0.95  ## discount for the ADAM norm update

    mblock = 1000  ## minibatch size, number of examples
    mblock_gap = mblock  ## shift of blocks
    batch_size = mblock

    # nrepeat = 10         ## number of epochs, repetition of the full online run
    nrepeat = n_repeat
    n_epochs = nrepeat

    ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    cregular = ncreg  ## regularization penaly constant for P and Q as well
    ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    ikolmogorov = 0
    kolm_t = 1.5

    ## nrankclass=1   ## number of steps in multilayer algorithm
    nsigmamax = 1  ## gradient scaling to avoid too long gradient, default=1

    ihomogen = 1  ## homegeous data vectors, default=1

    ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    ## rankuv parameter for projection dimension is included
    cmodel = tensor_cls.tensor_latent_vector_cls(norder=n_d, rank=rank, rankuv=rankuv)
    ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    ## dscale is included
    ## set optimization parameters
    cmodel.update_parameters(nsigma=nsigma, \
                             mblock=mblock, \
                             mblock_gap=mblock_gap, \
                             sigma0=sigma, \
                             dscale=sigmadscale, \
                             gamma=gamma, \
                             gammanag=gammanag, \
                             gammanag2=gammanag2, \
                             cregular=cregular, \
                             sigmamax=nsigmamax, \
                             ikolmogorov=ikolmogorov, \
                             kolm_t=kolm_t)

    ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    ## nrankuv and dscale are printed
    print('n-t:', cmodel.norder)
    print('Rank:', cmodel.nrank)
    print('Rankuv:',cmodel.nrankuv)
    print('Step size:', cmodel.sigma0)
    print('Step freq:', cmodel.nsigma)
    print('Epoch:', nrepeat)
    print('Block size:', mblock)
    print('Discount:', cmodel.gamma)
    print('Discount for NAG:', cmodel.gammanag)
    print('Discount for NAG norm:', cmodel.gammanag2)
    print('Bag size:', cmodel.mblock)
    print('Bag step:', cmodel.mblock_gap)
    print('Regularization:', cmodel.cregular)
    print('Regularization degree:',cmodel.regdegree)
    print('Gradient max ratio:', cmodel.sigmamax)
    print('Kolmogorov mean:', cmodel.ikolmogorov)
    print('Kolmogorov mean param:', cmodel.kolm_t)
    ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    #####################################
    ## loss function type slection
    ## activation function
    cmodel.iactfunc = 0  ## =0 identity, =1 arcsinh =2 2*sigmoid-1 =3 tanh
    ## loss degree
    # cmodel.lossdegree = 0  ## =0 L_2^2, =1 L^2, =0.5 L_2^{0.5}, ...L_2^{z}
    cmodel.lossdegree = lossdegree

    print('Activation:', cmodel.iactfunc)
    print('Loss degree:', cmodel.lossdegree)

    ##)))))))))))))))))))))))))))))))))))))))))))))

    ## prediction collector for all folds
    yprediction = np.zeros(m)

    xcorrpear = np.zeros(nfold)
    xcorrspear = np.zeros(nfold)
    xrmse = np.zeros(nfold)


    ifold = 0
    for itrain, itest in CV_indices_list:

        Xtrain = X[itrain]
        ytrain = y[itrain]
        Xtest = X[itest]
        ytest = y[itest]
        print('Training:', len(Xtrain))
        print('Test:', len(Xtest))
        mtrain = len(itrain)
        mtest = len(itest)
        time0 = time.time()
        ## training
        cmodel.fit(Xtrain, ytrain, nepoch=nrepeat)
        time1 = time.time()
        print('Training time:', time1 - time0)
        with open(outdir + 'Weights_%s_%s.Pickle'%(job_id, ifold), 'wb') as f:
            pickle.dump(cmodel.xlambdaU, f)
        ## Perform predictions
        ypred_test = cmodel.predict(Xtest)
        ypred_test = ypred_test.ravel()
        yprediction[itest] = ypred_test
        # Measure performance
        RMSE_test = np.sqrt(mean_squared_error(ytest*ymax+ymean, ypred_test*ymax+ymean))
        Rpears_test = np.corrcoef(ytest, ypred_test)[0, 1]
        Rspear_test, _ = spearmanr(ytest, ypred_test)
        xcorrpear[ifold] = Rpears_test
        xcorrspear[ifold] = Rspear_test
        xrmse[ifold] = RMSE_test
        np.savetxt((outdir + "ytestst_scaled_" \
                    + "order-" \
                    + str(order) + "_rank-" + str(rank) + "_learnR-" \
                    + str(learning_rate) + "_nepochs-" + str(n_epochs) \
                    + str(nrepeat) + "_nrepeat" + str(cmodel.lossdegree) + "_lossdegree" \
                    + "_batchSize-" + str(batch_size) \
                    + "_fold_%s" % ifold + "_CV_%s" % job_id + ".txt"), \
                   ytest)
        np.savetxt((outdir + "ypred_scaled_" \
                    + "order-" \
                    + str(order) + "_rank-" + str(rank) + "_learnR-" \
                    + str(learning_rate) + "_nepochs-" + str(n_epochs)
                    + str(nrepeat) + "_nrepeat" + str(cmodel.lossdegree) + "_lossdegree" \
                    + "_batchSize-" + str(batch_size) \
                    + "_fold_%s" % ifold + "_CV_%s" % job_id + ".txt"), \
                   ypred_test)
        np.savetxt((outdir + "ytest_" \
                    + "order-" \
                    + str(order) + "_rank-" + str(rank) + "_learnR-" \
                    + str(learning_rate) + "_nepochs-" + str(n_epochs) \
                    + str(nrepeat) + "_nrepeat" + str(cmodel.lossdegree) + "_lossdegree" \
                    + "_batchSize-" + str(batch_size) \
                    + "_fold_%s" % ifold + "_CV_%s" % job_id + ".txt"), \
                   ytest*ymax+ymean)
        np.savetxt((outdir + "ypred_" \
                    + "order-" \
                    + str(order) + "_rank-" + str(rank) + "_learnR-" \
                    + str(learning_rate) + "_nepochs-" + str(n_epochs)
                    + str(nrepeat) + "_nrepeat" + str(cmodel.lossdegree) + "_lossdegree" \
                    + "_batchSize-" + str(batch_size) \
                    + "_fold_%s" % ifold + "_CV_%s" % job_id + ".txt"), \
                   ypred_test*ymax+ymean)
        # Write results into a file
        f = open(outdir + "results" + "_fold_" + str(ifold) + "_order_" + str(order) + "_rank_" + str(rank) \
                 + "_repeat_" + str(nrepeat) + "_lossdegree_" + str(cmodel.lossdegree) + "_CV_%s" % job_id + ".txt",
                 'w')
        f.write("rank = %d\n" % (rank))
        f.write("n_epochs = %f\n" % (n_epochs))
        f.write("n_repeat = %f\n" % (nrepeat))
        f.write("lossdegree = %f\n" % (cmodel.lossdegree))
        f.write("TEST:\n")
        f.write("RMSE = %f\n" % (RMSE_test))
        f.write("R = %f\n" % (Rpears_test))
        f.write("R_spear = %f\n" % (Rspear_test))
        f.close()
        # Write results on the screen
        print('fold:', ifold)
        print("rank = %d" % (rank))
        print("n_epochs = %f" % (n_epochs))
        print("TEST:")
        print("RMSE = %f" % (RMSE_test))
        print("R = %f" % (Rpears_test))
        print("R_spear = %f" % (Rspear_test))
        # Save parameters
        cmodel.save_parameters(outdir + 'Params_%s_%s.Pickle'%(job_id, ifold))
        ifold += 1
        if ifold == nfold:
            break

    time2 = time.time()
    print('Total training time(s): ', time2 - time00)
    pcorr = np.corrcoef(y, yprediction)[0, 1]
    rmse = np.sqrt(np.mean((y*ymax+ymean - yprediction*ymax-ymean) ** 2))
    scorr, _ = spearmanr(y, yprediction)
    np.savetxt((outdir + "y_full_" \
                + "order-" \
                + str(order) + "_rank-" + str(rank) + "_learnR-" \
                + str(learning_rate) + "_nepochs-" + str(n_epochs) \
                + str(nrepeat) + "_nrepeat" + str(cmodel.lossdegree) + "_lossdegree" \
                + "_batchSize-" + str(batch_size) + "_CV_%s" % job_id \
                + ".txt"), \
               y*ymax+ymean)
    np.savetxt((outdir + "ypred_full_" \
                + "order-" \
                + str(order) + "_rank-" + str(rank) + "_learnR-" \
                + str(learning_rate) + "_nepochs-" + str(n_epochs)
                + str(nrepeat) + "_nrepeat" + str(cmodel.lossdegree) + "_lossdegree" \
                + "_batchSize-" + str(batch_size) + "_CV_%s" % job_id \
                + ".txt"), \
               yprediction*ymax+ymean)
    np.savetxt((outdir + "y_full_scaled_" \
                + "order-" \
                + str(order) + "_rank-" + str(rank) + "_learnR-" \
                + str(learning_rate) + "_nepochs-" + str(n_epochs) \
                + str(nrepeat) + "_nrepeat" + str(cmodel.lossdegree) + "_lossdegree" \
                + "_batchSize-" + str(batch_size) + "_CV_%s" % job_id \
                + ".txt"), \
               y)
    np.savetxt((outdir + "ypred_full_scaled_" \
                + "order-" \
                + str(order) + "_rank-" + str(rank) + "_learnR-" \
                + str(learning_rate) + "_nepochs-" + str(n_epochs)
                + str(nrepeat) + "_nrepeat" + str(cmodel.lossdegree) + "_lossdegree" \
                + "_batchSize-" + str(batch_size) + "_CV_%s" % job_id \
                + ".txt"), \
               yprediction)
    #cmodel.save_parameters('Weights_%s.Pickle'%job_id)

    print('Full result ---------------')
    print('P-corr:', '%6.4f' % pcorr)
    print('S-corr:', '%6.4f' % scorr)
    print('RMSE:', '%6.4f' % rmse)

    print('average on folds')
    print('P-corr:', '%6.4f' % np.mean(xcorrpear))
    print('S-corr:', '%6.4f' % np.mean(xcorrspear))
    print('RMSE:', '%6.4f' % np.mean(xrmse))

    ## return the Pearson, Spearman correlations and the rmse
    return(xcorrpear,xcorrspear,xrmse)

    
## ###################################################################
def get_data_from_SQLite(sql_db, fp='', per='', dataset='Small'):
    '''
    Retrieve data in np.ndarray format by specifying fingerprint and omics percent:
    fignerprints: '', maccs, circular, estate, extended, graph, hybrid, pubchem, shortp, standard.
    ('' : no fingerprint; shortp : shortest path; circular : ECFP6)
    omics percent: '', 05, 1, 2, 5 (no omics, 0.5%, 1%, 2%, 5%)
    dataset: 'Full' or 'Small'
    '''
    
    if dataset == 'Small':
        combo = 'Combo_Sub'
        drug  = 'Drugs_Sub'
        conc  = 'Concs_Sub'
    elif dataset == 'Full':
        combo = 'Combo'
        drug  = 'Drugs'
        conc  = 'Concs'
    else:
        print(get_data_from_SQLite.__doc__)
        raise ValueError('Wrong dataset!')
    
    if fp != '':
        fp = 'd1.%s, d2.%s,'%(fp, fp)
    if per != '':
        per = 'c.gene_expression_%s, c.gene_cnv_%s, c.crispr_ko_%s, c.proteomics_%s,'\
              %(per, per, per, per)
    print(fp, per)
    
    def adapt_array(arr):
        return arr.tobytes()
    
    def convert_array(text):
        return np.frombuffer(text)
    
    sqlite3.register_adapter(np.ndarray, adapt_array)    
    sqlite3.register_converter("array", convert_array)
    
    conn = sqlite3.connect(sql_db, detect_types=sqlite3.PARSE_DECLTYPES)
    
    cursor = conn.cursor()

    cursor.execute('''
                    SELECT d1.drug_code, d2.drug_code, conc1.conc_code, conc2.conc_code, c.cell_code, 
                    %s %s 
                    combo.response FROM %s combo 
                    INNER JOIN %s d1 ON d1.NSC = combo.drug1
                    INNER JOIN %s d2 ON d2.NSC = combo.drug2
                    INNER JOIN Cells c ON c.cell_name = combo.cell
                    INNER JOIN %s conc1 ON conc1.conc_value = combo.conc1
                    INNER JOIN %s conc2 ON conc2.conc_value = combo.conc2
                    ORDER BY combo.order_id
                    '''%(fp, per, combo, drug, drug, conc, conc)
                    )
    data_array = np.array([np.concatenate(i) for i in cursor.fetchall()])
    
    print('Data loaded from SQLite! Shape: ', data_array.shape)
    
    conn.close()
    
    return data_array



try:
    idx_run = int(sys.argv[1])
except:
    print('Wrong array id!')
    exit()


list_fp = ['', 'maccs']
list_data = ['', '1']
list_scene = ['S1', 'S2', 'S3', 'S4']
dataset = 'Full'


idx_tuple = np.unravel_index(idx_run, (len(list_fp), len(list_data), len(list_scene)))
i_fp = list_fp[idx_tuple[0]]
i_data = list_data[idx_tuple[1]]
i_scene = list_scene[idx_tuple[2]]
job_id = '_'.join([dataset, i_fp, i_data, i_scene])
print(job_id)

### Load training data with single drug cell data
## data directory
cwd = os.getcwd()
sdir = '.'
outdir = cwd + "/comboLTR_results/" + job_id + '/'
print(outdir)
if not os.path.exists(outdir):
    os.makedirs(outdir)

with open(sdir + '/CV_Folds_%s_%s.List.Pickle' %(i_scene, dataset), 'rb') as f:
    CV_indices_list = pickle.load(f)
print('CV splits: ', [len(j) for i in CV_indices_list for j in i])


## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
inorminf = 1  ## normalization by infinite norm ->[-1,+1]
dnormscale=10 ## scaling the norm
## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

Data = get_data_from_SQLite(sdir + '/DrugCombo.db', i_fp, i_data, dataset)
print('SLQ db used: ', sdir + '/DrugCombo.db')
X = Data[:, :-1]
y = Data[:, -1]

m = len(y)
ndim = X.shape[1]

ymean = np.mean(y)
print('Ymean: ', ymean)


##((((((((((((((((((((((((((((((
## output normalization
y -= np.mean(y)
ymax = np.max(np.abs(y))
print('Ymax:', np.max(np.abs(y)))
y /= np.max(np.abs(y))

## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
if inorminf==1:
    X = X / np.outer(np.ones(m), dnormscale*np.max(np.abs(X), 0))
## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

print('Dataset shape: {}'.format(X.shape))
print('Non-zeros rate: {:.05f}'.format(np.mean(X != 0)))

del Data

dstat={}

lstepsize=[ 0.01 ]
lcregular=[ 0.0001 ]
## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

orders = range(5, 6)
ranks = range(20,25,5)
repeats = range(20, 21)
for n_order in orders:
    for n_rank in ranks:
        for n_repeat in repeats:
            ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            for n_step in lstepsize:
                for nc in lcregular:
                    ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    print('*' * 50)
                    print('Stepsize:',n_step,',','Cregular:',nc)
                    print('Order: ', n_order, 'Rank: ', n_rank, 'Epoch: ', n_repeat)
                    ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    xcorrpear,xcorrspear,xrmse = \
                        main(CV_indices_list, n_order, n_rank, n_repeat, \
                             n_sigma = n_step, ncreg = nc)
                    ## collect all results for all parameter combinations 
                    dstat[(n_order, n_rank, n_repeat, n_step, nc)] = \
                        (np.mean(xcorrpear), np.mean(xcorrspear), np.mean(xrmse))
                    ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    

## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                    
## summary result on all parameter combination
for tkey in dstat.keys():
    for keyitem in tkey:
        print(str('%6.4f'%keyitem)+',', end='')
    print(',', end='')
    xvalue=dstat[tkey]
    for value in xvalue:
        print(str('%6.4f'%value)+',', end='')
    print()
## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                                                    
                            
