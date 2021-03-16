######################
## Version 0.10 #######
######################
"""
**********************************************************************
   Copyright(C) 2020 Sandor Szedmak  
   email: sandor.szedmak@aalto.fi
          szedmak777@gmail.com

   This file contains the code for Polynomial regression via latent
   tensor reconstruction (PRLTR.

   PRLTR is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version. 

   PRLTR is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with PRLTR.  If not, see <http://www.gnu.org/licenses/>.

***********************************************************************/
"""

"""
Polynomial regression via latent tensor reconstruction
Activation funtion   a(X\lambda_uU)V^{T} 

Version 0.10 (15.03.2021)
"""

"""
Vector output, one view, ranks range processed as in one step
polynomial encoding, a(X\lambda_{u}U)V^{T} 
"""

## #####################################################
import sys
import time
import pickle

import numpy as np

## #####################################################
## import matplotlib
## import matplotlib.pyplot as plt
## import matplotlib.cm as cmplot
## ## import matplotlib.animation as animation
## import mpl_toolkits.mplot3d.axes3d as p3
## import mpl_toolkits.mplot3d.art3d as art3d
## ## from matplotlib.collections import Line3DCollection
## from matplotlib.colors import colorConverter
## ###################################################

## ################################################################
## ################################################################
## ################################################################
class store_cls:

  def __init__(self):
    self.lPstore=None
    return
  
## ################################################################
class tensor_latent_vector_cls:
  """
  Task: to implement the latent tensor reconstruction based
        polynomial regression 
  """
  
  ## -------------------------------------------------
  def __init__(self,norder=1,nordery=1,rank=10,rankuv=None):
    """
    Task: to initialize the ltr object
    Input:  norder    input degree(oreder) of the polynomial
            nordery   output degree, in the current version it has to be 1 !!!
            rank      the number of rank-one layers
            rankuv    common rank in UV^{T}  U ndimx,rankuv, V nrank,nrankuv
    """

    self.ifirstrun=1       ## =1 first fit run
                           ## =0 next run for new data or 

    self.norder=norder      ## input degree(order)
    self.nordery=nordery    ## output degree
    ## -----------------------------------------
    self.nrank0=rank   ## initial saved maximum rank
    ## initial maximum rank, in the rank extension self.nrank it is changed
    ## if in function fit irank_add=1 
    self.nrank=rank
    ## ------------------------------------------
    self.nrankuv=rankuv  ## P=UV^{T},  U ndimx,nrankuv  V nrank,nrankuv

    self.nminrank=0    ## initial minimum rank

    self.lranks=[]     ## accumulated (nminrank,nrank) in the rank blocks
    self.rankcount=0   ## number of extensions

    self.ndimx=0   ## view wise input dimensions
    self.ndimy=0      ## output dimension

    self.iter=0       ## iteration counter used in the ADAM update

    self.cregular=0.000005
    self.cP=self.cregular           ## regularization penalty on U,V
    self.cQ=self.cregular           ## regularization penalty on Q
    self.clambda=self.cregular      ## lambda regularization constant

    ## parameter initialization
    # self.xP=None      ## input poly parameters (order,rank,ndimx)
    # self.xPnext=None  ## input next poly parameters in NAG
    # self.xGrad=None   ## the gardients of xP vectors
    # self.xV=None      ## aggregated gradients of xP
    # self.xnG=None     ## aggregated gradient lenght^2

    self.xU=None      ## input poly parameters (order,rankuv,ndimx)
    self.xUnext=None  ## input next poly parameters in NAG
    self.xGradU=None   ## the gardients of xU vectors
    self.xAU=None      ## aggregated gradients of xU
    self.xnAU=None     ## aggregated gradient lenght^2

    self.xV=None      ## input poly parameters (order,rank,nrankuv)
    self.xVnext=None  ## input next poly parameters in NAG
    self.xGradV=None   ## the gardients of xP vectors
    self.xAV=None      ## aggregated gradients of xP
    self.xnAV=None     ## aggregated gradient lenght^2

    self.xQ=None      ## output poly parameter  (rank,dimy)
    self.xQnext=None  ## output next in NAG
    self.xGradQ=None  ## the gradients of xQ
    self.xAQ=None     ## the aggregated gradients of xQ
    self.xnAQ=None    ## the aggregated gradient length^2

    self.xlambda=None  ## lambda factors 
    self.xlambdanext=0  ## next for NAG 
    self.xlambdagrad=0  ## lambda gradient
    self.xAlambda=0     ## aggregated lambda gradients
    self.xnAlambda=0    ## aggrageted lambda gadient length 

    self.xlambdaU=None  ## lambda factors 
    self.xlambdanextU=0  ## next for NAG
    self.xlambdagradU=0  ## lambda gradient
    self.xAlambdaU=0     ## aggregated lambda gradients
    self.xnAlambdaU=0    ## aggrageted lambda gadient length 

    self.ilambda=1     ## xlambda is updated by gradient
    self.ilambdamean=0  ## =1 mean =0 independent components


    self.f=None      ## computed function value
    self.yerr=[]     ## rmse block error
    
    self.irandom=1  ## =1 random block =0 order preserving blocks
    self.iscale=1   ## =1 error average =0 no
    self.dscale=2   ## x-1/dscale *x^2 stepsize update

    ## ADAM + NAG
    self.sigma0=0.1     ## initial learning speed
    self.sigma=0.1     ## updated learning speed
    self.sigmabase=1     ## sigma scale
    self.gamma=1.0     ## discount factor
    self.gammanag=0.99  ## Nesterov accelerated gradient factor
    self.gammanag2=0.99  ## Nesterov accelerated gradient factor
    self.ngeps=0.00001     ## ADAM correction to avoid 0 division 
    self.nsigma=1 ## range without sigma update
    self.sigmamax=1   ## maximum sigma*len_grad
    
    self.iyscale=1      ## =1 output vectors are scaled
    self.yscale=1       ## the scaling value

    self.mblock=10          ## data block size, number of examples
    self.mblock_gap=None      ## shift of blocks

    self.ibias=1          ## =1 bias is computed =0 otherwise
    self.pbias=None       ## bias vectors, matrix with size nrank,ndimy
    self.lbias=[]

    self.inormalize=1     ## force normalization in each iteration

    ## test environment
    self.istore=0
    self.cstore_numerator=store_cls()
    self.cstore_denominator=store_cls()
    self.cstore_output=store_cls()
    self.store_bias=None
    self.store_yscale=None
    self.store_lambda=None
    self.store_grad=[]

    self.store_acc=[]

    self.max_grad_norm=0  ## maximum gradient norm 
    self.maxngrad=0       ## to signal of long gradients

    self.Ytrain=None        ## deflated output vectors

    ## parameter variables which can be stored after training
    ## and reloaded in test
    self.lsave=['norder','nrank','iyscale','yscale','ndimx','ndimy', \
                'xU','xV','xQ','xlambda','pbias']

    ## activation function
    self.iactfunc=0    ## =0 identity, =1 arcsinh =2 2*sigmoid-1 =3 tanh =4 relu

    ## loss degree
    self.lossdegree=0  ## =0 L_2^2, =1 L_2, =0.5 L_2^{0.5}, ...L_2^{z}  

    ## Kolmogorov mean
    self.ikolmogorov=1 ## L_1 norm approximation log(cosh(tx))
    self.kolm_t=1      ## t factor in cosh 

    ## power regularization
    self.regdegree=1   ## degree of regularization

    ## penalty constant to enforce the orthogonality all P[order,rank,:] vectors
    self.iortho=0    ## =1 orthogonality forced =0 not
    self.cortho=0.0  ## penalty constant

    ## nonlinear regularization  \dfrac{\partial f}{\partial x \partial P}
    self.iregular_nonlinear=0  ## =1 regularize =0 not
    self.cregularnl=0.0005   ## penalty constant 
    
    self.report_freq=100 ## state report frequency relative to the number of minibatches.

    return

  ## ------------------------------------------------
  def init_lp(self):
    """
    Task: to initialize the parameter vectors U,V,Q, bias,lambda lambdaU
    Input:
    Output:
    Modifies: self.xU, self.xV, self,xQ, self.xlambda, self. xlambdaU, self.pbias
    """

    nrank=self.nrank
    nrankuv=self.nrankuv
    
    self.xU=np.random.randn(self.norder,self.ndimx,nrankuv)
    self.xV=np.random.randn(self.norder,nrank,self.nrankuv)
    self.xQ=np.random.randn(nrank,self.ndimy)

    self.xlambda=np.ones(nrank)
    self.xlambdaU=np.ones(self.ndimx)

    self.pbias=np.zeros((1,self.ndimy))
    
    return

  # ## ------------------------------------------------
  # def volume(self,irank):
  #   """
  #   Task: to compute of the volume spanned by vectors
  #         xP[0][irank],...,xP[norder-1][irank]
  #   Output: vol  scalar volume
  #   """

  #   P=np.array([ self.xP[t,irank] for t in range(self.norder)])
  #   PP=np.dot(P,P.T)
  #   vol=np.linalg.det(PP)
  #   if vol<0:
  #     vol=0
  #   vol=np.sqrt(vol)

  #   return(vol)
      
  # ## ------------------------------------------------
  def init_grad(self):
    """
    Task: to initialize the gradients
    Input:
    Output:
    Modifies: 
              self.xGradU   the gradient of the xU
              self.xAU      the accumulated gradient of xU
              self.xnAU     the accumulated gradient norms of xU
              self.xUnext  the pushforward xU ( Nesterov accelerated gradient)
              
              self.xGradV   the gradient of the xV
              self.xAV      the accumulated gradient of xV
              self.xnAV     the accumulated gradient norms of xV
              self.xVnext   the pushforward xU ( Nesterov accelerated gradient)

              self.xGradQ  the gradient of the xQ
              self.xAQ     the accumulated gradient of xQ
              self.xnAQ    the accumulated gradient norms of xQ
              self.xQnext  the pushforward xQ ( Nesterov accelerated gradient)
    
    """

    drank=self.nrank-self.nminrank
    ndimx=self.ndimx
    ndimy=self.ndimy
    norder=self.norder
    nrankuv=self.nrankuv
        
    self.xGradU=np.zeros((norder,ndimx,nrankuv))
    self.xAU=np.zeros((norder,ndimx,nrankuv))
    self.xnAU=np.zeros((norder,ndimx))
    self.xUnext=np.zeros((norder,ndimx,nrankuv))

    self.xGradV=np.zeros((norder,drank,nrankuv))
    self.xAV=np.zeros((norder,drank,nrankuv))
    self.xnAV=np.zeros((norder,drank))
    self.xVnext=np.zeros((norder,drank,nrankuv))

    self.xGradQ=np.zeros((drank,ndimy))
    self.xAQ=np.zeros((drank,ndimy))
    self.xnAQ=np.zeros((drank))
    self.xQnext=np.zeros((drank,ndimy))
    
    self.xlambdagrad=np.zeros(drank)
    self.xAlambda=np.zeros(drank)
    self.xnAlambda=0
    self.xlambdanext=np.zeros(drank)

    self.xlambdagradU=np.zeros(ndimx)
    self.xAlambdaU=np.zeros(ndimx)
    self.xnAlambdaU=0
    self.xlambdanextU=np.zeros(ndimx)

    return

  ## -------------------------------------------------
  def extend_poly_parameters_rank(self,nextrank):
    """
    Task: extedn the parameter lists to the next rank
    Input:  nextrank   the extended rank beyond self.nrank
    Output:
    Modifies: self.xV, self.xQ, self.xlambda, self.pbias
    """

    nrank=self.nrank
    nrankuv=self.nrankuv
    ndimx=self.ndimx
    ndimy=self.ndimy

    if nextrank>nrank:
      drank=nextrank-nrank
      xprev=np.copy(self.xV)
      self.xV=np.zeros((self.norder,nextrank,nrankuv))
      for d in range(self.norder):
        self.xV[d]=np.vstack((xprev[d],np.random.randn(drank,nrankuv)))

      xprev=np.copy(self.xQ)
      self.xQ=np.vstack((xprev,np.random.randn(drank,ndimy)))

      self.xlambda=np.concatenate((self.xlambda,np.ones(drank)))

      self.pbias=np.vstack((self.pbias,np.zeros(ndimy)))

    return

  ## -------------------------------------------------
  def update_parameters(self,**dparams):
    """
    Task: to update the initialized parameters
    Input:  dprams  dictionary { parameter name : value }
    Output:
    Modifies: corresponding parameters
    """

    for key,value in dparams.items():
      if key in self.__dict__:
        self.__dict__[key]=value

    if self.mblock_gap is None:
      self.mblock_gap=self.mblock

    return

  ## ------------------------------------------------
  def normalize_lp(self,ilambda=1):
    """
    Task: to project, normalize by L2 norm, the polynomial parameters
                      xP, xQ
    Input: 
           ilambda =1 xlambda[irank], vlambda nglambda is updated
                      with the product of lenght of the parameter
                      vectors before normalization
    Output: xnormlambda  the product of lenght of the parameter
                      vectors before normalization
    Modifies:      xP, xQ
                   or
                   xP[irank], xQ[irank], xlambda[irank], vlambda nglambda 
    """

    nrank=self.nrank
    nrankuv=self.nrankuv
    nminrank=self.nminrank
    drank=nrank-nminrank
    
    xnormlambda=np.ones(drank)
    xnormlambdaU=np.ones(self.ndimx)

    for d in range(self.norder):
      xnorm=np.sqrt(np.sum(self.xU[d]**2,1))
      xnorm=xnorm+(xnorm==0)
      ## xnorm/=np.sqrt(nrankuv)
      self.xU[d]=self.xU[d] \
        /np.outer(xnorm,np.ones(nrankuv))
      xnormlambdaU*=xnorm
    
    for d in range(self.norder):
      xnorm=np.sqrt(np.sum(self.xV[d,nminrank:nrank]**2,1))
      xnorm=xnorm+(xnorm==0)
      ## xnorm/=np.sqrt(nrankuv)
      self.xV[d,nminrank:nrank]=self.xV[d,nminrank:nrank] \
        /np.outer(xnorm,np.ones(nrankuv))
      ## xnormlambda*=xnorm

    xnorm=np.sqrt(np.sum(self.xQ[nminrank:nrank]**2,1))
    xnorm=xnorm+(xnorm==0)
    ## xnorm/=np.sqrt(self.ndimy)
    self.xQ[nminrank:nrank]=self.xQ[nminrank:nrank] \
        /np.outer(xnorm,np.ones(self.ndimy))
    ## xnormlambda*=xnorm

    self.xlambdaU*=xnormlambdaU
    xnorm=np.sqrt(np.sum(self.xlambdaU**2))
    xnorm=xnorm+(xnorm==0)
    xnorm/=np.sqrt(self.ndimx)
    self.xlambdaU/=xnorm
    ## xnormlambda*=xnorm
    
    self.xlambda[nminrank:nrank]*=xnormlambda
    self.xAlambda*=np.power(xnormlambda,0.5)
    ## to avoid overflow
    ## self.xnAlambda*=np.prod(xnormlambda)
    xscale=np.mean(xnormlambda)
    if np.min(xscale)>0:
      ## self.xnAlambda*=np.prod(np.power(xnormlambda,0.1))
      self.xnAlambda/=xscale
    
    return(xnormlambda)

  ## --------------------------------------------
  def update_lambda_matrix_bias(self,X,Y):
    """
    Task: to compute the initial estimate of xlambda and the bias
    Input:  X     list of 2d arrays, the arrays of input block views
            Y      2d array  of output block
    Output: xlambda  real, the estimation of xlambda
            bias     vector of bias estimation
    """

    nrank=self.nrank
    nminrank=self.nminrank
    norder=self.norder
    drank=nrank-nminrank
    nrankuv=self.nrankuv
    m=len(X)

    ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    Fr=np.array([None for _ in range(norder)]) 
    for d in range(norder):
      XU=np.dot(X,self.xU[d]*np.outer(self.xlambdaU,np.ones(nrankuv)))
      AXU=self.activation_func(XU,self.iactfunc)
      Fr[d]=np.dot(AXU,self.xV[d].T)
    ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    if len(Fr)>1:
      F=np.prod(Fr,0)
    else:
      F=np.copy(Fr[0])

    Q=self.xQ[nminrank:nrank]
      
    nt,ny=Q.shape
    m=Y.shape[0]

    ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    em=np.ones(m)
    YQT=np.dot(Y,Q.T)
    QQT=np.dot(Q,Q.T)
    FTF=np.dot(F.T,F)
    f1=np.dot(em,F)
    ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    xright=np.dot(em,F*YQT)-np.dot(em,F*np.outer(em,np.dot(em,YQT)))/m
    xleft=QQT*(FTF-np.outer(f1,f1)/m)
    xlambda=np.dot(xright,np.linalg.pinv(xleft))
    bias=np.dot(em,(Y-np.dot(F,np.outer(xlambda,np.ones(ny))*Q)))/m
      
    return(xlambda,bias)

  ## --------------------------------------------
  def nag_next(self,gamma=0,psign=-1):
    """
    Task: to compute the next point for Nesterov Acclerated gradient
    Input: gamma    =0 no change , gradient scale otherwise
           psign    =0 no change, -1 pull back, +1 push forward
    """

    ## pull back or push forward
    psign=-1  ## -1 pull back, +1 push forward

    pgamma=psign*gamma
    
    nrank=self.nrank
    nminrank=self.nminrank
    
    for d in range(self.norder):
      self.xVnext[d]=self.xV[d,nminrank:nrank]+pgamma*self.xAV[d]
      self.xUnext[d]=self.xU[d]+pgamma*self.xAU[d]

    self.xQnext=self.xQ[nminrank:nrank]+pgamma*self.xAQ
      
    self.xlambdanext=self.xlambda[nminrank:nrank]+pgamma*self.xAlambda
    
    self.xlambdanextU=self.xlambdaU+pgamma*self.xAlambdaU
        
    return

  # ## -----------------------------------------------------
  # def update_parameters_nag(self):
  #   """
  #   Task:  to update the parameters of a polynomial, cpoly,
  #          based on the Nesterov accelerated gradient
  #   Input:   irank   rank index in xP,xQ, xlambda 
  #   Modify:  xV, xP, xQ, xVQ, xlambda, vlambda
  #   """

  #   norder=self.norder
  #   nrank=self.nrank
  #   nminrank=self.nminrank

  #   xnorm=np.zeros(norder)
  #   for d in range(norder):
  #     xnorm[d]=np.linalg.norm(self.xGrad[d])
  #   ## self.store_grad.apprnd(xnorm)
  #   xmax=np.max(xnorm)
  #   if xmax>self.max_grad_norm:
  #     self.max_grad_norm=xmax
  #     ## print('Grad norm max:',xmax)
    
  #   if self.sigma*xmax>self.sigmamax:
  #     sigmacorrect=self.sigmamax/(self.sigma*xmax)
  #     ## print('>>>',self.sigma*xmax,sigmacorrect)
  #   else:  
  #     sigmacorrect=1

  #   for d in range(norder):
  #     self.xV[d]=self.gammanag*self.xV[d] \
  #     -sigmacorrect*self.sigmabase*self.sigma*self.xGrad[d]

  #   for d in range(norder):
  #     self.xP[d,nminrank:nrank]=self.gamma*self.xP[d,nminrank:nrank]+self.xV[d]

  #   self.xVQ=self.gammanag*self.xVQ \
  #     -sigmacorrect*self.sigmabase*self.sigma*self.xGradQ

  #   self.xQ[nminrank:nrank]=self.gamma*self.xQ[nminrank:nrank]+self.xVQ

  #   self.vlambda=self.gammanag*self.vlambda \
  #     -sigmacorrect*self.sigmabase*self.sigma \
  #     *self.xlambdagrad
  #   self.xlambda[nminrank:nrank]=self.gamma*self.xlambda[nminrank:nrank] \
  #     +self.vlambda
        
  #   return

  # ## -----------------------------------------------------
  def update_parameters_adam(self):
    """
    Task:  to update the parameters of a polynomial, cpoly,
           based on the ADAM additive update
    Input:   
    Modify:  xU,XV,xQ, xAU, xAV,xAQ, xnAU, xnAV,xnAQ, 
             xlambda, xAlambda, xnAlambda,
             xlambdaU, xAlambdaU, xnAlambdaU,
    """

    norder=self.norder
    nrank=self.nrank
    nrankuv=self.nrankuv
    nminrank=self.nminrank

    xnormU=np.zeros(norder)
    xnormV=np.zeros(norder)
    for d in range(norder):
      xnormU[d]=np.linalg.norm(self.xGradU[d])
      xnormV[d]=np.linalg.norm(self.xGradV[d])
      
    ## self.store_grad.apprnd(xnorm)
    xmax=np.max(np.vstack((xnormU,xnormV)))
    if xmax>self.max_grad_norm:
      self.max_grad_norm=xmax
      ## print('Grad norm max:',xmax)
    
    if self.sigma*xmax>self.sigmamax:
      sigmacorrect=self.sigmamax/(self.sigma*xmax)
      ## print('>>>',self.sigma*xmax,sigmacorrect)
    else:  
      sigmacorrect=1

    gammanag=self.gammanag
    gammanag2=self.gammanag2

    for d in range(norder):

      ## xU ---------------------------
      ngrad=np.sum((sigmacorrect*self.xGradU[d])**2,1)

      self.xAU[d]=gammanag*self.xAU[d]+(1-gammanag) \
        *sigmacorrect*self.xGradU[d]  

      vhat=self.xAU[d]/(1-gammanag**self.iter)

      self.xnAU[d]=gammanag2*self.xnAU[d]+(1-gammanag2)*ngrad
      ngradhat=self.xnAU[d]/(1-gammanag2**self.iter)

      self.xU[d]=self.gamma*self.xU[d] \
        -self.sigmabase*self.sigma \
        *vhat/(np.outer(np.sqrt(ngradhat),np.ones(self.nrankuv))+self.ngeps) 

      ## xV -----------------------
      ngrad=np.sum((sigmacorrect*self.xGradV[d])**2,1)

      self.xAV[d]=gammanag*self.xAV[d]+(1-gammanag) \
        *sigmacorrect*self.xGradV[d]  

      vhat=self.xAV[d]/(1-gammanag**self.iter)

      self.xnAV[d]=gammanag2*self.xnAV[d]+(1-gammanag2)*ngrad
      ngradhat=self.xnAV[d]/(1-gammanag2**self.iter)

      self.xV[d,nminrank:nrank]=self.gamma*self.xV[d,nminrank:nrank] \
        -self.sigmabase*self.sigma \
        *vhat/(np.outer(np.sqrt(ngradhat),np.ones(self.nrankuv))+self.ngeps) 

    ## xQ -----------------------------------------    
    ngrad=np.sum((sigmacorrect*self.xGradQ)**2,1)

    self.xAQ=gammanag*self.xAQ \
      +(1-gammanag)*sigmacorrect*self.xGradQ
    vhat=self.xAQ/(1-gammanag**self.iter)

    ## print('Qgrad:',ngrad)
    if self.maxngrad<np.sum(ngrad):
      self.maxngrad=np.sum(ngrad)
      if self.maxngrad>1.0:
        print('Max Qgrad:',self.maxngrad)

    self.xnAQ=gammanag2*self.xnAQ \
      +(1-gammanag2)*ngrad
    ngradhat=self.xnAQ/(1-gammanag2**self.iter)

    self.xQ[nminrank:nrank]=self.gamma*self.xQ[nminrank:nrank] \
        -self.sigmabase*self.sigma \
        *vhat/(np.outer(np.sqrt(ngradhat),np.ones(self.ndimy))+self.ngeps)

    ## lambda ------------------------------------------------    

    ngrad=np.sum((sigmacorrect*self.xlambdagrad)**2)

    self.xAlambda=gammanag*self.xAlambda \
      +(1-gammanag)*sigmacorrect*self.xlambdagrad
    vhat=self.xAlambda/(1-gammanag**self.iter)

    self.xnAlambda=gammanag2*self.xnAlambda \
      +(1-gammanag2)*ngrad
    ngradhat=self.xnAlambda/(1-gammanag2**self.iter)

    self.xlambda[nminrank:nrank]=self.gamma*self.xlambda[nminrank:nrank] \
      -self.sigmabase*self.sigma \
      *vhat/(np.sqrt(ngradhat)+self.ngeps)
        
    ## lambdaU ------------------------------------------------    

    ngrad=np.sum((sigmacorrect*self.xlambdagradU)**2)

    self.xAlambdaU=gammanag*self.xAlambdaU \
      +(1-gammanag)*sigmacorrect*self.xlambdagradU
    vhat=self.xAlambdaU/(1-gammanag**self.iter)

    self.xnAlambdaU=gammanag2*self.xnAlambdaU \
      +(1-gammanag2)*ngrad
    ngradhat=self.xnAlambdaU/(1-gammanag2**self.iter)

    self.xlambdaU=self.gamma*self.xlambdaU \
      -self.sigmabase*self.sigma \
      *vhat/(np.sqrt(ngradhat)+self.ngeps)

    return

  ## ------------------------------------------------
  def activation_func(self,f,ifunc=0):
    """
    Task: to compute the value of activation function
    Input:  f      array of input
            ifunc  =0  identity
                   =1  arcsinh  ln(f+(f^2+1)^{1/2})
                   =2  sigmoid  2e^x/(e^x+1)-1
                   =3  tangent hyperbolisc
    Output: F      array of activation values
    """

    if ifunc==0:
      F=f          ## identity
    elif ifunc==1: 
      F=np.log(f+(f**2+1)**0.5)   ## arcsinh
    elif ifunc==2:
      F=2/(1+np.exp(-f))-1   ## sigmoid
    elif ifunc==3: 
      F=np.tanh(f)  ## tangent hyperbolic
    elif ifunc==4: ## relu
      F=f*(f>0)
      
    return(F)
  ## ------------------------------------------------
  def activation_func_diff(self,f,ifunc=0,ipar=1):
    """
    Task: to compute the value of the pointwise derivative of
          activation function
    Input:  f      array of input
            ifunc  =0  identity
                   =1  arcsinh  ln(f+(f^2+1)^{1/2})
                   =2  sigmoid  e^x/(e^x+1)
                   =3  tangent hyperbolisc
    Output: DF     array of pointwise drivative of the activation function
    """

    m,n=f.shape
    if ifunc==0:
      DF=np.ones((m,n))          ## identity
    elif ifunc==1: 
      DF=1/(f**2+1)**0.5   ## arcsinh
    elif ifunc==2:               ## sigmoid 
      DF=2*np.exp(-f)/(1+np.exp(-f))**2
    elif ifunc==3: 
      DF=1/np.cosh(f)**2  ## tangent hyperbolic
    elif ifunc==4:  ## relu
      DF=ipar*(f>0) 
      
    return(DF)
  ## ------------------------------------------------
  def function_value(self,X,rankcount,xU=None,xV=None,xQ=None, \
                       xlambda=None,xlambdaU=None,bias=None, \
                       ifunc=None,irange=0):
    """
    Task:  to compute the rank related function value
           f=\lambda \circ_r Xp_r q^T +bias
    Input: X          list of 2d array of input data views
           rankcount  index of rank-block
           xU         tensor of input parameter arrays, (norder,nrank,nrankuv)
           xV         tensor of input parameter arrays, (norder,ndimx,nrankuv)
           xQ         matrix of output parameter arrays, (nrank,ndimy) 
           xlambda    singular values   (nrank) 
           xlambdaU   data variable weight values   (ndimx) 
           bias       vector of bias    (ndimy)
           ifunc  None => 0
                  =0  identity
                  =1  arcsinh  ln(f+(f^2+1)^{1/2})
                  =2  sigmoid  e^x/(e^x+1)
    Output: f    2d array =\sum  \circ_t XP^(t)T M_{\lambda} Q  +bias  
    """

    nminrank,nrank=self.lranks[rankcount]
    drank=nrank-nminrank
    nrankuv=self.nrankuv

    m,n=X.shape

    ## temporal case
    ifunc=0
    if ifunc is None:
      ifunc=0
    
    f=np.zeros((m,self.ndimy))

    if xU is None:
      xU=self.xU
    if xV is None:
      xV=self.xV[:,nminrank:nrank]
    if xQ is None:
      xQ=self.xQ[nminrank:nrank]
      
    if xlambda is None:
      xlambda=self.xlambda[nminrank:nrank]
    if xlambdaU is None:
      xlambdaU=self.xlambdaU
    
    if bias is None:
      bias=self.pbias[rankcount]
    
    ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    F0=np.ones((m,drank))
    for d in range(self.norder):
      XU=np.dot(X,self.xU[d]*np.outer(self.xlambdaU,np.ones(nrankuv)))
      AXU=self.activation_func(XU,self.iactfunc)
      F0*=np.dot(AXU,self.xV[d].T)

## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    F=np.dot(F0,np.outer(xlambda,np.ones(self.ndimy))*xQ)
    F+=np.outer(np.ones(m),bias)

    return(F)

  ## ------------------------------------------------
  def gradient(self,X,y,rankcount,xU=None,xV=None,xQ=None, \
                 xlambda=None,xlambdaU=None,bias=None,icount=None):
    """
    Task:  to compute the gradients for xP, xQ, xlambda
    Input: X          2d array of input data block
           y          2d array of output block
           rankcount  index of rank-block
           xU         tensor of parameters (norder,ndimx,nrankuv)
           xV         tensor of parameters (norder,nrank,nrankuv)
           xQ         2d array (nrank,ndimy)
           xlambda    vector (nrank)
           xlambdaU   vector (ndimx)
           bias       vector (ndimy)
    Output:
    Modifies:  self.xGradU, self.xGradV, self.xGradQ, 
              self.xlambdagrad, self.xlambdagradU
    """
    norder=self.norder
    nrankuv=self.nrankuv
    ndimx=self.ndimx
    ndimy=self.ndimy

    nminrank,nrank=self.lranks[rankcount]
    drank=nrank-nminrank

    m=X.shape[0]

    if xU is None:
      xU=self.xU
    if xV is None:
      xV=self.xV[:,nminrank:nrank]
    if xQ is None:
      xQ=self.xQ[nminrank:nrank]
      
    if bias is None:
      bias=self.pbias[self.rankcount]
    if xlambda is None:
      xlambda=self.xlambda[nminrank:nrank]
    if xlambdaU is None:
      xlambdaU=self.xlambdaU

    ## setting the regularization constants
    self.cP=self.cregular           ## regularization penalty on P
    self.cQ=self.cregular           ## regularization penalty on Q
    self.clambda=self.cregular      ## lambda regularization constant

    ## scaling the loss and the regularization
    if self.iscale==1:
      scale_loss=1/(m*ndimy)
      scale_lambda=1/drank
    else:
      scale_loss=1
      scale_lambda=1

    self.xGradU=np.zeros((norder,ndimx,nrankuv))
    self.xGradV=np.zeros((norder,drank,nrankuv))
    self.xGradQ=np.zeros((drank,ndimy))

    xXUV=np.array([None for _ in range(norder)]) 
    xXU=np.array([None for _ in range(norder)]) 
    xActD=np.array([None for _ in range(norder)])

    ## Compute the transformations of X by P_d 
    ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    for d in range(norder):
      XU=np.dot(X,self.xU[d]*np.outer(self.xlambdaU,np.ones(nrankuv)))
      AXU=self.activation_func(XU,self.iactfunc)
      xXU[d]=AXU
      xActD[d]=self.activation_func_diff(XU,self.iactfunc)
      xXUV[d]=np.dot(AXU,self.xV[d].T)

    ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    F0=np.prod(xXUV,0)    ## \circ_d XM_{lambdaU}U^((d)}V^{(d)}T}M_{\lambda} 

    ## entire predictor function values
    H=np.dot(F0,np.outer(xlambda,np.ones(ndimy))*xQ)   
    ## error
    ferr=H+np.outer(np.ones(m),bias)-y   ## the loss, error
    
    ## if the loss is not least square change it
    ## ikolmogorov=1 ## =1 kolmogorov mean variant =0 not
    ## kolm_t=1  ## kolmogorov mean scale factor
    if self.ikolmogorov==0:
      if self.lossdegree>0:
        dferr=np.linalg.norm(ferr)
        if dferr==0:
          dferr=1
        dferr=dferr**self.lossdegree
        ferr/=dferr
    elif self.ikolmogorov==1:
      ferr=np.tanh(self.kolm_t*ferr)
    elif self.ikolmogorov==2:
      dferr=np.sqrt(np.sum(ferr**2,1))
      dferr=dferr+(dferr==0)
      ferr=ferr*np.outer(np.tanh(self.kolm_t*dferr)/dferr,np.ones(ndimy))
    
    ## averaging the loss on the min-batch
    ferr=ferr*scale_loss

    ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2
    ## computation the gradients
    ## =================================================
    ## computing more than ones occuring terms and factors to save time
    ferrQT=np.dot(ferr,xQ.T)

    ## =================================================
    ## gradient not depending on degree
    self.xGradQ=np.dot(F0.T,ferr)*np.outer(xlambda,np.ones(ndimy))
    self.xlambdagrad=np.sum(F0*ferrQT,0)
        
    ## ================================================  
    ## compute F_{\subsetminus d}
    self.xlambdagradU=np.zeros(ndimx)
    if norder>1:
      for d in range(norder):
        ipx=np.arange(norder-1)
        if d<norder-1:
          ipx[d:]+=1
        Zd=np.prod(xXUV[ipx],0)    ## Z^{(d)}
        dEQF=Zd*ferrQT
        dEQFV=np.dot(dEQF,xV[d]*np.outer(xlambda,np.ones(nrankuv)))
        dEQFVH=dEQFV*xActD[d]
        dXEQFVH=np.dot(X.T,dEQFVH)
        ##Gd=np.dot(X.T,Zd*xActD[d]*ferrQT)
        
        ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        self.xGradV[d]=np.outer(xlambda,np.ones(nrankuv))*np.dot(dEQF.T,xXU[d])
        self.xGradU[d]=dXEQFVH*np.outer(xlambdaU,np.ones(nrankuv))
        self.xlambdagradU+=np.sum(dXEQFVH*xU[d],1)
        ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    else:
      ## Zd is empty, or = ones(m,nrank)
      dEQF=ferrQT
      dEQFV=np.dot(dEQF,xV[d]*np.outer(xlambda,np.ones(nrankuv)))
      dEQFVH=dEQFV*xActD[d]
      dXEQFVH=np.dot(X.T,dEQFVH)
      ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
      self.xGradV[d]=np.outer(xlambda,np.ones(nrankuv))*np.dot(dEQF.T,xXU[d])
      self.xGradU[d]=dXEQFVH*np.outer(xlambdaU,np.ones(nrankuv))
      self.xlambdagradU+=np.sum(dXEQFVH*xU[d],1)
      ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    ## regularization terms
    if self.ilambda==1:
      self.xlambdagrad-=self.clambda*scale_lambda \
        *np.sign(xlambda)*np.abs(xlambda)**(self.regdegree-1)
          
    return

  ## ------------------------------------------------
  def incremental_step(self,X,y,icount):
    """
    Task: to compute one step of the iteration
    Input: X        2d array of input block
           y        2d array of output block
           icount     block index
    Modifies: via the called functions, xP,xQ and the gradient related variables
    """

    m,n=X.shape

    if self.inormalize==1:
      self.normalize_lp()

    ## if first parameter =0 then no Nesterov push forward step
    self.nag_next(gamma=self.gammanag,psign=-1)
    
    xlambdanext=self.xlambdanext
    xlambdanextU=self.xlambdanextU
    ## xlambdanext=None
    self.gradient(X,y,self.rankcount,xU=self.xUnext,xV=self.xVnext,xQ=self.xQnext, \
                  xlambda=xlambdanext, xlambdaU=xlambdanextU, \
                  bias=self.pbias[self.rankcount],icount=icount)

    ## self.update_parameters_nag()
    self.update_parameters_adam()

    ## self.xlambda=self.normalize_lp(ilambda=0)
    ## self.normalize_lp(ilambda=0)
        
    f=self.function_value(X,self.rankcount,xU=None,xV=None,xQ=None, \
              xlambda=None,xlambdaU=None,bias=None, \
              ifunc=None,irange=1)

    ## bias is averaged on all processed block and data
    if self.ibias==1:
      prev_bias=self.pbias[self.rankcount]
      self.pbias[self.rankcount]=np.mean(y-f,0) \
        /(icount+1)+prev_bias*icount/(icount+1)
      f+=np.outer(np.ones(m),self.pbias[self.rankcount]-prev_bias)  
        
    return(f)

  ## --------------------------------------------------
  def fit(self,Xtrain,Ytrain,nepoch=10,idata_add=0,irank_add=0,nextrank=0):
  ## def fit(self,lXtrain,Ytrain,nepoch=10, \
  ##         Y0=None,lXtest=None,Ytest=None,testeval=None):
    """
    Task: to compute the full training cycle
    Input: Xtrain        2d arrays of training input data
           Ytrain        2d array of training output
           nepoch       number of repetation of the iteration steps
           idata_add     =1 new data is added to the existing model
           irank_add     =1 rank range is extend to nextrank
                         in the existing model
           nextrank      the exteded rank range if irank_add==1
                         nextrank has to be greater than self.nrank
    
    Output:
    Modifies:  computes the polynomial parameters xP,xQ, xlambda,xbias
    """

    if idata_add==0 and irank_add==0:
      self.ifirstrun=1
      self.nrank=self.nrank0     ## initialize rank for rank extensions

    self.nrepeat=nepoch
    dscale=self.dscale

    mtrain,ndimx=Xtrain.shape
    self.ndimx=ndimx
    if self.nrankuv is None:
      self.nrankuv=ndimx

    ## reshape vector into matrix
    if len(Ytrain.shape)>1:
      self.ndimy=Ytrain.shape[1]
    else:
      Ytrain=Ytrain.reshape((mtrain,1))
      self.ndimy=1

    ## parameters initialized only in the first run
    ## additional data or rank run on the herited parameters
    if idata_add==0 and irank_add==0:  
      self.init_lp()

    self.Ytrain=np.copy(Ytrain)  ## for future
    
    xselect=np.arange(mtrain)

    if self.ifirstrun==1:
      self.nminrank=0
      self.rankcount=0
      self.lranks.append((self.nminrank,self.nrank))
    else:
      ## rank extension requires to extedn the parameter matrices
      if irank_add==1:
        if nextrank>self.nrank:
          self.nminrank=self.nrank
          self.extend_poly_parameters_rank(nextrank)
          self.nrank=nextrank
          self.rankcount+=1
          self.lranks.append((self.nminrank,self.nrank))
    
    ## self.store_lambda=[ [] for _ in range(self.nrank)]
    ## self.store_rmse=np.zeros(self.nrank)
    ## self.store_fpred=np.zeros((mtrain,self.ndimy,self.nrank))
    ## self.store_fpred_test=np.zeros((mtrain,self.ndimy,self.nrank))
        
    icount=0
    self.sigma=self.sigma0   ## initial learning rate
    ## self.init_lp()
    self.init_grad()
    self.max_grad_norm=0.0
      
    ifirst=1   ## for a new rank iteration initialization is needed        

    for irepeat in range(self.nrepeat):
      self.sigma-=self.sigma**2/dscale  ## learning rate update

      nblock=0
      ##print('mblock:',self.mblock)
      ## blocks might be random chosen 
      if self.irandom==1:
        np.random.shuffle(xselect)

      self.iter=1  ## iteration counter
      
      ## compute block lenght for chunk blocks
      if self.mblock_gap is None:
        mblock_gap=self.mblock
      else:
        mblock_gap=self.mblock_gap

      for iblock in range(0,mtrain,mblock_gap):
        
        if iblock+self.mblock<=mtrain:
          mb=self.mblock
        else:
          mb=mtrain-iblock
        if mb==0:
          break

        ## load random block
        ib=np.arange(iblock,iblock+mb)
        iib=xselect[ib]

        ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        ## if Xtrain, Ytrain are sparse matrices then blocks need to be
        ## converted here into dense 2darray
        ## block data
        x_b=Xtrain[iib]
        y_b=self.Ytrain[iib]
        ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        ## in the first iteration estimate lambda and the bias
        if ifirst==1:
          ## print('init 1')
          self.normalize_lp()
          xlambda,bias=self.update_lambda_matrix_bias(x_b,y_b)
          self.xlambda[self.nminrank:self.nrank]=xlambda
          prevlambda=xlambda
          self.pbias[self.rankcount]=bias
          ## self.yerr=np.sqrt(np.mean((f-y_i)**2))
          ifirst=0
          ## print('init 2')

        ## f,foutput shape rank,mblock
        f=self.incremental_step(x_b,y_b,icount)

        if icount%self.report_freq==0:

          if icount==160:
            print('!!!!!')
          # tdim=self.xP.shape
          # xPM=self.xP.reshape((tdim[0]*tdim[1],tdim[2]))
          # deye=np.sqrt(np.sum((np.eye(tdim[0]*tdim[1])-np.dot(xPM,xPM.T))**2))
          deye=np.linalg.norm(self.xnAlambda)
          
          print(self.nminrank,self.nrank,icount,irepeat,iblock, \
                '%7.4f'%np.linalg.norm(f-y_b), \
                '%7.4f'%np.corrcoef(f.ravel(),y_b.ravel())[0,1], \
                '%8.2f'%(np.linalg.norm(self.xlambda)),
                '%8.2f'%(np.linalg.norm(self.xlambdaU)),
                '%8.2f'%(np.linalg.norm(self.xQ)),
                '%8.2f'%(np.linalg.norm(self.xU)),
                '%8.2f'%(np.linalg.norm(self.xV)),
                '%8.2f'%deye)  
          ## print(self.xlambda)
          sys.stdout.flush()

        ## reduce the deacy of the learning speed,
        ## diminish the set size only after self.nsigma iterations 
        if icount%self.nsigma==0:
          self.sigma-=self.sigma**2/dscale
        icount+=1
        nblock+=1

      self.iter+=1

      ## print('nblock:',nblock)

      prevlambda=self.xlambda

      sys.stdout.flush()
          
    fpred=self.function_value(Xtrain,self.rankcount, \
                                ifunc=None,irange=1)
    ## self.store_fpred[:,:,irank]=fpred

    ## volume of xP[irank] to check the linear dependency and orthogonality
    ## vol=self.volume(self.nrank-1)
    ## print('Volume:',vol)

    ## deflated output
    self.Ytrain-=fpred

    print('icount:',icount)

    self.ifirstrun=0   ## first run is finished

    return

  ## ---------------------------------------------
  ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  ## If Xtest, Ytrain are sparse they eed to converted into dense 2darray
  ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  def predict(self,Xtest,Ytrain=None,itest=1,nselect=1):
    """
    Task: to compute the predictions
    Input: Xtest      2d arrays of test input data
           Ytrain     2d array of training outputs
                      it is not None then the prediction is chosen as the best
                      training output vector
           itest      test mode =0 take the best training output
                      =1  set the "nselect" largest predicted output components
                          to 1 and the others to 0 
    """

    ## direct prediction, regression 
    if Ytrain is None:
      m,ndimx=Xtest.shape
      Ypred=np.zeros((m,self.ndimy))
      for irank in range(self.rankcount+1):
        nminrank,nrank=self.lranks[irank]
        Ypred+=self.function_value(Xtest,irank,xU=self.xU, \
                                   xV=self.xV[:,nminrank:nrank], \
                                   xQ=self.xQ[nminrank:nrank], \
                                   xlambda=self.xlambda[nminrank:nrank], \
                                   xlambdaU=self.xlambdaU, \
                                   bias=self.pbias[irank], \
                                   ifunc=self.iactfunc,irange=0)

      if self.iyscale==1:
        Ypred=Ypred*self.yscale

      ## if self.ndimy==1:
      ##   Ypred=Ypred.ravel()
        
    else:
      ## prediction based on the most similar training example
      m,n=Xtest.shape
      Ypred0=np.zeros((m,self.ndimy))
      for irank in range(self.rankcount+1):
        nminrank,nrank=self.lranks[irank]
        Ypred0+=self.function_value(Xtest,irank,xU=self.xU, \
                                   xV=self.xV[:,nminrank:nrank], \
                                   xQ=self.xQ[nminrank:nrank], \
                                   xlambda=self.xlambda[nminrank:nrank], \
                                   xlambdaU=self.xlambdaU, \
                                   bias=self.pbias[irank], \
                                   ifunc=self.iactfunc,irange=0)
          
      if itest==0:
        if Ypred0.shape[1]>1:
          ynorm1=np.sqrt(np.sum(Ytrain**2,1))
          ynorm1+=1*(ynorm1==0)
          ynorm2=np.sqrt(np.sum(Ypred0**2,1))
          ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
          zscore=np.dot(Ytrain,Ypred0.T)
          ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
          zscore/=np.outer(ynorm1,ynorm2)
          iscore=np.argmax(zscore,0)    ## blocks might be random chosen 
          Ypred=Ytrain[iscore]
             
      elif itest==1:

        if Ypred0.shape[1]>1:
          xmax=np.max(Ypred0,1)
          Ypred=2*(Ypred0>=np.outer(xmax,np.ones(self.ndimy)))-1
        else:  ## ndimy==1
          Ypred=np.sign(Ypred0)
      
    return(Ypred)

  ## ---------------------------------------------------
  def save_parameters(self,filename):
    """
    Task: to save the parameters learned
    Input:   filename   filename of the file used to store the parameters
    """

    savedict={}
    for var in self.lsave:
      value=self.__dict__[var]
      savedict[var]=value

    fout=open(filename,'wb')
    pickle.dump(savedict,fout)
    
    return

  ## ---------------------------------------------------
  def load_parameters(self,filename):
    """
    Task: to load the parameters learned and saved earlier
    Input:    filename   filename of the file used to store the parameters
    Modifies: (self) norder,nordery,nrank,iyscale,yscale,ldim,ndimy
                     xP,xQ,xlambda,pbias
    """

    fin=open(filename,'rb')
    savedict=pickle.load(fin)

    for var in self.lsave:
      if var in self.__dict__:
        self.__dict__[var]=savedict[var]
      else:
        print('Missing object attribute:',var)

    return
    
## ###################################################
## ################################################################
## if __name__ == "__main__":
##   if len(sys.argv)==1:
##     iworkmode=0
##   elif len(sys.argv)>=2:
##     iworkmode=eval(sys.argv[1])
##   main(iworkmode)
