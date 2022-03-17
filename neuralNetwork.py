"""
use neural network to fit the function
"""
import time
import torch
import torch.nn as nn
from torch.nn import Linear, Sigmoid, ReLU, Sequential
from torch.optim import Adam
import numpy as np
from matplotlib import pyplot as plt
import warnings
import copy


class NeuralNetwork(object):
    def __init__(
            self,
            mod="phi", 
            dm=2,
            dataType=torch.float32,
        ):
        net1 = Sequential(
            Linear(dm, 4, bias=True),  # 第1层有4个神经元, 相当于用4根直线来划分区域，做一个分类器
            # ReLU(),  # 用 ReLU, 多条直线分割的形状会更明显，但对随机初值的依赖性大，需要多次拟合
            Sigmoid(),  # 用 Sigmoid, 拟合效果更smooth，对初值依赖性小
            Linear(4, 1, bias=True),  # 第2层有1个神经元
            Sigmoid(),
        )
        net2 = Sequential(
            Linear(dm, 4, bias=True),  # 第1层有4个神经元, 相当于用4根直线来划分区域，做一个分类器
            # ReLU(),  # 用 ReLU, 多条直线分割的形状会更明显，但对随机初值的依赖性大，需要多次拟合
            Sigmoid(),  # 用 Sigmoid, 拟合效果更smooth，对初值依赖性小
            Linear(4, 1, bias=True),  # 第2层有1个神经元
        )
        nets = {"phi": net1, "stress": net2}
        loss_modes = {
            "phi": "BCELoss", 
            "stress": "MSELoss"
        }
        self.mod = mod
        self.net = nets[mod]
        self.loss_mode = loss_modes[mod]
        self.dataType = dataType
        self.dimension = dm


    def getFitting(  # anagolus to the learning process by gradient decent method
            self,
            x, val, 
            lr=0.01,  # learning rate
            loopMax=2000,  # maximum iteration number for gradient optimizer
            betas=[0.8, 0.9],
            printData=True, plotData=True, alwaysPlotData=True,
            region_cen=[0., 0., 0.],
            loss_tolerance=float("inf"),
            initialWei=None,  # initial weights
            innerLoops=2,  # whether fit multiple times in the inner loop and choose a net with lowest loss
            frozenClassifier=False,  
                ### frozenClassifier, whether freeze the classier (possiotion and direction), 
                ### so that the wieghts and bias at hidden layer remian unchanged 
            prematurelyBreak=False,  # whether prematurely break when the parameters nearly unchanged at a step
            refitTimes=10,  # refit by changing the initial values of weights
        ):
        """
        fit by neural network (analogus to the learning process)
        """
        dm = self.dimension
        dataType = self.dataType
        net = self.net
        tolerance = 1.e-8
        self.alwaysPlotData=True
        startTime = time.time()

        ### if initial weights unchanges, (given specific initial weights instead of random initial weights) 
        ### so result unchanged no matter how many times you fit
        if initialWei != None:  
            innerLoops = 1
            refitTimes = 1

        ### --------------------------------------------------------------------------------data clean
        """
        # x is the coordinates of many points
        # x should has size of [np, dm]
        #   where np is number of points,
        #   and dm is dimension of coordinates
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=dataType)
        if printData:
            print('x.size() =', x.size())
        if len(x.size()) == 2:
            if x.size()[1] > dm:
                x = x[:, :dm]
        else:
            warnings.warn('dimension (of input x) = ' + str(len(x.size())))
            if dm == 1:
                x__ = []
                for i in range(len(x)):
                    x__.append([x[i]])
                x = np.mat(x__)
                x = torch.tensor(x)
        print('x.size() =', x.size())

        # val is the field values at corresponding points
        if not torch.is_tensor(val):
            val = torch.tensor(val, dtype=dataType)
        # val = val.reshape([val.numel()])

        if x.size()[0] != len(val):
            raise ValueError('the number of points not consistent in coordinates and field values')
        ### --------------------------------------------------------------------------------end of data clean

        if self.loss_mode == 'BCELoss':
            criterion = torch.nn.BCELoss()
        elif self.loss_mode == 'MSELoss':
            criterion = torch.nn.MSELoss()
        else:
            raise ValueError('please select a mode for the loss')    

        loss = float("inf")
        outerloop = 0
        while loss > loss_tolerance and outerloop <= (refitTimes // innerLoops):  # restart the fitting process until loss is lower than an ideal value
            outerloop += 1

            innerLoop = 1
            loss_innerloop, net_innerloop = [], []
            historyLoss_innerloop, historyLoop_innerloop = [], []
            while innerLoop <= innerLoops:
                innerLoop += 1

                historyLoss = []
                historyLoop = []

                # initialize the weight of the neural network
                # the initialization has probability character,
                # thus you need to fitting many times to get the right fitting
                if initialWei == None:
                    net.apply(weights_init)
                    params = []
                    for tmp in net.parameters():
                        params.append(tmp.data)
                    if printData:
                        print("\033[32;1m{} \033[40;33;1m{}\033[0m".format(
                            "^^^^^^^^ params =", params
                        ))
                    for i in range(len(params[1])):
                        params[1][i] = - params[0][i] @ torch.tensor(region_cen[:dm], dtype=dataType)
                    
                    if printData:
                        print("\033[32;1m{}\033[0m".format("------------------- initial values of weight is:"))
                        for tmp in net.parameters():
                            print("\033[40;33;1m{}\033[0m".format(
                                    tmp.data
                                ))
                    parameters = list(net.parameters())
                    print(
                        "\033[32;1m{} \033[40;33;1m{}\033[0m".format(
                            "for each linear neuron, ax + by + c = ", [
                                float(
                                    parameters[0][i] @ torch.tensor(region_cen[:dm], dtype=dataType) + parameters[1][i]
                                )
                                for i in range(len(parameters[1]))
                            ]
                        )
                    )
                else:  # use input weight as initial weight
                    for i, param in enumerate(list(net.parameters())):
                        param.data[:] = initialWei[i][:]

                if not frozenClassifier:
                    optimizer = Adam(net.parameters(), lr=lr, betas=betas)
                else: 
                    ### fix the classifier direction, fitting the output layer
                    optimizer = Adam(list(net.parameters())[2:], lr=lr, betas=betas)  
                optimizer.zero_grad()

                less_than_tolerance = False
                for step in range(loopMax):
                    if step:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        """ whether satisfy the convergence condition """
                        paramNew = list(net.parameters())
                        paramNew = [paramNew[_].data for _ in range(len(paramNew))]
                        sqSum = square_sum_of_net(paramNew)
                        if prematurelyBreak:
                            if square_sum_of_net(paramNew, paramOld) / (sqSum + 1.e-8) < tolerance:
                                print('break, step =', step)
                                less_than_tolerance = True
                                historyLoss.append(float(loss))
                                historyLoop.append(step)
                                break
                            # if loss < tolerance:
                            #     print('break, step =', step)
                            #     less_than_tolerance = True
                            #     historyLoss.append(float(loss))
                            #     historyLoop.append(step)
                            #     break

                    pred = net(x)
                    loss = criterion(pred[:, 0], val)
                    
                    if prematurelyBreak:
                        paramOld = list(net.parameters())
                        paramOld = [paramOld[_].data for _ in range(len(paramOld))]
                        paramOld = copy.deepcopy(paramOld)

                    if step % 100 == 0:
                        historyLoss.append(float(loss))
                        historyLoop.append(step)
                        # np.set_printoptions(suppress=True)
                        if printData:
                            print('第{}步：loss = {:g}, parameters = \n{}'.format(step, loss, list(net.parameters())))               

                if less_than_tolerance == False:
                    historyLoop.append(loopMax)
                    historyLoss.append(float(loss))

                loss_innerloop.append(float(loss))
                net_innerloop.append(copy.deepcopy(net))
                historyLoop_innerloop.append(historyLoop)
                historyLoss_innerloop.append(historyLoss)
                print("\033[35;1m{} \033[40;33;1m{}\033[0m".format("loss =", loss))
            
            ### choose the net with lowest loss
            loss = min(loss_innerloop)
            minloos_id = loss_innerloop.index(loss)
            historyLoss = historyLoss_innerloop[minloos_id]
            historyLoop = historyLoop_innerloop[minloos_id]
            net = net_innerloop[minloos_id]
            self.net = net
        print('\033[35;1m^^^^^^^^^^^ finally, loss = \033[40;33;1m{:g}\033[0m'.format(
            loss**0.5 if self.loss_mode == 'MSELoss' else loss
        ))
        self.finalLoss = loss
        if printData:
            print("parameters = \n{}".format(list(net.parameters())))

        endTime = time.time()
        print('consuming time for fitting is', endTime - startTime, 's')

        if (plotData or (loss > loss_tolerance)) and alwaysPlotData:
            params = []
            for param in net.parameters():
                params.append(param.data)
            # plot the history loss versus history loop
            plt.figure()
            plt.plot(
                historyLoop, 
                np.array(historyLoss)**0.5 if self.loss_mode == 'MSELoss' else historyLoss, 
            )
            plt.title(self.loss_mode + ' vs steps', fontsize=20.)
            plt.xlabel('steps', fontsize=20.)
            plt.ylabel('loss', fontsize=20.)
            plt.xticks(fontsize=15.)
            plt.yticks(fontsize=20.)
            plt.tight_layout()
            plt.pause(1.)

            # plot the linear classifier of each neuron
            if x.size()[1] > 1:
                heightY = (x[:, 1].max() - x[:, 1].min()) / (x[:, 0].max() - x[:, 0].min())
                plt.figure(figsize=(7, 7 * heightY))
                x_ = torch.arange(x[:, 0].min(), x[:, 0].max(), 0.1)
                for i in range(len(params[0])):
                    y_ = (-params[1][i] - params[0][i, 0] * x_) / params[0][i, 1]
                    plt.plot(x_, y_)
                plt.xlim((x[:, 0].min(), x[:, 0].max()))
                plt.ylim((x[:, 1].min(), x[:, 1].max()))
                plt.xticks(fontsize=20.)
                plt.yticks(fontsize=20.)
                plt.tight_layout()
                plt.pause(1.)

        return net

    
    def reasoning(self, x):
        """
            the reasoning of neural network
            i.e., the function value of the net
        """
        x = list(x[:self.dimension])  # unify the dimension
        x = torch.tensor(x, dtype=self.dataType)  # unify the dataType
        return self.net(x)
    

    def getGradient(self, x):
        x = list(x[:self.dimension])  # unify the dimension
        x = torch.tensor(x, dtype=self.dataType, requires_grad=True)  # unify the dataType
        f = self.net(x)
        f.backward()
        return x.grad


# initialize the weight of the neural network
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.xavier_normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=1.)
        # nn.init.xavier_normal_(m.bias, gain=100.)
        # nn.init.normal_(m.weight, mean=100., std=10.)


def square_sum_of_net(*params):
    if len(params) == 1:
        params = params[0]
        s = 0
        for a in params:
            s += (a**2).sum()
        return s
    elif len(params) == 2:
        s = 0
        for i in range(len(params[0])):
            s += ((params[0][i] - params[1][i]) ** 2).sum()
        return s
    else:
        raise ValueError('at least two params should be input to square sum')


def functionValue(
        w, x,
        dataType=torch.float32,
    ):
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=dataType)
    f = w(x)  # w is equivalent to the net function
    return float(f)


if __name__ == "__main__":

    dataPack = {"vals": [], "coors": []}
    fileName = input("\033[35;1m{}\033[0m".format(
        "please input the data file name: "
    ))
    with open("data/{}.txt".format(fileName), "r") as file:
        i = 0
        for line in file:
            if i:
                line = line.strip().split()
                data = list(map(float, line))
                dataPack["vals"].append(data[0])
                dataPack["coors"].append(data[1:])
            i += 1
    net = NeuralNetwork()
    net.getFitting(dataPack["coors"], dataPack["vals"])


