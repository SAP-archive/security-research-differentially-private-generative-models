# SPDX-FileCopyrightText: 2020 SAP SE
#
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import numpy as np

class discretedata_manager(object):
    def __init__(self, data):
        self.createOHdataset(data)
        self.dictLables = self.extractDictLables()
        df_num = self.OHdataset.select_dtypes(include=[np.number])
        df_norm = (df_num - df_num.min()) / ((df_num.max() - df_num.min()))
        self.OHdataset[df_norm.columns] = df_norm

    def extractDictLables(self):
        dictLables = {}
        column = ''
        for var in self.OHdataset.columns:
            if ('_' not in var):
                dictLables[var] = [[max(self.OHdataset[var]),min(self.OHdataset[var]),np.mean(self.OHdataset[var])]]
            else:
                if(column != var.split('_')[0]):
                    dictLables[var.split('_')[0]] = [var.split('_')[1]]
                    column = var.split('_')[0]
                else:
                    dictLables[column].append(var.split('_')[1])
        self.variables = list(dictLables.keys())
        return dictLables

    def mapOHlabels(self,ohlabels,labels):
        return [labels[i] for i in ohlabels]

    def getLabelsLength(self):
        labelsLength = []
        for key in self.dictLables:
            labelsLength.append(len(self.dictLables[key]))
        return labelsLength

    def getLabelLengthTot(self):
        totLength = 0
        for key in self.dictLables:
            totLength += (len(self.dictLables[key]))
        return totLength

    def createOHdataset(self,data):
        self.OHdataset = pd.get_dummies(data)
        #TODO add in case of numerical variables
#         cols = self.OHdataset.columns.tolist()
#         temp = cols[0]
#         cols = cols[1:]
#         cols.append(temp)
#         self.OHdataset = self.OHdataset[cols]

    def convertDiscreteMatrix(self,data):
        newData = np.zeros(data.shape)
        newTextDataset = []
        lengths = self.getLabelsLength()
        totIndex = 0
        for index, length in enumerate(lengths):
            samplesDisc = []
            textData = []
            for i in range(data.shape[0]):
                if(length>1):
                    p = np.array(data[i, totIndex:totIndex + length])
                    p /= p.sum()
                    sampled_value = np.random.choice(length, 1, p=p)
                    samplesDisc.append(sampled_value)
                    newData[i, samplesDisc[i] + totIndex] = 1
                    textData.append(self.dictLables[self.variables[index]][sampled_value[0]])
                else:
                    sampled_value = data[i, totIndex]
                    norm_values = self.dictLables[self.variables[index]][0]
                    denormalized_value = sampled_value  * ((norm_values[0] - norm_values[1]) ) + norm_values[1]
                    samplesDisc.append(denormalized_value)
                    newData[i, totIndex] = samplesDisc[i]
                    textData.append(int(denormalized_value)) 
            newTextDataset.append(textData)
            totIndex += length
        return newData, newTextDataset

      