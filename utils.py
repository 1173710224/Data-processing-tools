'''
author:cbz
'''
'''
pandas 基本操作
'''
file = ''
import pandas as pd
df = pd.read_csv(file)
indexs = []
df.drop(index=indexs,inplace=True)
columns = []
df.drop(columns=columns,axis=1,inplace=True)
df.to_csv(file,index=False)
'''
自定义函数
'''
def is_number(s):
    '''
    判断s是不是数字
    :param s:
    :return:
    '''
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def badrate(lis):
    '''
    缺失数据率
    :param lis:
    :return:
    '''
    num = len(lis)
    badnum = 0
    for i in range(num):
        if is_number(lis[i][0]) == False:
            badnum += 1
            continue
        if math.isnan(float(lis[i][0])):
            badnum += 1
            continue
    return float(badnum / num)

def canuse(x):
    '''
    判断数据是否可用
    :param x:
    :return:
    '''
    if is_number(x) == False:
        return False
    if math.isnan(float(x)):
        return False
    return True

def normalization(data):
    '''
    对dataframe进行标准化
    :param data:
    :return:
    '''
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def pca():
    # 降维
    estimator = PCA(n_components='mle')
    # print(df2)
    data_pca = estimator.fit_transform(df.values)
    print('pca结果')
    print(estimator.explained_variance_ratio_)
    print('特征值')
    print(estimator.singular_values_)
    print('转换矩阵')
    print(estimator.components_)
    # 线性组合系数
    N = estimator.singular_values_[0:2]
    a = estimator.components_[0:2]

def lars():
    model = LassoLarsCV(normalize=True, max_iter=100000)
    model.fit(X, Y)
    print(len(columns))
    print(model.coef_)
    print(model.coef_path_)

def get_lr_stats(x, y, model):
    '''
    计算拟合优度
    :param x:
    :param y:
    :param model:
    :return:
    '''
    rss = sum([(tmp[0])**2 for tmp in y - model.predict(x)])
    mean = np.mean(y)
    tss = sum([(tmp[0] - mean)**2 for tmp in y])
    print(rss,tss)
    return 1 - tss / rss

def logistic():
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X, Y.astype('int'))
    print(model.score(X, Y.astype('int')))
    print(model.coef_)
    print(model.intercept_)

import numpy as np
# 计算皮尔逊相关系数
def pearson(x,y):
    pccs = np.corrcoef(x, y)
    return pccs
# 计算Kendall's Tau相关系数
# 非正态适用，ranking适用
from scipy.stats import kendalltau
def kendall(x,y):
    Kendallta2, p_value = kendalltau(x,y)
    return Kendallta2
# 计算spearmanr相关系数
# 适用条件同kendall's tau
from scipy.stats import spearmanr
def spearman(x,y):
    coef,p = spearmanr(x, y)
    return coef
# 偏相关系数
# 由于相关系数的不可传递性，在进行相关分析时，
# 往往要控制对xx、yy同时产生作用的变量pp的影响。
# 剔除其他变量影响之后再进行相关分析
from scipy import stats
def partial_corr(x, y, partial):
    xy, xyp = stats.pearsonr(x, y)
    xp, xpp = stats.pearsonr(x, partial)
    yp, ypp = stats.pearsonr(y, partial)
    n = len(x)
    df = n - 3
    r = (xy - xp * yp) / (np.sqrt(1 - xp * xp) * np.sqrt(1 - yp * yp))
    if abs(r) == 1.0:
        prob = 0.0 # 犯错误概率
    else:
        t = (r * np.sqrt(df)) / np.sqrt(1 - r * r)
        prob = (1 - stats.t.cdf(abs(t), df)) ** 2 # 犯错误概率
    return r
# 灰色关联分析
# 分析多个因素对结果的影响程度
import pandas as pd
def gra(x):
    '''
    :param x:df类型的,并且要将目标函数列放在第一列
    :return:
    '''
    # print(x)
    x=x.iloc[:,:].T
    # print(x)
    x_mean=x.mean(axis=1)
    for i in range(x.index.size):
        x.iloc[i,:] = x.iloc[i,:]/x_mean[i]
    print(x)
    ck=x.iloc[0,:]
    cp=x.iloc[1:,:]
    print(ck)
    print(cp)
    print()
    t=pd.DataFrame()
    for j in range(cp.index.size):
        temp=pd.Series(cp.iloc[j,:]-ck)
        t=t.append(temp,ignore_index=True)
    print(t)
    mmax=t.abs().max().max()
    mmin=t.abs().min().min()
    rho=0.5
    ksi=((mmin+rho*mmax)/(abs(t)+rho*mmax))
    r=ksi.sum(axis=1)/ksi.columns.size
    return r
