# -*- coding: utf-8 -*-

import traceback
import adodbapi
import re
import xlsxwriter
import random
import numpy as np
from sklearn.metrics import cohen_kappa_score
import pandas as pd
import xlrd
from sklearn import svm, metrics
from sklearn.model_selection import KFold
from lifelines.utils import concordance_index as cc_index


def ReadVTT(vttfile):
    metas,snippets,seglabels,kws = [],[],[],[]
    markUpStart = -1
    n = -1
    with open(vttfile,encoding='utf-8') as f:
        lines = [line for line in f]
    nLines = len(lines)
    textStart, textEnd, markUpStart = -1, -1, -1
    sepLines,labels = [],[]
    pos = 0
    for n in range(nLines):
        if markUpStart<0:
            for c in lines[n]:
                if ord(c)>=128:
                    print('Line',n,'has non-ascii characters')
                    print(repr(lines[n]))
                    break
        if textStart>0 and textEnd<0 and lines[n].startswith('#<-----'):
            textEnd = n
            sepLines.append(n+1)
        if textStart==0 and lines[n].startswith('#<-----'):
            textStart = n+1
            sepLines.append(n)
        if textStart<0 and lines[n].startswith('#<Text Content>'):
            textStart = 0
        if textStart>0 and textEnd<0 and lines[n]=='-'*82+'\n':
            sepLines.append(n)
        if markUpStart>0:
            items = lines[n].split('|')
            if items[2].startswith('SnippetColumn'):
                subitems = items[4].split('"')
                snippetNumber = int(subitems[1])
                if len(metas)==snippetNumber-1:
                    metas.append({})
                if len(seglabels)==snippetNumber-1:
                    seglabels.append([])
                columnName = subitems[-2]
                columnValue = items[5].rstrip()
                if len(columnValue)>50:
                    columnValue = columnValue[:50]+'...'
                metas[snippetNumber-1][columnName] = columnValue
            else:
                label = items[2].strip()
                start = int(items[0])
                end = start+int(items[1])
                labels.append((label,start,end))
        if markUpStart==0 and lines[n].startswith('#<-----'):
            markUpStart = n+1
        if markUpStart<0 and lines[n].startswith('#<MarkUps Information>'):
            markUpStart = 0

    pos = 0
    j = 0
    for i in range(len(sepLines)-1):
        kw = lines[sepLines[i]+3][9:].strip()
        kws.append(kw)
        snippet = ''.join(lines[sepLines[i]+6:sepLines[i+1]-1])
        pos += len(''.join(lines[sepLines[i]+1:sepLines[i]+6]))
        snippets.append(snippet)
        sniplen = len(snippet)

        while j<len(labels) and labels[j][1]<=pos+sniplen:
            if pos<=labels[j][1]:
                #print(i,pos,pos+sniplen,j,labels[j][1])
                seglabels[i].append((labels[j][1]-pos,labels[j][2]-pos,labels[j][0]))
            j += 1
        pos += len(''.join(lines[sepLines[i]+6:sepLines[i+1]+1]))
    print(len(labels)-len(metas))
    return metas,snippets,seglabels,kws

if __name__ == '__main__':
    mainDir = r'P:\**************\*****'
    connectors = (
    'Provider=SQLOLEDB',
    'Data Source=*******',
    'Initial Catalog=*******',
    'Integrated Security=SSPI',)

# NLP Model
    metas,snippets,seglabels,kws = [],[],[],[]
    vttfiles = [
        mainDir+r'\Snippets_450_SP_New2.vtt',
        mainDir+r\Snippets_450_SP_4_Revision.vtt',
        mainDir+r\Snippets_100_Common_Done.vtt'
    ]
    for vttfile in vttfiles:
        metas_,snippets_,seglabels_,kws_ = ReadVTT(vttfile)
        metas.extend(metas_)
        snippets.extend(snippets_)
        seglabels.extend(seglabels_)
        kws.extend(kws_)

    corrections = [{},{}]
    book = xlrd.open_workbook(mainDir+r'\Annotation\Consensus -aa sp.xlsx')
    for n in range(2):
        sheet = book.sheet_by_index(n)
        #print(sheet.name)
        for i in range(4,sheet.nrows):
            cells = sheet.row(i)
            #print(i,cells[0].value)
            corrections[n][int(cells[0].value)] = cells[3].value

    labels = ['Keyword',
    'Yes',
    'Uncertain',
    'No']
    label2id = {label[0]:i for i,label in enumerate(labels)}
    classes,kwstarts,kwends = [],[],[]
    for i in range(0,len(seglabels)):
        label = ''
        for start,end,seglabel in seglabels[i]:
            if seglabel=='Relevant Context':
                continue
            label = seglabel
            kw = snippets[i][start:end].lower()
            break
        if label=='Keyword':
            print('1 Keyword',i+1)
        if label=='':
            print('1 Missing',i+1)
            classes.append(-1)
            kwstarts.append(-1)
            kwends.append(-1)
        else:
            if i<450 and i+1 in corrections[0]:
                label = corrections[0][i+1]
            elif i<900 and i+1-450 in corrections[1]:
                label = corrections[1][i+1-450]
            classes.append(label2id[label[0]])
            kwstarts.append(start)
            kwends.append(end)

    snippet_idxs = []
    kwdoc_data = []
    kwdoc_cls = []
    wordBef2df = {}
    wordAft2df = {}
    kw2df = {}
    rawkw2df = {}
    for i in range(len(snippets)):
        cls,kwstart,kwend = classes[i],kwstarts[i],kwends[i]
        if cls==-1:
            continue
        snippet_idxs.append(i)
        kwdoc_cls.append(3-cls)
        kw = re.sub(r'\s+',' ',kws[i].lower())
        kw2df[kw] = kw2df.get(kw,0)+1
        snippet = snippets[i]
        rawkw = snippet[kwstart:kwend]
        rawkw2df[rawkw] = rawkw2df.get(rawkw,0)+1
        rawshape = 1
        if rawkw==rawkw.lower():
            rawshape = 0
        elif rawkw==rawkw.upper():
            rawshape = 2
        textBef = snippet[:kwstart+1].lower()
        textBef = re.sub(r'\s+',' ',textBef)
        textBef = re.sub(r'\d+',' ',textBef)
        textBef = re.sub('[^a-z]',' ',textBef)
        wordsBef = textBef.split(' ')[:-1]
        wordBef2wt = {}
        words = [kw]+wordsBef[::-1]
        for i in range(1,len(words)):
            if i>30:
                break
            w = words[i]
            if w=='':
                continue
            wordBef2wt[w] = max(wordBef2wt.get(w,0),(31-i)/30.0)
            if words[i-1]=='':
                continue
            ww = w+'_'+words[i-1]
            wordBef2wt[ww] = max(wordBef2wt.get(ww,0),(31-i)/30.0)*1.2
            if i<2:
                continue
            if words[i-2]=='':
                continue
            www = ww+'_'+words[i-2]
            wordBef2wt[www] = max(wordBef2wt.get(www,0),(31-i)/30.0)*1.5
        textAft = snippet[kwend-1:].lower()
        textAft = re.sub('\s+',' ',textAft)
        textAft = re.sub('\d+',' ',textAft)
        textAft = re.sub('[^a-z]',' ',textAft)
        wordsAft = textAft.split(' ')[1:]
        wordAft2wt = {}
        words = [kw]+wordsAft
        for i in range(1,len(words)):
            if i>30:
                break
            w = words[i]
            if w=='':
                continue
            wordAft2wt[w] = max(wordAft2wt.get(w,0),(31-i)/30.0)
            if words[i-1]=='':
                continue
            ww = words[i-1]+'_'+w
            wordAft2wt[ww] = max(wordAft2wt.get(ww,0),(31-i)/30.0)*1.2
            if i<2:
                continue
            if words[i-2]=='':
                continue
            www = words[i-2]+'_'+ww
            wordAft2wt[www] = max(wordAft2wt.get(www,0),(31-i)/30.0)*1.5
        for w in wordBef2wt:
            wordBef2df[w] = wordBef2df.get(w,0)+1
        for w in wordAft2wt:
            wordAft2df[w] = wordAft2df.get(w,0)+1
        kwdoc_data.append((kw,rawshape,wordBef2wt,wordAft2wt))

    s = 8
    random.seed(s)
    idxs = list(range(len(kwdoc_cls)))
    model = svm.LinearSVC(C=0.03)
    c_div = [1.2,1.2]
    k = 0
    Y_tests,Y_scores,c_stats,Y_preds,accuracies = [],[],[],[],[]
    ten_folds = KFold(n_splits=10)

    for train,test in ten_folds.split(kwdoc_data):
#    if True:
#        train = test = range(len(idxs))

        k += 1
        featBefSet,featAftSet = set(),set()
        for cut in range(2):
            kwdoc_bcls = (np.array(kwdoc_cls)>cut).astype(int)
            wordBef2df,wordAft2df = {},{}
            c_wordBef2df,c_wordAft2df = [{},{}],[{},{}]
            c_count = [0,0]
            kw2df = {}
            for n in train:
                i = idxs[n]
                c = kwdoc_bcls[i]
                c_count[c] += 1
                kw = kwdoc_data[i][0]
                kw2df[kw] = kw2df.get(kw,0)+1
                for w in kwdoc_data[i][2]:
                    wordBef2df[w] = wordBef2df.get(w,0)+1
                    c_wordBef2df[c][w] = c_wordBef2df[c].get(w,0)+1
                for w in kwdoc_data[i][3]:
                    wordAft2df[w] = wordAft2df.get(w,0)+1
                    c_wordAft2df[c][w] = c_wordAft2df[c].get(w,0)+1
            keywords = sorted([kw for kw in kw2df if kw2df[kw]>=2])
            kw2idx = {w:i for i,w in enumerate(keywords)}
            c_ratio = [c_count[c]*1.0/len(train) for c in (0,1)]
            c_highBef,c_highAft = [{},{}],[{},{}]
            for w in wordBef2df:
                if wordBef2df[w]<5*sum(c_count)/1000.0:
                    continue
                for c in (0,1):
                    r = c_wordBef2df[c].get(w,0)*1.0/wordBef2df[w]
                    if (1-r)*c_div[c]<1-c_ratio[c]:
                        c_highBef[c][w] = r
            featBefSet |= set(c_highBef[0])|set(c_highBef[1])
            for w in wordAft2df:
                if wordAft2df[w]<5*sum(c_count)/1000.0:
                    continue
                for c in (0,1):
                    r = c_wordAft2df[c].get(w,0)*1.0/wordAft2df[w]
                    if (1-r)*c_div[c]<1-c_ratio[c]:
                        c_highAft[c][w] = r
            featAftSet |= set(c_highAft[0])|set(c_highAft[1])
        featuresBef = sorted(featBefSet)
        featuresAft = sorted(featAftSet)
        featBef2idx = {w:i for i,w in enumerate(featuresBef)}
        featAft2idx = {w:i for i,w in enumerate(featuresAft)}
        X_train = np.zeros((len(train),3+len(keywords)+len(featuresBef)+len(featuresAft)))
        Y_train = np.zeros(len(train),dtype=int)
        for n,n_train in enumerate(train):
            i = idxs[n_train]
            Y_train[n] = kwdoc_cls[i]
            kw,rawshape,wordBef2wt,wordAft2wt = kwdoc_data[i]
            X_train[n,rawshape] = 1
            if kw in kw2idx:
                X_train[n,3+kw2idx[kw]] = 1
            for w in wordBef2wt:
                if not w in featBef2idx:
                    continue
                j = 3+len(keywords)+featBef2idx[w]
                X_train[n,j] = wordBef2wt[w]
            for w in wordAft2wt:
                if not w in featAft2idx:
                    continue
                j = 3+len(keywords)+len(featuresBef)+featAft2idx[w]
                X_train[n,j] = wordAft2wt[w]
        X_test = np.zeros((len(test),3+len(keywords)+len(featuresBef)+len(featuresAft)))
        Y_test = np.zeros(len(test),dtype=int)
        for n,n_test in enumerate(test):
            i = idxs[n_test]
            Y_test[n] = kwdoc_cls[i]
            kw,rawshape,wordBef2wt,wordAft2wt = kwdoc_data[i]
            X_test[n,rawshape] = 1
            if kw in kw2idx:
                X_test[n,3+kw2idx[kw]] = 1
            for w in wordBef2wt:
                if not w in featBef2idx:
                    continue
                j = 3+len(keywords)+featBef2idx[w]
                X_test[n,j] = wordBef2wt[w]
            for w in wordAft2wt:
                if not w in featAft2idx:
                    continue
                j = 3+len(keywords)+len(featuresBef)+featAft2idx[w]
                X_test[n,j] = wordAft2wt[w]
        X_train2 = np.concatenate([np.concatenate([X_train,X_train],axis=0),
                np.concatenate([np.zeros((len(train),1))+10,np.zeros((len(train),1))],axis=0)],axis=1)
        Y_train2 = np.concatenate([Y_train>0,Y_train>1],axis=0).astype(int)
        X_test2 = np.concatenate([X_test,np.zeros((len(test),1))+5],axis=1)
        model.fit(X_train2,Y_train2)

        Y_score2_tr = model.decision_function(X_train2)
        Y_score = model.decision_function(X_test2)
        c_stat = cc_index(Y_test,Y_score)
        c_stats.append(c_stat)
        Y_score_tr = (Y_score2_tr[:len(train)]+Y_score2_tr[len(train):])/2

        fprs,tprs,thresholds = metrics.roc_curve(Y_test>1,Y_score)
        auc = metrics.auc(fprs,tprs)
        npos,nneg = (Y_train>1).sum(),(Y_train<=1).sum()
        fprs,tprs,thresholds = metrics.roc_curve(Y_train>1,Y_score_tr)
        accs = (tprs*npos+(1-fprs)*nneg)/(npos+nneg)
        idxmax = accs.argmax()
        threshold1 = 0.5*(thresholds[idxmax]+thresholds[idxmax+1])

        fprs,tprs,thresholds = metrics.roc_curve(Y_test>0,Y_score)
        auc = metrics.auc(fprs,tprs)
        npos,nneg = (Y_train>0).sum(),(Y_train<=0).sum()
        fprs,tprs,thresholds = metrics.roc_curve(Y_train>0,Y_score_tr)
        accs = (tprs*npos+(1-fprs)*nneg)/(npos+nneg)
        idxmax = accs.argmax()
        threshold2 = 0.5*(thresholds[idxmax]+thresholds[idxmax+1])

        Y_pred = np.searchsorted([threshold2,threshold1],Y_score,side='right')
        acc = metrics.accuracy_score(Y_test,Y_pred)
        accuracies.append(acc)
        Y_preds.append(Y_pred)
        Y_scores.append(Y_score)
        Y_tests.append(Y_test)
    Y_score_final = np.concatenate(Y_scores)
    Y_test_final = np.concatenate(Y_tests)
    Y_pred_final = np.concatenate(Y_preds)
    c_stat = cc_index(Y_test_final,Y_score_final)
    print('C-Index: %.3f' %(c_stat*100))
    acc = metrics.accuracy_score(Y_test_final,Y_pred_final)
    print('Accuracy: %.3f' %(acc*100))



