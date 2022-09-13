# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 17:23:00 2022

@author: VHAWASShaoY
"""

import traceback
import adodbapi
import numpy as np
#import statsmodels.stats.outliers_influence as smoi
#import statsmodels.api as sm
#from statsmodels.api import Logit
from sklearn.svm import LinearSVC
from sklearn import metrics
import matplotlib.pyplot as plt



if __name__ == '__main__':
    mainDir = r'P:\ORD_Ahmed_201906060D\Yijun'
    connectors = (
    'Provider=SQLOLEDB',
    'Data Source=vhacdwrb03.vha.med.va.gov',
    'Initial Catalog=ORD_Ahmed_201906060D',
    'Integrated Security=SSPI',)

    columns = []
    sql = '''
select COLUMN_NAME
from INFORMATION_SCHEMA.COLUMNS
where TABLE_NAME = 'CKD_cohort_covariates'
'''
    print('Connecting to database ...', end=' ')
    conn = adodbapi.connect(';'.join(connectors))
    db = conn.cursor()
    print('done')
    try:
        db.execute(sql)
        n = 0
        for row in db:
            n += 1
            columns.append(row[0])
        print()
    except Exception:
        traceback.print_exc()
    finally:
        conn.close()

    varnames = ['inpat']+columns[2:-13]+[columns[-1]]+columns[-13:-1]

#Dxs,LabValues,VitalValues
    icns = []
    Y = np.zeros(20000,dtype=int)
    data = np.zeros((20000,len(varnames)))
    sql = '''
select a.*, c.PatientSetting
from Dflt.CKD_cohort_covariates as a
join Dflt.CKD_cohort_RowID as b
on a.PatientICN = b.PatientICN
join Dflt.Project_cohort_long as c
on a.PatientICN = c.PatientICN
order by b.RowID, b.PatientICN
'''
    print('Connecting to database ...', end=' ')
    conn = adodbapi.connect(';'.join(connectors))
    db = conn.cursor()
    print('done')
    try:
        db.execute(sql)
        n = -1
        for row in db:
            breakIfRead('CKD')
            n += 1
            if (n+1)%1000==0:
                print((n+1)//1000, end=' ')
            icns.append(row['PatientICN'])
            Y[n] = int(row['Cohort']=='HF')
            data[n,0] = int(row['PatientSetting']=='Inpat')
            for k in range(1,len(varnames)):
                value = row[varnames[k]]
                data[n,k] = row[varnames[k]]
        print()
    except Exception:
        traceback.print_exc()
    finally:
        conn.close()
    icn2idx = {icn:i for i,icn in enumerate(icns)}

#Meds
    meds = []
    sql = '''
select *
from Dflt.CKD_cohort_Med_ChiSq2
where Chi2>=10
and HFCt+NonHFCt>=160
order by Chi2 desc
'''
    print('Connecting to database ...', end=' ')
    conn = adodbapi.connect(';'.join(connectors))
    db = conn.cursor()
    print('done')
    try:
        db.execute(sql)
        for row in db:
            meds.append(row[0])
        print()
    except Exception:
        traceback.print_exc()
    finally:
        conn.close()
    med2idx = {mid:i for i,mid in enumerate(meds)}

    data_med = np.zeros((20000,len(meds)))
    sql = '''
select a.*
from Dflt.CKD_cohort_Med_1MoBeAf as a
join (
	select Med
	from Dflt.CKD_cohort_Med_ChiSq2
	where Chi2>=10
	and HFCt+NonHFCt>=160
) as b
on a.DrugNameWithoutDose = b.Med
'''
    print('Connecting to database ...', end=' ')
    conn = adodbapi.connect(';'.join(connectors))
    db = conn.cursor()
    print('done')
    try:
        db.execute(sql)
        n = 0
        for row in db:
            n += 1
            if n%10000==0:
                print(n//10000, end=' ')
            icn = row['PatientICN']
            med = row['DrugNameWithoutDose']
            if not med in med2idx:
                continue
            data_med[icn2idx[icn],med2idx[med]] = 1
        print()
    except Exception:
        traceback.print_exc()
    finally:
        conn.close()

#CPTs
    cpt_ids = []
    cpts = []
    sql = '''
select *
from Dflt.CKD_cohort_CPT_ChiSq2
where Chi2>=10
and HFCt+NonHFCt>=160
order by Chi2 desc
'''
    print('Connecting to database ...', end=' ')
    conn = adodbapi.connect(';'.join(connectors))
    db = conn.cursor()
    print('done')
    try:
        db.execute(sql)
        for row in db:
            cpt_ids.append(row[0])
            cpts.append(row[1])
        print()
    except Exception:
        traceback.print_exc()
    finally:
        conn.close()
    cpt_id2idx = {cptid:i for i,cptid in enumerate(cpt_ids)}

    data_cpt = np.zeros((20000,len(cpt_ids)))
    sql = '''
select a.*
from Dflt.CKD_cohort_CPT_1MoBeAf as a
join (
	select CPTCode
	from Dflt.CKD_cohort_CPT_ChiSq2
	where Chi2>=10
	and HFCt+NonHFCt>=160
) as b
on a.CPTCode = b.CPTCode
'''
    print('Connecting to database ...', end=' ')
    conn = adodbapi.connect(';'.join(connectors))
    db = conn.cursor()
    print('done')
    try:
        db.execute(sql)
        n = 0
        for row in db:
            n += 1
            if n%10000==0:
                print(n//10000, end=' ')
            icn = row['PatientICN']
            cpt_id = row['CPTCode']
            if not cpt_id in cpt_id2idx:
                continue
            data_cpt[icn2idx[icn],cpt_id2idx[cpt_id]] = 1
        print()
    except Exception:
        traceback.print_exc()
    finally:
        conn.close()

#Labs
    lab_ids = []
    #labs = []
    sql = '''
select *
from Dflt.CKD_cohort_LabTest_ChiSq2
where Chi2>=10
and HFCt+NonHFCt>=160
order by Chi2 desc
'''
    print('Connecting to database ...', end=' ')
    conn = adodbapi.connect(';'.join(connectors))
    db = conn.cursor()
    print('done')
    try:
        db.execute(sql)
        for row in db:
            lab_ids.append(row[0])
            #labs.append(row[1])
        print()
    except Exception:
        traceback.print_exc()
    finally:
        conn.close()
    lab_id2idx = {labid:i for i,labid in enumerate(lab_ids)}

    data_lab = np.zeros((20000,len(lab_ids)))
    sql = '''
select a.*
from Dflt.CKD_cohort_LabTestOrder_1MoBeAf as a
join (
	select *
	from Dflt.CKD_cohort_LabTest_ChiSq2
	where Chi2>=10
	and HFCt+NonHFCt>=160
) as b
on a.LOINC = b.LOINC
'''
    print('Connecting to database ...', end=' ')
    conn = adodbapi.connect(';'.join(connectors))
    db = conn.cursor()
    print('done')
    try:
        db.execute(sql)
        n = 0
        for row in db:
            n += 1
            if n%10000==0:
                print(n//10000, end=' ')
            icn = row['PatientICN']
            lab_id = row['LOINC']
            if not lab_id in lab_id2idx:
                continue
            data_lab[icn2idx[icn],lab_id2idx[lab_id]] = 1
        print()
    except Exception:
        traceback.print_exc()
    finally:
        conn.close()

#Titles
    titles = []
    sql = '''
select *
from Dflt.CKD_cohort_NoteTitle_ChiSq2
where Chi2>=10
and HFCt+NonHFCt>=160
order by Chi2 desc
'''
    print('Connecting to database ...', end=' ')
    conn = adodbapi.connect(';'.join(connectors))
    db = conn.cursor()
    print('done')
    try:
        db.execute(sql)
        for row in db:
            titles.append(row[0])
        print()
    except Exception:
        traceback.print_exc()
    finally:
        conn.close()
    title2idx = {tid:i for i,tid in enumerate(titles)}

    data_title = np.zeros((20000,len(titles)))
    sql = '''
select a.*
from Dflt.CKD_cohort_TIU3_NoteTitle as a
join (
	select NoteTitle
	from Dflt.CKD_cohort_NoteTitle_ChiSq2
	where Chi2>=10
	and HFCt+NonHFCt>=160
) as b
on a.TIUStandardTitle = b.NoteTitle
'''
    print('Connecting to database ...', end=' ')
    conn = adodbapi.connect(';'.join(connectors))
    db = conn.cursor()
    print('done')
    try:
        db.execute(sql)
        n = 0
        for row in db:
            n += 1
            if n%10000==0:
                print(n//10000, end=' ')
            icn = row['PatientICN']
            title = row['TIUStandardTitle']
            if not title in title2idx:
                continue
            data_title[icn2idx[icn],title2idx[title]] = 1
        print()
    except Exception:
        traceback.print_exc()
    finally:
        conn.close()

#Topics
    topic_ids = []
    topics = []
    sql = '''
select a.TripleIdx, b.Top5Words
from Dflt.CKD_TIU3_TopicIdx_ChiSq as a
join Dflt.CKD_TIU2_Dim_StableTopic as b
on a.TripleIdx = b.TripleIdx
where Chi2>10
and HFCt+NonHFCt>160
order by Chi2 desc
'''
    print('Connecting to database ...', end=' ')
    conn = adodbapi.connect(';'.join(connectors))
    db = conn.cursor()
    print('done')
    try:
        db.execute(sql)
        for row in db:
            topic_ids.append(row[0])
            topics.append(row[1])
        print()
    except Exception as e:
        print(e)
    finally:
        conn.close()
    topic_id2idx = {tid:i for i,tid in enumerate(topic_ids)}

    data_topic = np.zeros((20000,len(topic_ids)))
    sql = '''
select *
from Dflt.CKD_cohort_TopicWordCt2
where VWCMax>=4
'''
    print('Connecting to database ...', end=' ')
    conn = adodbapi.connect(';'.join(connectors))
    db = conn.cursor()
    print('done')
    try:
        db.execute(sql)
        n = 0
        for row in db:
            n += 1
            if n%10000==0:
                print(n//10000, end=' ')
            icn = row['PatientICN']
            tid = row['TripleIdx']
            if not tid in topic_id2idx:
                continue
            data_topic[icn2idx[icn],topic_id2idx[tid]] = 1
        print()
    except Exception as e:
        print(e)
    finally:
        conn.close()
#NLP
    data_nlpcls = np.zeros((20000,4))
    sql = '''
select PatientICN, YesHF = max(IIF(score>0.5943633025094739,1,0)),
	NoHF = max(IIF(score<-0.45686525688164836,1,0)),
	UnHF = max(IIF(score between -0.45686525688164836 and 0.5943633025094739,1,0))
from Dflt.CKD_cohort_TIU2_HFSnip_Score as a
join Dflt.CKD_cohort_TIU2_HFKW as b
on a.TIUDocumentSID = b.TIUDocumentSID
group by PatientICN
'''
    print('Connecting to database ...', end=' ')
    conn = adodbapi.connect(';'.join(connectors))
    db = conn.cursor()
    print('done')
    try:
        db.execute(sql)
        n = 0
        for row in db:
            n += 1
            if n%1000==0:
                print(n//1000, end=' ')
            icn = row['PatientICN']
            for k in range(3):
                value = row[k+1]
                data_nlpcls[icn2idx[icn],k] = value
                if value==1:
                    break
        print()
    except Exception as e:
        print(e)
    finally:
        conn.close()
    data_nlpcls[data_nlpcls.max(axis=1)==0,3] = 1

#Concatenate
    nlpcls = ['YesHF','NoHF','UncHF',
              'NoKW']
    X = np.concatenate([data,data_med,data_cpt,data_lab,data_title,data_topic
                        ,data_nlpcls
                        ],axis=1)#
    idxtr,idxvl,idxte = slice(4000,20000),slice(2000,4000),slice(0,2000)
    Xtr,Xvl,Xte = X[idxtr],X[idxvl],X[idxte]
    Ytr,Yvl,Yte = Y[idxtr],Y[idxvl],Y[idxte]
#    offset = np.zeros(X.shape[1])
#    scale = np.ones(X.shape[1])
    offset = np.mean(Xtr,axis=0)
    scale = np.std(Xtr,axis=0)

    for k in range(data.shape[1]):
        if np.isnan(data[:,k].mean()):
            notnull = ~np.isnan(Xtr[:,k])
            offset[k] = np.mean(Xtr[notnull,k])
            scale[k] = np.std(Xtr[notnull,k])
            X[np.isnan(data[:,k]),k] = offset[k]
            print('%d\t%.2f\t%.2f' %(k,offset[k],scale[k]))

#LinearSVC
    clf = LinearSVC(C=0.0002,dual=False,intercept_scaling=1)
    clf.fit((Xtr-offset)/scale,Ytr)
    scores = clf.decision_function((X-offset)/scale)

    fprs,tprs,_ = metrics.roc_curve(Y[idxtr],scores[idxtr])
    auc_tr = metrics.auc(fprs,tprs)
    fprs,tprs,_ = metrics.roc_curve(Y[idxvl],scores[idxvl])
    auc_vl = metrics.auc(fprs,tprs)
    fprs,tprs,thresholds = metrics.roc_curve(Y[idxte],scores[idxte])
    auc_te = metrics.auc(fprs,tprs)
    print('{:.3f}    {:.3f}'.format(
        auc_tr*100,auc_vl*100))
    print('{:.3f}    {:.3f}    {:.3f}'.format(
        auc_tr*100,auc_vl*100,auc_te*100))