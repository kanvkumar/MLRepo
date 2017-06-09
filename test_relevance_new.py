import nltk
from nltk.tokenize import sent_tokenize
from nltk.stem.porter import *
import re,os,json
import timezone_timeexpr as te
from relevance import RelevanceFinder as rf
from sutime import SUTime

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

#root word list
pos_relevance = ['meet','arrang','connect','organ','schedul','avail','chat','set','free','discuss','call','setup',
'confirm','coordin','look','work','time','help','make','forward','will','say','great','suggest','open','plan','good',
'better','ok','shoot','keep','anytim','appoint','book']
set_pos_relevance = set(pos_relevance)

neg = ["not","n't","but"]
set_neg = set(neg)

#neg root word list
neg_relevance = ['block','busi','leav','postpon','reschedul','re-schedul','sorri','tricki','vacat','skip','unfortun',
'imposs']
set_neg_relevance = set(neg_relevance)

past = ['was','had','did']
set_past = set(past)

file = open('task4_cases.txt','r').read().split('\n')
# fname = 'task4_cases.txt'
# with open(fname) as f:
#     content = f.readlines()

x = "relevant positive"
y = "relevant negative"
z = "not relevant"

stemmer = PorterStemmer()
#path_jar = "/home/kanv/python-sutime/"
jar_files = os.path.join(os.path.dirname(__file__), 'jars')
sutime = SUTime(jars=jar_files, mark_time_ranges=True)

for line in file:
    #line = line.encode('utf-8')
    line = line.encode('ascii', 'ignore')
    sent_tokenize_list = sent_tokenize(line)
    #print sent_tokenize_list
    print "-------------------------------"
    res = []
    dt = []
    for sent in sent_tokenize_list:
        sent_list = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)(\s|[A-Z].*)',sent)
        print sent_list
        for sent_new in sent_list:
            #print sent_new
            #print sent,type(sent)
            #dt = list(json.dumps(sutime.parse(sent), sort_keys=True))
            dt = sutime.parse(sent_new)
            #print dt
            #irrelev if no timeframe info
            if len(dt)==0:
                #res.append(('',z))
                continue
            else:
                word_neg_rel = ''
                word_pos_rel = ''
                #print dt
                tokens = nltk.word_tokenize(sent_new.lower())
                neg_flag = 0
                for token in tokens:
                    if token.lower() in set_neg:
                        neg_flag = 1
                        break
                
                #check neg_relevance
                neg_rel_flag = 0
                for token in tokens:
                    word = str(stemmer.stem(token.lower()))
                    if word in set_neg_relevance:
                        word_neg_rel = token.lower()
                        neg_rel_flag = 1
                        break

                #check pos_relevance
                pos_rel_flag = 0
                for token in tokens:
                    word = str(stemmer.stem(token.lower()))
                    if word in set_pos_relevance:
                        word_pos_rel = token.lower()
                        pos_rel_flag = 1
                        break

                # no neg word present 
                if neg_flag == 0:                    
                    #neg relev
                    if neg_rel_flag == 1:
                        if len(dt) > 0:
                            for val in dt:
                                time_event = val['text'].split()[0]
                                
                                te = tokens.index(time_event.lower())
                                tr = tokens.index(word_neg_rel)
                                #rel positive
                                if te >=1 and (tokens[te-1]=='to' or tokens[te-1]=='for' or tokens[te-1]=='fro') and tr < te:
                                    res.append((str(val['text']),x))
                                # rel negative
                                else:
                                    res.append((str(val['text']),y))
                    
                    # no neg relev
                    else:
                        
                        #temp=""
                        #print "'"+sent
                        for val in dt:
                            # p = sent.find(val['text'])
                            # temp = sent[:p]
                            # temp_tokens = nltk.word_tokenize(temp)
                            # flg_pos = 0
                            # for temp_token in temp_tokens:
                            #     word = str(stemmer.stem(temp_token.lower()))
                            #     if word in set_pos_relevance:
                            #         flg_pos = 1
                            #         break
                            
                            # if 'so' in tokens:
                            #     so = sent_new.find('so')
                            print val['text']
                            
                            #time_event = val['text'].split()[0]
                            ind = -1
                            if len(word_pos_rel)>0:
                                ind = tokens.index(word_pos_rel)
                            #tp = tokens.index(time_event)

                            if pos_rel_flag == 1 and (ind >=1 and tokens[ind-1] in set_past):
                                res.append((str(val['text']),z))

                            elif pos_rel_flag == 1 :
                                res.append((str(val['text']),x))
                            
                            elif tokens[len(tokens)-1]=='?':
                                res.append((str(val['text']),x))
                            elif tokens[len(tokens)-1]=='!':
                                res.append((str(val['text']),x))
                            #irrelev
                            else:
                                res.append((str(val['text']),z))

                # negation present
                else:
                    p = -1
                    q = -1
                    #r = -1
                    if 'but' in tokens:
                        p = sent_new.find('but')
                    if 'not' in tokens:
                        q = sent_new.find('not')
                    # 'but' and 'not' present
                    if p!=-1 and q!=-1:
                        for val in dt:
                            r = sent_new.find(val['text'])
                            if r < p and r < q:
                                res.append((str(val['text']),x))
                            elif p < r and q < r:
                                res.append((str(val['text']),y))
                            elif p < r and r < q :
                                res.append((str(val['text']),x))
                    # 'not' present
                    elif p==-1 and q!=-1:
                        for val in dt:
                            r = sent_new.find(val['text'])
                            if r < q:
                                res.append((str(val['text']),x))
                            elif q < r:
                                res.append((str(val['text']),y))
                    
                    # 'but' present
                    elif p!=-1 and q==-1:
                        for val in dt:
                            r = sent_new.find(val['text'])
                            if r < p :
                                res.append((str(val['text']),y))
                            elif p < r:
                                res.append((str(val['text']),x))

    print res