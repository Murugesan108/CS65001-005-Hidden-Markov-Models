from util import *
import sklearn_crfsuite as crfsuite
from sklearn_crfsuite import metrics

class CRF(object):
    
    def __init__(self, trnfile, devfile):
        
        self.trn_text = load_data(trnfile)
        self.dev_text = load_data(devfile)
        #
        print("Extracting features on training data ...")
        self.trn_feats, self.trn_tags = self.build_features(self.trn_text)
        print("Extracting features on dev data ...")
        self.dev_feats, self.dev_tags = self.build_features(self.dev_text)
        #
        self.model, self.labels = None, None

    def build_features(self, text):
        feats, tags = [], []
        for sent in text:
            
            N = len(sent.tokens)
            sent_feats = []
            for i in range(N):
                word_feats = self.get_word_features(sent, i)
                sent_feats.append(word_feats)
            feats.append(sent_feats)
            tags.append(sent.tags)
        return (feats, tags)

        
    def train(self):
        print("Training CRF ...")
        self.model = crfsuite.CRF(
		### Earlier algorithm = 'lbfgs'
            algorithm='ap',
            max_iterations=5)
        self.model.fit(self.trn_feats, self.trn_tags)
        
        trn_tags_pred = self.model.predict(self.trn_feats)
        self.eval(trn_tags_pred, self.trn_tags)
        dev_tags_pred = self.model.predict(self.dev_feats)
        self.eval(dev_tags_pred, self.dev_tags)


    def eval(self, pred_tags, gold_tags):
        if self.model is None:
            raise ValueError("No trained model")
        print(self.model.classes_)
        print("Acc =", metrics.flat_accuracy_score(pred_tags, gold_tags))

        
    def get_word_features(self, sent, i):
        """ Extract features with respect to time step i
        """
        # the i-th token
        word_feats = {'tok':sent.tokens[i]}
        
        # TODO for question 1
        # the i-th tag
        ### Updating with tags
        #word_feats.update({'tags':sent.tags[i]})
        
        
        # TODO for question 2
        # add more features here
        ### Updating with extra features
        word_feats.update({'Frist_char':sent.tokens[i][0],
                           'First_two_char':sent.tokens[i][0:2],
                            'First_three_char':sent.tokens[i][0:3],
                            'First_char_upper': sent.tokens[i][0].isupper(),
                             'All_char_upper': sent.tokens[i].isupper(),
                           
                           'Last_char':sent.tokens[i][-1],
                           'Last_two_char':sent.tokens[i][-3:-1],
                           'Last_three_char':sent.tokens[i][-4:-1]
                              })
        
        ### Adding first and the last features
        
        ## Cast: First word
        ##If we do not have a previous word, we create a feature called 'Start word'
        if(i == 0):
            word_feats.update({'Prev_word':'<Start>'})
        else:
            word_feats.update({'Prev_word':sent.tokens[i-1]})
            
        ### Case: Last word
        
        ## If we have the last word, we create a feature called 'Last word'
        if(i == (len(sent.tokens)-1) ):
            word_feats.update({'Next_word': '<Last>'})
        else:
            word_feats.update({'Next_word': sent.tokens[i+1]})
        
        
        return word_feats


if __name__ == '__main__':
    trnfile = "trn-tweet.pos"
    devfile = "dev-tweet.pos"
    crf = CRF(trnfile, devfile)
    crf.train()