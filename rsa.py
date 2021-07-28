# RSA code after merging old data and new data

import numpy as np
import math
from collections import defaultdict

BOUND=0.1 #0.6730116670092565
BASE=math.e
NON_ZERO=0.0000000000000001
theta_type_old_data = 0.0007
theta_type_new_data = 0.012
theta_att_old_data = 0.00252
theta_att_new_data = 0.00252

beta = 0.5
alpha = 1

RelationBetween = ['next to']
RelationAmong = ['the left', 'the right', 'the middle', 'the biggest', 'the bigger', 'the smallest', 'the smaller']
RelationOrdinal = []
ordinal_map = {
    1: 'first',
    2: 'second',
    3: 'third',
    4: 'fourth',
    5: 'fifth'
}

def row_objs(objects, target, targeted_type):
    name,salience, x,y,w,h = list(objects[objects['box_alias']==target].iloc[0,:])
    row = []
    for obj in targeted_type:
        other_name, other_salience, other_x,other_y,other_w, other_h = list(objects[objects['box_alias']==obj].iloc[0,:])
        if y <= other_y <= y + h or other_y <= y <= other_y+ other_h:
            row.append([obj, other_x, other_y])
    row.sort(key=lambda x: x[1])
    return [item[0] for item in row]

class RSA:
    def __init__(self, df, **kwargs):
        self.df = df
        self.types = [col[5:] for col in df.columns if 'TYPE_' in col]
        self.attributes = [col[5:] for col in df.columns if 'ATTR_' in col]
        self.objects = [obj for obj in df['box_alias']]
        self.saliences = list(df['salience'])
        self.objects_by_type = self._sort_obj_by_type()

        self.obj_to_types, self.obj_to_attributes = self.create_obj_to_attr_types()
        self.attributes_for_type = self.create_attributes_for_type()
        generated_relations = kwargs['generated_relations']
        self._process_relations(generated_relations)
        if 'model' in kwargs:
            self.model = kwargs['model']
            self.word_to_idx = kwargs['word_to_idx']
            self.idx_to_word = kwargs['idx_to_word']
        self.last_3_word = ["", ""]

    def _encode_sentence(self, sentence):
        out = []
#       if the word does not appear in the dictionary/vocabulary, treat it as a blank character
        for word in sentence:
            if word not in self.word_to_idx:
                out.append(self.word_to_idx[''])
            else:
                out.append(self.word_to_idx[word])
        return np.array([out])

    def _process_relations(self, relation_data):
        for obj in relation_data.keys():
            relations = relation_data[obj]
            for relation_blob in relations:
                relation, target, prob = ' '.join(relation_blob[:-2]), relation_blob[-2], relation_blob[-1]
                ## TODO: find a representation for target (currently use the object word) ##
                target_str = target[:target.find('-')]
                self.obj_to_attributes[obj][f'{relation} {target_str}'] = 1 #prob

    def _sort_obj_by_type(self):
        objs_by_type = defaultdict(list)
        for obj in self.objects:
            obj_type = obj[:obj.index('-')]
            objs_by_type[obj_type].append(obj)
        return objs_by_type

    def create_obj_to_attr_types(self):
        obj_to_types = defaultdict(dict)
        for t in self.types:
            col_type = f'TYPE_{t}'
            type_df = self.df[['box_alias', col_type, 'is_old_type']]
            for index, row in type_df.iterrows():
                obj = row.iloc[0]
                prob = row.iloc[1]
                is_old_type = row.iloc[2]
                if ((is_old_type and prob > theta_type_old_data) or ((not is_old_type) and prob > theta_type_new_data) ):
                    obj_to_types[obj][t] = 1 #prob
        
        obj_to_attributes = defaultdict(dict)
        for att in self.attributes:
            col_att = f'ATTR_{att}'
            attr_df = self.df[['box_alias', col_att, 'is_old_attr']]
            for index, row in attr_df.iterrows():
                obj = row.iloc[0]
                prob = row.iloc[1]
                is_old_attr = row.iloc[2]
                if ((is_old_attr and prob > theta_att_old_data) or ((not is_old_attr) and prob > theta_att_new_data) ):
                    obj_to_attributes[obj][att] = 1 #prob
        return obj_to_types, obj_to_attributes


    def create_attributes_for_type(self):
        attributes_for_type = defaultdict(list)
        for t in self.types:
            attributes_for_type[t] = self.attributes
        return attributes_for_type


    def utterancePrior(self, utts):
        utt = np.full((1,len(utts)),1/len(utts))
        return utt[0]


    def objectPrior(self):
        return self.saliences/np.sum(self.saliences)


    def _attribute_meaning(self, utterance, obj):
        if obj in self.obj_to_attributes and utterance in self.obj_to_attributes[obj]:
            return self.obj_to_attributes[obj][utterance]
        return 0
    def _type_meaning(self, utterance, obj):
        if obj in self.obj_to_types and utterance in self.obj_to_types[obj]:
            return self.obj_to_types[obj][utterance]
        return 0
    def meaning(self, utterance, obj, utterance_type='ATTR'):
        if utterance_type == 'ATTR':
            return self._attribute_meaning(utterance, obj)
        return self._type_meaning(utterance, obj)


    # utt: the utterance in evaluation
    # pri: prior of the objects
    # utterance_type: type of the utterance - either a TYPE utterance or an ATTR utterance (attribute)
    def literal_listener(self, utt,pri, utterance_type):
        result = [self.meaning(utt, obj, utterance_type)*p for obj, p in zip(self.objects, pri)]
        return result/np.sum(result)


    # utt: the utterance under consideration
    def cost(self, utt):
        result = len(utt.split('_'))
        return 1#result


    # obj: the object under consideration
    # pri: the object prior
    # t: a specific type
    # curr: list of words that have been spoken already
    def speaker(self, obj, pri, t, curr):
        # either consider the types utterance or the attribute utterance if t != 0 then we use all attributes related to type t
        if len(t) == 0:
            us = list(self.obj_to_types[obj].keys())#self.types
            utterance_type = 'TYPE'
        else:
            us = list(self.obj_to_attributes[obj].keys())#self.attributes_for_type[t]
            utterance_type = 'ATTR'
        # discard words that have been used already
        utts = [c for c in us if c not in curr]
        if len(utts) > 0:
            idx = self.objects.index(obj)
            # probability of all the utterances given the input object
            prob = [self.literal_listener(utt, pri, utterance_type)[idx] for utt in utts]
            # feeding last 3 word to a model to adjust utterances' prior
            if len(self.last_3_word) >= 3 and hasattr(self, 'model'):
                model_input = self._encode_sentence(self.last_3_word[-3:])
                output = self.model(model_input)
                # adjust all utterances that appear in the vocabulary of the training model
                for i, utt in enumerate(utts):
                    if utt in self.word_to_idx:
                        prob[i] += output[0][self.word_to_idx[utt]]
                        prob[i] = prob[i].numpy()
            #calculate the likelihood with the formula: exp (alpha * (log - beta*cost))
            logvalue = [math.log(p) if p > 0 else -2147483647 for p in prob]
            utility = [logv-beta*self.cost(utt) for logv,utt in zip(logvalue,utts)]
            result = [math.exp(alpha*util) for util in utility]
            if np.sum(result) == 0:
                return utts,list(np.zeros(len(result))), 'None' # return utterance_type of None to signify that there's no preference in next word
            res = result/np.sum(result)
            return utts,res, utterance_type
        else:
            return utts,[], 'None' 


    def entropy(self, p):
        ent=0
        for i in p:
            ent -= i*math.log(i+NON_ZERO,BASE)
        return ent

    # target_object: the target object
    # objects: list of all the objects
    # types: all the possible types (used as utterance)
    # attributes_for_type: a dict that map all possible attributes to each type
    def full_speaker(self, target_object):
        output = []
        prior = self.objectPrior()
        t = ''
        ############################
        # CREATE SPATIAL UTTERANCE #
        ############################
        target_type = target_object[:target_object.index('-')]
        target_type_objs = self.objects_by_type[target_type]
        objects_with_box = self.df[['box_alias','salience', 'x1','y1','w', 'h']]
        for i, obj in enumerate(target_type_objs):
            name,salience, x,y,w,h = list(objects_with_box[objects_with_box['box_alias']==obj].iloc[0,:])
            if len(target_type_objs) == 2:
                other_obj = target_type_objs[(i+1)%2]
                other_name, other_salience, other_x,other_y,other_w, other_h = list(objects_with_box[objects_with_box['box_alias']==other_obj].iloc[0,:])
                max_prob = max(self.obj_to_attributes[obj].values())
                if x < other_x:
                    ordinal_rel = "the left"
                    self.obj_to_attributes[obj][ordinal_rel] = 1 #max_prob
                elif x > other_x:
                    ordinal_rel = "the right"
                    self.obj_to_attributes[obj][ordinal_rel] = 1 #max_prob
                if w*h >= 1.1 * other_w * other_h:
                    ordinal_rel = "the bigger"
                    self.obj_to_attributes[obj][ordinal_rel] = 1 #max_prob
                elif w*h <= 0.9 * other_w * other_h:
                    ordinal_rel = "the smaller"
                    self.obj_to_attributes[obj][ordinal_rel] = 1 #max_prob
            elif len(target_type_objs) >= 3:
                row_of_objs = row_objs(objects_with_box, obj, target_type_objs)
                target_idx = row_of_objs.index(obj)
                max_prob = max(self.obj_to_attributes[obj].values())
                if target_idx < len(row_of_objs)/2 and target_idx <= 3:
                    ordinal_rel = f"the {ordinal_map[target_idx+1]} from left"
                    # setting the probability of this ordinal to max value among all attributes of this obj (originally set to 1)
                    self.obj_to_attributes[obj][ordinal_rel] = 1 #max_prob
                elif target_idx >= len(row_of_objs)/2 and target_idx >=  len(row_of_objs) - 3:
                    ordinal_rel = f"the {ordinal_map[len(row_of_objs)- target_idx]} from right"
                    self.obj_to_attributes[obj][ordinal_rel] = 1 #max_prob #1
        ############################
        # DONE CREATE SPATIAL UTTERANCE #
        ############################
        # during speaker iterations, keep track if we ever run into situation where we have nothing to say or no preference
        # a True low_confidence means that either the list of possible utterance is empty OR the calculating probability is
        # all 0 everywhere. This might be because of theta value we set is too high (there's no type/attr with high enough
        # value to be associated with the target_object in obj_to_types and obj_to_atttributes) 
        low_confidence = False
        # stop after reaching the expression limit (i.e preventing the case of generating expressions that are too long)
        expression_limit = 4
        for iter in range(expression_limit):
            # the utterances and the corresponding probabilities that a pragmatic speaker would take
            utts,pro, utterance_type = self.speaker(target_object, prior, t, output) 
            if utterance_type == 'None':
                low_confidence = True
            if len(utts) > 0:
                # idx of the most likely word that speaker will choose (highest probability)
                idx=np.argmax(pro)
                # if the prob of using this word is <= 0, priors of object is reset to default
                if pro[idx] <= 0:
                    new_c = prior
                else:
                    u = utts[idx]
                    # update the prior
                    new_c = self.literal_listener(u,prior, utterance_type)
                    # add the new utterance to the output
                    output.append(u)
                    # add the new utterance to the list of wors spoken so far (used to predict next word with a model)
                    self.last_3_word.append(u)
                # calculate entropy, break if entropy is small enough
                ent = self.entropy(new_c)
                # print('ENTROPY:', ent)
                # print(iter,'new_c',new_c)
                # print(iter,'ent',ent)
                prior = new_c
                t = utterance_type
                if ent <= BOUND:
                    break
        return output, low_confidence