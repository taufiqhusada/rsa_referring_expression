import numpy as np
import math
from collections import defaultdict

BOUND=0.1 #0.6730116670092565
BASE=math.e
NON_ZERO=0.0000000000000001
theta_type = 0.0007
theta_att = 0.0025
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

    def _process_relations(self, relation_data):
        for obj in relation_data.keys():
            relations = relation_data[obj]
            print("###")
            for relation_blob in relations:
                relation, target, prob = ' '.join(relation_blob[:-2]), relation_blob[-2], relation_blob[-1]
                ## TODO: find a representation for target (currently use the object word) ##
                target_str = target[:target.find('-')]
                print(f'{relation} {target_str}')
                self.obj_to_attributes[obj][f'{relation} {target_str}'] = prob

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
            type_df = self.df[['box_alias', col_type]]
            for index, row in type_df.iterrows():
                obj = row.iloc[0]
                prob = row.iloc[1]
                if prob > theta_type:
                    obj_to_types[obj][t] = prob
        
        obj_to_attributes = defaultdict(dict)
        for att in self.attributes:
            col_att = f'ATTR_{att}'
            attr_df = self.df[['box_alias', col_att]]
            for index, row in attr_df.iterrows():
                obj = row.iloc[0]
                prob = row.iloc[1]
                if prob > theta_att:
                    obj_to_attributes[obj][att] = prob
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
            us = self.types
            utterance_type = 'TYPE'
        else:
            us = list(self.obj_to_attributes[obj].keys())#self.attributes_for_type[t]
            print(t, obj, us[:10], len(us))
            print("$$$$$")
            print(list(self.obj_to_attributes[obj].keys()))
            utterance_type = 'ATTR'
        # discard words that have been used already
        utts = [c for c in us if c not in curr]
        print('the first from left' in utts)
        print(utts[:10])
        if len(utts) > 0:
            idx = self.objects.index(obj)
            # probability of all the utterances given the input object
            prob = [self.literal_listener(utt, pri, utterance_type)[idx] for utt in utts]
            #calculate the likelihood with the formula: exp (alpha * (log - beta*cost))
            logvalue = [math.log(p) if p > 0 else -2147483647 for p in prob]
            utility = [logv-beta*self.cost(utt) for logv,utt in zip(logvalue,utts)]
            result = [math.exp(alpha*util) for util in utility]
            if np.sum(result) == 0:
                return utts,list(np.zeros(len(result))), utterance_type
            res = result/np.sum(result)
            return utts,res, utterance_type
        else:
            return utts,[], utterance_type


    def entropy(self, p):
        ent=0
        for i in p:
            ent -= i*math.log(i+NON_ZERO,BASE)
        return ent

    # obj: the target object
    # objects: list of all the objects
    # types: all the possible types (used as utterance)
    # attributes_for_type: a dict that map all possible attributes to each type
    def full_speaker(self, obj):
        output = []
        prior = self.objectPrior()
        t = ''
        ############################
        # CREATE SPATIAL UTTERANCE #
        ############################
        target_type = obj[:obj.index('-')]
        target_type_objs = self.objects_by_type[target_type]
        print(f'# of {target_type}: {len(target_type_objs)}')
        objects_with_box = self.df[['box_alias','salience', 'x1','y1','w', 'h']]
        print(target_type_objs)
        for i, obj in enumerate(target_type_objs):
            name,salience, x,y,w,h = list(objects_with_box[objects_with_box['box_alias']==obj].iloc[0,:])
        #     print(name,salience, x,y,w,h)
            if len(target_type_objs) == 2:
                other_obj = targeted_type[(i+1)%2]
                other_name, other_salience, other_x,other_y,other_w, other_h = list(objects[objects['box_alias']==other_obj].iloc[0,:])
                if x < other_x:
                    print(f"the left {obj}")
                elif x > other_x:
                    print(f"the right {obj}")
                print(w*h, 1.1 * other_w * other_h)
                if w*h >= 1.1 * other_w * other_h:
                    print(f"the bigger {obj}")
                elif w*h <= 0.9 * other_w * other_h:
                    print(f"the smaller {obj}")
            elif len(target_type_objs) >= 3:
                row_of_objs = row_objs(objects_with_box, obj, target_type_objs)
                print(obj, row_of_objs)
                target_idx = row_of_objs.index(obj)
                print("HELOOOOOOOOOOOOO")
                if target_idx < len(row_of_objs)/2 and target_idx <= 3:
                    ordinal_rel = f"the {ordinal_map[target_idx+1]} from left"
                    print(ordinal_rel)
                    self.obj_to_attributes[obj][ordinal_rel] = 1
                elif target_idx >= len(row_of_objs)/2 and target_idx >=  len(row_of_objs) - 3:
                    ordinal_rel = f"the {ordinal_map[len(row_of_objs)- target_idx]} from right"
                    print(ordinal_rel)
                    self.obj_to_attributes[obj][ordinal_rel] = 1
        ############################
        # DONE CREATE SPATIAL UTTERANCE #
        ############################
                

        for iter in range(10):
            # print('iteration',iter)
            print("".join([f'{obj}\t' for obj in self.objects]))
            print("".join([f'{pri}\t' for pri in prior]))
            utts,pro, utterance_type = self.speaker(obj, prior, t, output) 
            #print(iter, prior)
            if len(utts) > 0:
                idx=np.argmax(pro)
                # print("YOYO",utts[idx])
            if pro[idx] <= 0:
                new_c = prior
            else:
                u = utts[idx]
                new_c = self.literal_listener(u,prior, utterance_type)
                output.append(u)
            ent = self.entropy(new_c)
            print('ENTROPY:', ent)
            # print(iter,'new_c',new_c)
            # print(iter,'ent',ent)
            prior = new_c
            t = output[0]
            if ent <= BOUND:
                break
        return output
