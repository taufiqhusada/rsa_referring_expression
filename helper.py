import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import heapq
from nltk.corpus import wordnet

# FUNCTION TO DRAW A BOX AROUND AN OBJECT 
def draw_box_obj(name, x,y,w,h, img=None, ax=None, filled=False, color=None):
    if not ax:
        fig,ax = plt.subplots(1)
        ax.add_image(img)
    color = np.random.rand(3,) if not color else color
    box_plot = Rectangle((x,y),w,h, fill=filled, edgecolor=color, linewidth=2)
    ax.add_patch(box_plot)
    ax.text(x+w//2,y+h//2, name, fontsize=15, color=color, horizontalalignment='center', verticalalignment='center')
    return ax, color

#CALCULATING THE OVERLAPPING AREA OF 2 BOXES
def calc_overlap(x1_tl,y1_tl,w1,h1,x2_tl,y2_tl,w2,h2):
    # find the bottom right corner of the 2 images
    x1_br, y1_br = x1_tl + w1, y1_tl + h1
    x2_br, y2_br = x2_tl + w2, y2_tl + h2
    dx = min(x1_br, x2_br) - max(x1_tl, x2_tl)
    dy = min(y1_br, y2_br) - max(y1_tl, y2_tl)
    # uncomment nextline to return the overlapped box
    #return max(x1_tl, x2_tl), max(y1_tl, y2_tl), dx, dy
    return dx*dy if (dx >= 0 and dy >= 0) else 0

#CALCULATING THE TOP 5 BEST MATCH TO THE TARGET LABEL
def top_5_match(candidates, target):
    x1, y1, w1,h1 = target[0],target[1],target[2],target[3]
    best_matches = []
    for i in range(len(candidates)):
        name, x2,y2,w2,h2 = list(candidates.iloc[i,:])
        overlapped_area = calc_overlap(x1,y1,w1,h1,x2,y2,w2,h2)
        total_area = w1*h1 + w2*h2 - overlapped_area
        similarity = overlapped_area/total_area
        heapq.heappush(best_matches, [-similarity, name, i])

    top_5 = [heapq.heappop(best_matches) for _ in range(min(5, len(best_matches)))]
    for elem in top_5:
        elem[0], elem[1], elem[2] = elem[2], elem[1], -elem[0]
    return top_5


# GET THE SYNONYMS OF A WORD FROM NLTK WORDNET
def get_synonyms(word):
    synsets = wordnet.synsets(word)
    synonyms = set()
    for syn in synsets:
        for l in syn.lemmas(): 
            synonyms.add(l.name())
    return synonyms