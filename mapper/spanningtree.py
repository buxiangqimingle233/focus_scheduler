import numpy as np
import pandas as pd
from functools import reduce
from utils.global_control import *


class SpanningTree():

    rotate_to_mapper = {0: 3, 1: 4, 2: 2, 3: 5, 5: 1}

    def __init__(self):
        super().__init__()

    def wrapper_genST(self, captain, region):

        array = np.asarray(region)
        array_x = (array / array_diameter).astype(np.int64)
        array_y = (array % array_diameter).astype(np.int64)

        max_x, min_x = array_x.max(), array_x.min()
        max_y, min_y = array_y.max(), array_y.min()

        self.x_ = [min_x, min_x, max_x, max_x]
        self.y_ = [min_y, max_y, max_y, min_y]

        captain = (int(captain / array_diameter), int(captain % array_diameter))
        raw = self.genST(captain)

        def transform2focus(item):
            node, son = item
            node_id = node[0] * array_diameter + node[1]
            ret = [(node_id, s) for s in son]
            return ret

        ret = map(transform2focus, raw.items())
        ret = reduce(lambda x, y: x + y, ret)

        return ret, max_y-min_y+max_x-min_x

    def genST(self, root):
        x_, y_ = self.x_, self.y_
        rx, ry = root
        
        dx = [1,0,-1,0]
        dy = [0,-1,0,1]
        lx, ly = x_[0], y_[0]
        mark = np.zeros((x_[2]-x_[0]+1,y_[2]-y_[0]+1), int)
        tree = []
        tree.append((rx, ry))
        mark[rx-lx][ry-ly] = 1
        son = []
        l = 0
        r = 1

        def out(x,y):
            #1是在矩形内，0是在矩形外
            if x < x_[0] or x > x_[2] or y < y_[0] or y > y_[2]:
                return 0
            return 1

        while l < r:
            x = tree[l][0]
            y = tree[l][1]
            son.append([])
            for i in range(4):
                xx = x + dx[i]
                yy = y + dy[i]
                if out(xx,yy) and mark[xx-lx][yy-ly]==0:
                    mark[xx-lx][yy-ly] = 1
                    son[l].append(i)
                    tree.append((xx,yy))
                    r += 1
            son[l].append(5)
            l += 1

        ret = {n: [self.rotate_to_mapper[factor] for factor in s] for n, s in zip(tree, son)}
    
        return ret


if __name__ == "__main__":
    sp = SpanningTree()
    ret = sp.wrapper_genST(0, [0, 1, 2, 8, 9, 10])
    print(ret)
    
    # sp = SpanningTree([(1, 1), (1, 3), (3, 3), (3, 1)])
    # print(sp.genST((1, 1)))