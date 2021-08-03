import numpy as np
# from utils.global_control import *

array_diameter = 8

class SpanningTree():

    rotate_to_mapper = {0: 3, 1: 4, 2: 2, 3: 5, 5: 1}

    def __init__(self, rectangle):
        #左下角，左上角，右上角，右下角
        self.x_, self.y_ = zip(*rectangle)
        # self.y_ = [array_diameter-y-1 for y in self.y_]
        # self.x_, self.y_ = self.y_, self.x_
        
    def genST(self, root):
        x_, y_ = self.x_, self.y_
        rx, ry = root
        
        dx = [1,0,-1,0]
        dy = [0,-1,0,1]
        lx, ly = x_[0], y_[0]
        print(x_, y_)
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
    sp = SpanningTree([(1, 1), (1, 3), (3, 3), (3, 1)])
    print(sp.genST((1, 1)))
# x_ = [1,1,3,3]
# y_ = [1,3,3,1]

# lx = x_[0]
# ly = y_[0]
# rx = 2
# ry = 2
# mark = np.zeros((x_[2]-x_[0]+1,y_[2]-y_[0]+1),int)
# tree = []
# tree.append((rx,ry))
# mark[rx-lx][ry-ly] = 1
# son = []
# l = 0
# r = 1
# def out(x,y):
#     #1是在矩形内，0是在矩形外
#     if x < x_[0] or x > x_[2] or y < y_[0] or y > y_[2]:
#         return 0
#     return 1
# while(l<r):
#     x = tree[l][0]
#     y = tree[l][1]
#     son.append([])
#     for i in range(4):
#         xx = x + dx[i]
#         yy = y + dy[i]
#         if out(xx,yy) and mark[xx-lx][yy-ly]==0:
#             mark[xx-lx][yy-ly] = 1
#             son[l].append(i)
#             tree.append((xx,yy))
#             r += 1
#     son[l].append(5)
#     l += 1
# for i in range(r):
#     print('{}:{}'.format(tree[i], son[i]))