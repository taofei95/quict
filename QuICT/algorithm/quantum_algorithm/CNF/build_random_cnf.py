import random
#n：变量数；k：子句的文字数 子句宽度。输出一个子句，参数为n、k
def ratio(n,k):
    if 3 == k:
        return int(n*4.267)
    elif 4 == k:
        return int(n*9.931)
    elif 5 == k:
        return int(n*21.117)
    elif 6 == k:
        return int(n*43.37)
    elif 7 == k:
        return int(n*87.79)
    else:
        print("Error there is no ratio to output")
        return -1
    return -1

def rclause(n,k):
    templist=random.sample(range(1,n+1),k)
    templist.sort()
    for i in range(k):
        r = random.randint(0,1)
        if r == 1:
            templist[i] = -templist[i]
    return templist

def check(cnf,new):
    out = 1
    for i in range(len(cnf)):
        if len(cnf[i]) != len(new):
            print("Check error: length not match")
            return -1
        temp = 1
        for j in range(len(new)):
            if cnf[i][j] != new[j]:
                temp = 0
        if temp == 1:
            out = 0
    return out

#k：每个子句的最大文字数(子句宽度);n：变量数；m：子句数。输出一个随机CNF，参数为n、m、k
def rcnf(n,m,k,num):
    cnf = []
    count = 0
    while(True):
        new = rclause(n,k)
        nice = check(cnf,new)
        if 0 == nice:
            continue
        elif 1 == nice:
            cnf.append(new)
            #print(count)
            count = count + 1
        if count == m:
            break
    dir = "QuICT/algorithm/quantum_algorithm/CNF/cnf_test/"
    name = dir + str(k) + "_" + str(n) + "_" + str(m) + "_" + str(num)
    fo = open(name, "w")
    string = "p cnf " + str(n) + " " + str(m) + "\n"
    fo.write(string)
    #print("p cnf",n,m)
    for i in range(m):
        string = ""
        for j in range(k):
            string = string + str(cnf[i][j]) + " "
            #print(cnf[i][j], end = ' ')
        string = string + "0\n"
        #print('0')
        fo.write(string)
    fo.close()

n_list = [10, 12, 14, 16, 18, 20]
for num in range(1, 4):
    for n in n_list:
        rcnf(n,int(n*4.5),4,num)