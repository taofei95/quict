#include "greedy_search_1D.hh"

#define HEURISTIC mapping::byCenter

//#define SIFT

bool mapping::s_cmp::operator()(const std::pair<int,float>& a, const std::pair<int,float>& b){
        return a.second < b.second;
}

void mapping::swap(int& a, int& b){
    int t = a;
    a = b;
    b = t;
}

bool mapping::L(const int& a, const int &b){
    return a<b;
}

bool mapping::G(const int& a, const int &b){
    return a>b;
}




//暴力算逆序对的个数
int mapping::countInversions(std::vector<int>& disF, std::vector<int>& constraint, mapping::compare func ){
    int res=0, l = constraint.size();
    if(l==1){
        return 0;
    }
    for(int i = 0; i<l; i++){
        for(int j=i+1; j<l; j++){
            if(!func(disF[constraint[i]], disF[constraint[j]])){
                res += 1;
            }
        }
    }
    return res;
}

int mapping::countSubConstraintInversion(std::vector<std::vector<int>>& constraint, std::vector<int>& disF, int s, int t){
    int res = 0;
    for(int i = 0; i<constraint[s].size(); i++){
        for(int j = 0; j<constraint[t].size(); j++){
            int cur = constraint[s][i], cmp = constraint[t][j];
            if(disF[cur] > disF[cmp]){
                res += 1;
            }
        }
    }
    return res;
}

int mapping::countConstraintInversion(std::vector<std::vector<int>>& constraint, std::vector<int>& initMapping, std::vector<int>& permutation){
    int n = initMapping.size();
    int res = 0;
    std::vector<int> disF(n, 0);
    for(int i =0; i<initMapping.size(); i++){
        disF[initMapping[i]] = i;
    }
    int l = permutation.size();
    for(int i = 0; i < l; i++){
        for(int j = i+1; j<l;j++){
            res += mapping::countSubConstraintInversion(constraint, disF, permutation[i], permutation[j]);
        }
    }
    return res;
}

//将均值作为得分
float mapping::byCenter(std::vector<int>& disF, std::vector<int>& constraint){
    int sum = 0;
    for(auto& p:constraint){
        sum += disF[p];
    }
    return sum/(float)(constraint.size());
}

//中位数作为得分
float mapping::median(std::vector<int>& disF, std::vector<int>&  constraint){
    int n = constraint.size();
    return disF[constraint[n/2]];
} 

//找最优初始mapping

std::vector<utility::edge> mapping::constructGraph(std::vector<utility::gate>& circuit, int n){
    std::vector<std::vector<int>> graph(n, std::vector<int>(n, 0));
    for(auto &p : circuit){
        if(p.type == 2){
            int s = std::min(p.ctrl, p.tar), t = std::max(p.ctrl, p.tar);
            graph[s][t] += 1;
        }
    }
    std::vector<utility::edge> res;
    for (int i = 0; i < n;i++){
        for (int j = i + 1; j < n;j++){
            if(graph[i][j]>0){
                res.emplace_back(std::move(utility::edge(i,j,graph[i][j])));
            }
        }
    }
    return res;
}

int mapping::calLA(std::vector<utility::edge>& graph, std::vector<int>& permuatation){
    int n = permuatation.size();
    int res = 0;
    std::vector<int> disF(n, 0);
    for (int i = 0; i < n; i++){
        disF[permuatation[i]] = i;
    }
    for (auto &e : graph){
        res += e.w * abs(disF[e.t] - disF[e.s]);
    }
    return res;
}

std::vector<int> mapping::minLA(std::vector<utility::edge>& graph, int n){
    std::vector<int> direction(n, 0);
    std::vector<int> permutation(n, 0);
    for (int i = 0; i < n;i++){
        permutation[i] = i;
    }
    std::vector<std::vector<int>> visible;
    int mini = INT_MAX;
    while (true){
        int cur = mapping::calLA(graph, permutation);
        //cout << cur << endl;
        if(cur<mini){
            mini = cur;
            visible.clear();
            visible.push_back(permutation);
            //cout << cur << endl;
        }else if (cur == mini){
            visible.push_back(permutation);
        }
        int li = mapping::findLargestMove(permutation, direction);
        //cout << li << endl;
        if (li == -1){
            break;
        }
        int elem = permutation[li];
        if(direction[elem] == 0){
            swap(permutation[li], permutation[li - 1]);
        }else if(direction[elem] == 1){    
            swap(permutation[li], permutation[li + 1]);
        }
        for (int i = elem + 1; i < n; i++){
            direction[i] = (direction[i] == 0?1 : 0);
        }
       /* for(auto &p:permutation){
            cout << p << " ";
        }*/
        //cout << endl;
    }
    int nv = visible.size(), pos = rand()%nv;
    permutation.swap(visible[pos]);
    return permutation;
}


std::vector<int> mapping::findInitMapping(std::vector<utility::gate>& circuit, int n){
    std::vector<utility::edge> graph = mapping::constructGraph(circuit, n);
    return minLA(graph, n);
}

//枚举所有的permutation找最优解

int mapping::findLargestMove(std::vector<int>& permutation, std::vector<int>& direction){
    int index = -1;
    for (int i = 0; i < permutation.size(); i++){
        if(direction[permutation[i]] == 0){
            if(i !=0 && permutation[i-1]<permutation[i]){
                index = index == -1 || permutation[i] > permutation[index] ?i : index;
            }
        }else{
            if(i != permutation.size()-1  && permutation[i]> permutation[i+1]){
                index = index == -1 || permutation[i] > permutation[index] ?i : index;
            }
        }
    }
    return index;
}

std::vector<int> mapping::enumerate(std::vector<int>& initMapping, std::vector<std::vector<int>>& constraint){
    int n = initMapping.size(), l = constraint.size();
    std::vector<int> direction(l, 0);
    std::vector<int> permutation(l, 0);
    std::vector<std::vector<int>> visible;
    int mini = n * n;
    for (int i = 0; i < l; i++){
        permutation[i] = i;
    }
    std::vector<int> disF(n, 0);
    for(int i =0; i<initMapping.size(); i++){
        disF[initMapping[i]] = i;
    }

    for(auto& p: constraint){
         if(mapping::countInversions(disF,p, L)>mapping::countInversions(disF,p, G)){
            std::reverse(p.begin(),p.end());
         }
    }
  //  int mark = 1;
    int i = 0;
    while (true){
        int cur = mapping::countConstraintInversion(constraint, initMapping, permutation);
        //cout << cur << endl;
        if(cur < mini){
            mini = cur;
            visible.clear();
            visible.push_back(permutation);
            //cout << mini << endl;
        }else if (mini == cur){
            visible.push_back(permutation);
        }
        int li = mapping::findLargestMove(permutation, direction);
        if (li == -1){
            //cout << li << endl;
            break;
        }

        int elem = permutation[li];
        if(direction[elem] == 0){
            mapping::swap(permutation[li], permutation[li - 1]);
        }else if(direction[elem] == 1){    
             mapping::swap(permutation[li], permutation[li + 1]);
        }
        for (int i = elem + 1; i < l; i++){
            direction[i] = (direction[i] == 0?1 : 0);
        }
        // for(auto&p: direction){
        //     cout << p << "  ";
        // }
        // cout << endl;
        // for (auto &p : permutation)
        // {
        //     cout << p << "  ";
        // }
        // cout << endl; 
    }

    //srand(time(NULL));
    int nv = visible.size(), pos = rand()%nv;
    permutation.swap(visible[pos]);
    std::vector<int> res(n);
    int index = 0;
    for (int i = 0; i < l; i++){
        std::vector<int> temp = constraint[permutation[i]];
        for(auto& p : temp){
            res[index++] = p;
        }
    }
    return res;
}

//构建初始分布




// 用启发式算法找满足约束的permutation
std::vector<int> mapping::searchMapping(std::vector<int>& initMapping, std::vector<std::vector<int>>& constraints){
    int n = initMapping.size();
    std::vector<int>  res(n,0);
    std::vector<int> disF(n, 0);
    for(int i = 0; i < n; i++){
        disF[initMapping[i]] = i;
    }
    //每个约束内部的排序，应该正排还是倒排
    for(auto& p: constraints){
         if(mapping::countInversions(disF,p, L)>mapping::countInversions(disF,p, G)){
            std::reverse(p.begin(),p.end());
         }
    }
    std::vector<std::pair<int, float> > scores(constraints.size());
    for(int i = 0;i < constraints.size();i++){
        scores[i].first = i;
        //scores[i].second = mapping::median(disF, constraints[i]);
        scores[i].second = HEURISTIC(disF, constraints[i]);
    }
    std::sort(scores.begin(),scores.end(),mapping::s_cmp());
    //把所有约束拼接成一个permuatation
    int index = 0;
    for(int i = 0; i < scores.size(); i++){
        std::vector<int>& temp = constraints[scores[i].first]; 
        for(auto &p:temp){
            res[index++] = p;
        }
    }   
    return  res;
}


// 输出两个permutation之间准换所需的最小swap门序列
std::vector<utility::gate> mapping::mappingTrans(std::vector<int>& initMapping, std::vector<int>& tarMapping){
      int n = initMapping.size();
      std::vector<utility::gate> res;
      std::vector<int> disF(n, 0);
      for(int i = 0; i < n; i++){
           disF[tarMapping[i]] = i;
      }
      for(int i = 0; i < n; i++){
          for(int j = n-1; j >i; j--){
            int cur = initMapping[j], next = initMapping[j-1];
            if(disF[cur] < disF[next]){
                utility::gate temp(j, j-1, 2, 30);
                res.push_back(temp);
                mapping::swap(initMapping[j], initMapping[j-1]);
            }
          }
      }
      return res;
}

//在约束中插入一个门
void mapping::insertConstraint(std::vector<int>& constraint, int connect, int hang){

    if(constraint.front() == connect){
        constraint.insert(constraint.begin(), hang);
    }else{
        constraint.insert(constraint.end(), hang);
    }
}
//合并两个约束
void mapping::mergeConstraint(std::vector<int>& constraint1, std::vector<int>& constraint2, int c1, int c2){
    if(c1 == constraint1.front()  && c2 == constraint2.front()){
        std::reverse(constraint1.begin(),constraint1.end());
        constraint1.insert(constraint1.end(),constraint2.begin(),constraint2.end());
    }else if(c1 == constraint1.front()  && c2 == constraint2.back()){
        constraint1.insert(constraint1.begin(),constraint2.begin(),constraint2.end());
    }else if(c1 == constraint1.back()  && c2 == constraint2.front()){
        constraint1.insert(constraint1.end(),constraint2.begin(),constraint2.end());
    }else{
        std::reverse(constraint2.begin(),constraint2.end());
        constraint1.insert(constraint1.end(),constraint2.begin(),constraint2.end());
    }

}

// 找到gates中从pos开始最大可映射的门集合，以约束的形式返回

std::vector<std::vector<int>> mapping::findConstraint(std::vector<utility::gate>& gates, int& pos, int n){
     std::unordered_map<int, std::vector<int>> constraints;
     std::vector<int> index(n,0);
     std::vector<int> degree(n,0);
     std::vector<std::vector<int>> res;
     int keys = 0;
     for(; pos < gates.size(); pos++){
         utility::gate p = gates[pos];
         if(p.type == 2){
            int c = degree[p.ctrl], t = degree[p.tar];
            if( c == 0 && t == 0){
                index[p.ctrl] =keys;
                index[p.tar] =keys;
                degree[p.ctrl] =1;
                degree[p.tar] =1;
                std::vector<int> temp{p.ctrl,p.tar};
                constraints[keys++]=temp;
            }else if(c == 0 && t == 1){
                mapping::insertConstraint(constraints[index[p.tar]], p.tar, p.ctrl);
                degree[p.ctrl]+=1;
                degree[p.tar]+=1;
                index[p.ctrl] = index[p.tar]; 
            }else if(c == 1 && t== 0){
                mapping::insertConstraint(constraints[index[p.ctrl]], p.ctrl, p.tar);
                degree[p.ctrl]+=1;
                degree[p.tar]+=1;
                index[p.tar] = index[p.ctrl]; 
            }else if(c == 1 && t == 1){
                if(index[p.ctrl] == index[p.tar]){
                    if(constraints[index[p.ctrl]].size()>2){
                        break;
                    }
                }else{
                    mapping::mergeConstraint(constraints[index[p.ctrl]], constraints[index[p.tar]], p.ctrl, p.tar);
                    int tempKey = index[p.tar];
                    for(auto& elem : constraints[index[p.tar]]){
                        index[elem] = index[p.ctrl];
                    }
                    constraints.erase(tempKey);
                }
            }else{
                if(index[p.ctrl] == index[p.tar]){
                    std::vector<int> temp =  constraints[index[p.tar]];
                    int mark = 0;
                for(int i = 0;i<temp.size()-1;i++){
                    if((temp[i] == p.ctrl && temp[i+1] == p.tar) ||(temp[i] == p.tar && temp[i+1] == p.ctrl)){
                        mark = 1;
                        break;
                    }
                }
                if(mark == 0){
                    break;
                }
                }
            }
         }
     }

    for(auto& p : constraints){
        res.emplace_back(p.second);   
    }

    for(int i = 0; i <n ;i++){
        if(degree[i] == 0){
            res.emplace_back(std::vector<int>(1, i));
        }
    }

    return res;
}

std::vector<utility::gate> mapping::logicToPhysics(std::vector<utility::gate>& gates, std::vector<int>& mapping, int start, int end){
     int n = mapping.size();
     std::vector<int> invMapping(n);
     std::vector<utility::gate> res;
     for(int i =0; i < n; i++){
         invMapping[mapping[i]] = i;  
     }
     for(int i = start; i<end; i++){
        int c = gates[i].ctrl, t = gates[i].tar;
        res.emplace_back(utility::gate(invMapping[c], invMapping[t], gates[i].type, gates[i].gateName));
     }
    return res;
}


std::vector<utility::gate> mapping::greedySearch(std::vector<utility::gate>& gates, std::vector<int>& mapping, int n){
    int curPos = 0, prePos = -1, l = gates.size();
    std::vector<int> initMapping(n,0);
    for(int i=0; i<n; i++){
        initMapping[i] = i;
    }
    std::vector<utility::gate> res;
    int init = 1;
    
    while(curPos <l){
        prePos = curPos;    
        //cout<<"permutation"<<endl;
        if(!init){
            std::vector<std::vector<int>> constraints = mapping::findConstraint(gates, curPos, n);
            std::vector<int> curMapping = mapping::searchMapping(initMapping, constraints);
            //std::vector<int> curMapping = mapping::enumerate(initMapping, constraints);
            std::vector<utility::gate> swaps = mapping::mappingTrans(initMapping, curMapping);
            std::vector<utility::gate> transGates = mapping::logicToPhysics(gates, curMapping, prePos, curPos);
            res.insert(res.end(), swaps.begin(), swaps.end());
            res.insert(res.end(), transGates.begin(), transGates.end());
        }else{
            //cout<<"init"<<endl;
            std::vector<int> curMapping = mapping::findInitMapping(gates, n);
            
            initMapping = curMapping;
            //res.insert(res.end(), transGates.begin(), transGates.end());
            std::vector<std::vector<int>> constraints = mapping::findConstraint(gates, curPos, n);
            initMapping = mapping::searchMapping(initMapping, constraints);  
            mapping = initMapping;
            std::vector<utility::gate> transGates = mapping::logicToPhysics(gates, initMapping, prePos, curPos);
            res.insert(res.end(), transGates.begin(), transGates.end());
            init = 0;
        }
    }
    return res;
} 



#ifdef SIFT

// 利用sifting技术找好的permutation

//将电路划分为conflict-free的集合
std::vector<std::vector<std::vector<int>>> mapping::partitionGateSets(std::vector<utility::gate>& circuit, int n){
    int curPos = 0, l = circuit.size();
    std::vector<std::vector<std::vector<int>>> res;
    while(curPos <l){
        res.emplace_back(std::move(mapping::findConstraint(circuit, curPos, n)));
    }
    return res;
}


//按集合中门的数量简历索引表
bool cmp(const std::vector<int>& a, const std::vector<int>& b){
    return a[0] > b[0];
}

std::vector<std::vector<int>> mapping::constructIndex(std::vector<std::vector<std::vector<int>>>& constraints, int n){
    int num = 0;
    for (auto& p : constraints){
        num += p.size();
    }
    std::vector<std::vector<int>> res(num);
    int ind = 0;
    for (int i = 0; i < constraints.size(); i++){
        int w = constraints[i].size();
        for (int j = 0; j < w; j++){
            res[ind++] = std::move(std::vector<int>{i, j, (int)constraints[i][j].size(), w});
        }
    }
    std::sort(res.begin(), res.end(), cmp);
    return res;
}

void mapping::fillMapping(std::vector<std::vector<int>>& constraint, std::vector<int>& mapping, int n){
    int w = constraint.size();
    int ind = 0;
    for(int i =0; i<w; i++){
        for(int j = 0; j<constraint[i].size(); j++){
            mapping[ind++] = constraint[i][j];
        }
    }
}

std::vector<std::vector<int>>  constructInterMapping(std::vector<std::vector<std::vector<int>>>& constraints, int n){
    int l = constraints.size();
    std::vector<std::vector<int>> res(l,std::vector<int>(n,0));
    for(int i = 0; i < l; i++){
        mapping::constructIndex(constraints[i], res[i], n);
    }
    return res;
}
//global sifting

//swap two constraint 

void  swapConstraint(std::vector<std::vector<int>>& constraint, int i, int j){
    int n = constraint.size();
    constraint[i].swap(constraint[j]);
}

std::vector<utility::gate> globalSifting(std::vector<utility::gate>& circuit, std::vector<int>& mapping ,int n){
    std::vector<std::vector<std::vector<int>>> constraints = std::move(mapping::findInitMapping(circuit, n));
    std::vector<std::vector<int>> list = std::move(mapping::constructIndex(constraints, n));
    std::vector<std::vector<int>> interMapping = std::move(constructInterMapping(constraints,n));
    int failCount = 0, maxFail = 5, fail = 0, ind = 0;
    while(failCount < maxFail){
        if(fail == 0){
            int cs = 0;            
            std::vector<int> cur = list[ind];
            
        }
        else{
            
        }
    }
}

#endif