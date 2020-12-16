#include "utility.hh"

bool utility::fill(std::vector<int>& mark, int n){
    if(mark[n]==1){
        return false;
    }else{
        mark[n] =1;
        return true;
    }
}


int utility::calDepth(std::vector<utility::gate>& circuit, int n){
    std::vector<int> layerMark(n,0);
    int depth=1, cur=0;
    while(cur < circuit.size()){
        utility::gate g = circuit[cur];
        if(g.type == 1){
            if(!utility::fill(layerMark, g.tar)){
                 depth += 1;
                 layerMark = std::move(std::vector<int>(n,0));
            }else{
                cur += 1;
            }
        }else if(g.type == 2){
            if(!utility::fill(layerMark, g.tar) || !utility::fill(layerMark, g.ctrl)){
                 depth += 1;
                 layerMark = std::move(std::vector<int>(n,0));
            }else{
                cur += 1;
            }
        }
    }
    return depth;
}


int utility::countQubits(std::vector<utility::gate>& circuit, int n){
    std::vector<int> mark(n,0);
    int res = 0;
    for(auto& g : circuit){
        if(mark[g.tar] == 0){
            mark[g.tar] = 1;
            res+=1;
        }
        if(mark[g.ctrl] == 0){
            mark[g.ctrl] = 1;
            res+=1;
        }
    }
    return res;
}