#ifndef UTILITY
#define UTILITY
#include<string>
#include<vector>

namespace utility{
class edge{
    public:
        int s;
        int t;
        int w;
        edge(){
            s = 0;
            t = 0;
            w = 0;
        }
        edge(int x, int y, int wt) : s(x), t(y), w(wt){};
};

class qreg{
    public:
        std::string qubitsName;
        int numOfQubits;
        qreg(){
        numOfQubits = 0; 
        }
        qreg(const std::string& qn, int n):numOfQubits(n){
            qubitsName = qn;
        }
};

class gate{
    public:
        int gateName;
        int type;
        int ctrl;
        int tar;
        std::string p1;
        std::string p2;
        std::string p3;
        gate(){
            type = 1;
            ctrl = 0;
            tar =0;
        }
        gate(int c, int t, int p, int g):type(p), ctrl(c), tar(t), gateName(g){
        }
};

bool fill(std::vector<int>& mark, int n);

int calDepth(std::vector<gate>& circuit, int n);

int countQubits(std::vector<gate>& circuit, int n);
}
#endif