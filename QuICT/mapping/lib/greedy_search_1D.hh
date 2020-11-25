#ifndef  GREEDY_SEARCH_1D
#define  GREEDY_SEARCH_1D

//#define __QASM__
#include<limits.h>
#include<vector>
#include<unordered_map>
#include<algorithm>
#include<stdlib.h>
#include<ctime>
#include "utility.hh"

#ifdef  __REVLIB__
    #include "revLib.hh"
#endif

#ifdef  __QASM__
    #include "qasm.hh"
#endif


namespace mapping{

typedef bool (*compare)(const int &, const int &);

class s_cmp{
public:
    bool operator()(const std::pair<int,float>& a, const std::pair<int,float>& b);
};


void swap(int& a, int& b);


bool L(const int& a, const int &b);
bool G(const int& a, const int &b);
//暴力算逆序对的个数
int countInversions(std::vector<int>& disF, std::vector<int>& constraint, compare func );

//将均值作为得分
float byCenter(std::vector<int>& disF, std::vector<int>& constraint);

//中位数作为得分
float median(std::vector<int>& disF, std::vector<int>&  constraint);

int countSubConstraintInversion(std::vector<std::vector<int>> &constraint, std::vector<int> &disF, int s, int t);

int countConstraintInversion(std::vector<std::vector<int>> &constraint, std::vector<int> &initMapping, std::vector<int> &permutation);

std::vector<utility::edge> constructGraph(std::vector<utility::gate> &circuit, int n);
int calLA(std::vector<utility::edge> &graph, std::vector<int> &permuatation);
std::vector<int> minLA(std::vector<utility::edge> &graph, int n);
std::vector<int> findInitMapping(std::vector<utility::gate> &circuit, int n);

int findLargestMove(std::vector<int> &permutation, std::vector<int> &direction);

std::vector<int> enumerate(std::vector<int> &initMapping, std::vector<std::vector<int>> &constraint);


// 用启发式算法找满足约束的permutation
std::vector<int> searchMapping(std::vector<int>& initMapping, std::vector<std::vector<int>>& constraints);


// 输出两个permutation之间准换所需的最小swap门序列
std::vector<utility::gate> mappingTrans(std::vector<int>& initMapping, std::vector<int>& tarMapping);

//在约束中插入一个门
void insertConstraint(std::vector<int>& constraint, int connect, int hang);
//合并两个约束
void mergeConstraint(std::vector<int>& constraint1, std::vector<int>& constraint2, int c1, int c2);

// 找到utility::gates中从pos开始最大可映射的门集合，以约束的形式返回

std::vector<std::vector<int>> findConstraint(std::vector<utility::gate>& gates, int& pos, int n);
std::vector<utility::gate> logicToPhysics(std::vector<utility::gate>& gates, std::vector<int>& mapping, int start, int end);
std::vector<utility::gate> greedySearch(std::vector<utility::gate>& gates, std::vector<int>& mapping, int n);


}
#endif