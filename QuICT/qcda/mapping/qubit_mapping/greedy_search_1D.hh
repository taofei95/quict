#ifndef  GREEDY_SEARCH_1D
#define  GREEDY_SEARCH_1D

#define SIFT

#include<iostream>
#include<limits.h>
#include<vector>
#include<unordered_map>
#include<algorithm>
#include<stdlib.h>
#include<ctime>
#include<exception>
#include "utility.hh"



#ifdef  __REVLIB__
    #include "revLib.hh"
#endif

#ifdef  __QASM__
    #include "qasm.hh"
#endif


namespace mapping{


typedef bool (*compare)(const int &, const int &);

typedef  std::vector<std::vector<std::vector<int>>>  ConstraintMatrix;
typedef  std::vector<std::vector<int>>  ConstraintVector;
typedef  std::vector<int>  Constraint;

typedef std::vector<utility::gate> Circuit;

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

int countSubConstraintInversion(ConstraintVector &constraint, std::vector<int> &disF, int s, int t);

int countConstraintInversion(ConstraintVector &constraint, std::vector<int> &initMapping, std::vector<int> &permutation);

std::vector<utility::edge> constructGraph(Circuit &circuit, int n);
int calLA(std::vector<utility::edge> &graph, std::vector<int> &permuatation);
std::vector<int> minLA(std::vector<utility::edge> &graph, int n);
std::vector<int> findInitMapping(Circuit &circuit, int n);

int findLargestMove(std::vector<int> &permutation, std::vector<int> &direction);

std::vector<int> enumerate(std::vector<int> &initMapping, ConstraintVector &constraint);


// 用启发式算法找满足约束的permutation
std::vector<int> searchMapping(std::vector<int>& initMapping, std::vector<std::vector<int>>& constraint);


// 输出两个permutation之间准换所需的最小swap门序列
Circuit mappingTrans(std::vector<int>& initMapping, std::vector<int>& tarMapping);

//在约束中插入一个门
void insertConstraint(std::vector<int>& constraint, int connect, int hang);
//合并两个约束
void mergeConstraint(std::vector<int>& constraint1, std::vector<int>& constraint2, int c1, int c2);

// 找到utility::gates中从pos开始最大可映射的门集合，以约束的形式返回

ConstraintVector findConstraint(Circuit& gates, int& pos, int n);
Circuit logicToPhysics(Circuit& gates, std::vector<int>& mapping, int start, int end);
Circuit greedySearch(Circuit& gates, std::vector<int>& mapping, int n, const std::string& initMethod, const std::string& searchMethod);


#ifdef SIFT
ConstraintMatrix partitionGateSets(Circuit& circuit, std::vector<int>& partitionPoint, int n);

bool cmp(const std::vector<int>& a, const std::vector<int>& b);

ConstraintVector constructIndex(ConstraintMatrix& constraints, int n);

void fillMapping(ConstraintVector& constraint, std::vector<int>& mapping, int n);

std::vector<std::vector<int>>  constructInterMapping(ConstraintMatrix& constraints, int n);

void  swapConstraint(ConstraintVector& constraint, int i, int j);

int countInversions(std::vector<int>& curMapping, std::vector<int>& tarMapping);

std::vector<std::vector<int>> initializeInterMapping(ConstraintMatrix &constraints, std::vector<int> &initMapping,int n);

void innerConstraintDirection(ConstraintVector &constraint, std::vector<int> &initMapping, int n);

int sifting(ConstraintMatrix& constraints, std::vector<std::vector<int>>& list, 
            std::vector<std::vector<int>>& indexDict,std::vector<std::vector<int>>& mappingList, int ind, int n);

Circuit globalSifting(Circuit& circuit, std::vector<int>& mapping ,int n);


#endif
}

#endif