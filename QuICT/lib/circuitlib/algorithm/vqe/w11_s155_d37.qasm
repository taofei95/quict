OPENQASM 2.0;
include "qelib1.inc";
qreg q[11];
creg c[11];
x q[0];
x q[1];
x q[2];
x q[3];
x q[4];
sqiswap q[4], q[5];
rz(-1.0139395930373958) q[4];
rz(4.155532246627189) q[5];
sqiswap q[4], q[5];
rz(-3.141592653589793) q[5];
sqiswap q[3], q[4];
rz(-5.996964894978558) q[3];
rz(9.13855754856835) q[4];
sqiswap q[3], q[4];
rz(-3.141592653589793) q[4];
sqiswap q[5], q[6];
rz(-1.468172638010181) q[5];
rz(4.609765291599974) q[6];
sqiswap q[5], q[6];
rz(-3.141592653589793) q[6];
sqiswap q[2], q[3];
rz(-1.162823179376098) q[2];
rz(4.304415832965891) q[3];
sqiswap q[2], q[3];
rz(-3.141592653589793) q[3];
sqiswap q[4], q[5];
rz(-2.673534676110496) q[4];
rz(5.815127329700289) q[5];
sqiswap q[4], q[5];
rz(-3.141592653589793) q[5];
sqiswap q[6], q[7];
rz(-1.0989714255767782) q[6];
rz(4.240564079166571) q[7];
sqiswap q[6], q[7];
rz(-3.141592653589793) q[7];
sqiswap q[1], q[2];
rz(-5.251467864361957) q[1];
rz(8.39306051795175) q[2];
sqiswap q[1], q[2];
rz(-3.141592653589793) q[2];
sqiswap q[3], q[4];
rz(-0.5992465775547526) q[3];
rz(3.7408392311445455) q[4];
sqiswap q[3], q[4];
rz(-3.141592653589793) q[4];
sqiswap q[5], q[6];
rz(-0.8775172724112169) q[5];
rz(4.01910992600101) q[6];
sqiswap q[5], q[6];
rz(-3.141592653589793) q[6];
sqiswap q[7], q[8];
rz(-1.2063938928682516) q[7];
rz(4.347986546458045) q[8];
sqiswap q[7], q[8];
rz(-3.141592653589793) q[8];
sqiswap q[0], q[1];
rz(-0.37374307201900886) q[0];
rz(3.515335725608802) q[1];
sqiswap q[0], q[1];
rz(-3.141592653589793) q[1];
sqiswap q[2], q[3];
rz(-2.384461185003608) q[2];
rz(5.526053838593401) q[3];
sqiswap q[2], q[3];
rz(-3.141592653589793) q[3];
sqiswap q[4], q[5];
rz(-1.6141071912714577) q[4];
rz(4.755699844861251) q[5];
sqiswap q[4], q[5];
rz(-3.141592653589793) q[5];
sqiswap q[6], q[7];
rz(-3.6314487970404765) q[6];
rz(6.77304145063027) q[7];
sqiswap q[6], q[7];
rz(-3.141592653589793) q[7];
sqiswap q[8], q[9];
rz(-1.4726582626290265) q[8];
rz(4.61425091621882) q[9];
sqiswap q[8], q[9];
rz(-3.141592653589793) q[9];
sqiswap q[1], q[2];
rz(-4.917302168136951) q[1];
rz(8.058894821726744) q[2];
sqiswap q[1], q[2];
rz(-3.141592653589793) q[2];
sqiswap q[3], q[4];
rz(-4.055566685893902) q[3];
rz(7.1971593394836955) q[4];
sqiswap q[3], q[4];
rz(-3.141592653589793) q[4];
sqiswap q[5], q[6];
rz(-2.2288608234870235) q[5];
rz(5.370453477076817) q[6];
sqiswap q[5], q[6];
rz(-3.141592653589793) q[6];
sqiswap q[7], q[8];
rz(-0.2838559081714709) q[7];
rz(3.425448561761264) q[8];
sqiswap q[7], q[8];
rz(-3.141592653589793) q[8];
sqiswap q[9], q[10];
rz(-5.008671829920936) q[9];
rz(8.150264483510728) q[10];
sqiswap q[9], q[10];
rz(-3.141592653589793) q[10];
sqiswap q[2], q[3];
rz(-6.266862088459728) q[2];
rz(9.408454742049521) q[3];
sqiswap q[2], q[3];
rz(-3.141592653589793) q[3];
sqiswap q[4], q[5];
rz(-1.6057970834053283) q[4];
rz(4.747389736995121) q[5];
sqiswap q[4], q[5];
rz(-3.141592653589793) q[5];
sqiswap q[6], q[7];
rz(-2.5531583190139653) q[6];
rz(5.694750972603758) q[7];
sqiswap q[6], q[7];
rz(-3.141592653589793) q[7];
sqiswap q[8], q[9];
rz(-0.33525207550980146) q[8];
rz(3.4768447290995947) q[9];
sqiswap q[8], q[9];
rz(-3.141592653589793) q[9];
sqiswap q[3], q[4];
rz(-3.8101209379870005) q[3];
rz(6.951713591576794) q[4];
sqiswap q[3], q[4];
rz(-3.141592653589793) q[4];
sqiswap q[5], q[6];
rz(-2.0096406684531996) q[5];
rz(5.151233322042993) q[6];
sqiswap q[5], q[6];
rz(-3.141592653589793) q[6];
sqiswap q[7], q[8];
rz(-3.76594979866251) q[7];
rz(6.907542452252303) q[8];
sqiswap q[7], q[8];
rz(-3.141592653589793) q[8];
sqiswap q[4], q[5];
rz(-1.689212641088195) q[4];
rz(4.8308052946779885) q[5];
sqiswap q[4], q[5];
rz(-3.141592653589793) q[5];
sqiswap q[6], q[7];
rz(-1.4121873743793665) q[6];
rz(4.553780027969159) q[7];
sqiswap q[6], q[7];
rz(-3.141592653589793) q[7];
sqiswap q[5], q[6];
rz(-2.1765582477252505) q[5];
rz(5.318150901315043) q[6];
sqiswap q[5], q[6];
rz(-3.141592653589793) q[6];
