OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
x q[0];
x q[1];
sqiswap q[1], q[2];
rz(-5.306324545630215) q[1];
rz(8.447917199220008) q[2];
sqiswap q[1], q[2];
rz(-3.141592653589793) q[2];
sqiswap q[0], q[1];
rz(-0.12244223915941913) q[0];
rz(3.2640348927492124) q[1];
sqiswap q[0], q[1];
rz(-3.141592653589793) q[1];
sqiswap q[2], q[3];
rz(-2.801275123689416) q[2];
rz(5.94286777727921) q[3];
sqiswap q[2], q[3];
rz(-3.141592653589793) q[3];
sqiswap q[1], q[2];
rz(-3.434851296528033) q[1];
rz(6.576443950117826) q[2];
sqiswap q[1], q[2];
rz(-3.141592653589793) q[2];
