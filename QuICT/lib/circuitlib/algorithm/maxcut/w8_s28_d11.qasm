OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
cx q[3], q[1];
rz(-1.178985834121704) q[1];
cx q[3], q[1];
cx q[4], q[7];
rz(-1.178985834121704) q[7];
cx q[4], q[7];
cx q[6], q[1];
rz(-1.178985834121704) q[1];
cx q[6], q[1];
cx q[4], q[6];
rz(-1.178985834121704) q[6];
cx q[4], q[6];
rx(0.49825018644332886) q[0];
rx(0.49825018644332886) q[1];
rx(0.49825018644332886) q[2];
rx(0.49825018644332886) q[3];
rx(0.49825018644332886) q[4];
rx(0.49825018644332886) q[5];
rx(0.49825018644332886) q[6];
rx(0.49825018644332886) q[7];