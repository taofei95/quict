OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
creg c[16];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];
h q[15];
cx q[2], q[9];
rz(-0.20676590502262115) q[9];
cx q[2], q[9];
cx q[12], q[7];
rz(-0.20676590502262115) q[7];
cx q[12], q[7];
cx q[13], q[5];
rz(-0.20676590502262115) q[5];
cx q[13], q[5];
cx q[11], q[9];
rz(-0.20676590502262115) q[9];
cx q[11], q[9];
rx(0.4207802712917328) q[0];
rx(0.4207802712917328) q[1];
rx(0.4207802712917328) q[2];
rx(0.4207802712917328) q[3];
rx(0.4207802712917328) q[4];
rx(0.4207802712917328) q[5];
rx(0.4207802712917328) q[6];
rx(0.4207802712917328) q[7];
rx(0.4207802712917328) q[8];
rx(0.4207802712917328) q[9];
rx(0.4207802712917328) q[10];
rx(0.4207802712917328) q[11];
rx(0.4207802712917328) q[12];
rx(0.4207802712917328) q[13];
rx(0.4207802712917328) q[14];
rx(0.4207802712917328) q[15];