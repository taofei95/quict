OPENQASM 2.0;
include "qelib1.inc";
qreg q[15];
creg c[15];
rx(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[5];
t q[4];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[4];
swap q[2], q[6];
cy q[10], q[11];
id q[8];
h q[6];
u1(1.5707963267948966) q[0];
u1(1.5707963267948966) q[10];
u3(0, 0, 1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[7];
h q[1];
s q[12];
t q[13];
cu3(1.5707963267948966, 0, 0) q[13], q[0];
u3(0, 0, 1.5707963267948966) q[12];
u1(1.5707963267948966) q[14];
h q[8];
rx(1.5707963267948966) q[1];
id q[6];
u2(1.5707963267948966, 1.5707963267948966) q[5];
cx q[5], q[10];
rxx(0) q[10], q[6];
rxx(0) q[13], q[2];
cu1(1.5707963267948966) q[11], q[5];
p(0) q[11];
sdg q[8];
u1(1.5707963267948966) q[8];
p(0) q[14];
h q[3];
p(0) q[7];
ry(1.5707963267948966) q[6];
rzz(1.5707963267948966) q[7], q[12];
sdg q[3];
u2(1.5707963267948966, 1.5707963267948966) q[0];
h q[6];
u2(1.5707963267948966, 1.5707963267948966) q[7];
ry(1.5707963267948966) q[11];
rx(1.5707963267948966) q[11];
t q[7];
tdg q[14];
tdg q[9];
h q[4];
s q[11];
h q[13];
h q[2];
rxx(0) q[2], q[3];
rx(1.5707963267948966) q[11];
ry(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[12];
u2(1.5707963267948966, 1.5707963267948966) q[12];
u2(1.5707963267948966, 1.5707963267948966) q[4];
tdg q[0];
u1(1.5707963267948966) q[4];
t q[9];
rz(1.5707963267948966) q[0];
id q[0];
p(0) q[0];
u1(1.5707963267948966) q[0];
sdg q[2];
u2(1.5707963267948966, 1.5707963267948966) q[11];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[12];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[3];
sdg q[0];
cu1(1.5707963267948966) q[1], q[9];
sdg q[12];
cu3(1.5707963267948966, 0, 0) q[14], q[12];
rxx(0) q[10], q[1];
u1(1.5707963267948966) q[13];