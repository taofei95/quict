OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c[9];
cu3(1.5707963267948966, 0, 0) q[2], q[6];
rx(1.5707963267948966) q[0];
h q[5];
s q[1];
id q[1];
rx(1.5707963267948966) q[3];
u1(1.5707963267948966) q[3];
u1(1.5707963267948966) q[8];
cu3(1.5707963267948966, 0, 0) q[1], q[5];
s q[5];
ry(1.5707963267948966) q[6];
sdg q[6];
ry(1.5707963267948966) q[3];
id q[0];
sdg q[8];
u3(0, 0, 1.5707963267948966) q[2];
rzz(1.5707963267948966) q[6], q[8];
h q[1];
cu1(1.5707963267948966) q[1], q[7];
t q[3];
tdg q[3];
u3(0, 0, 1.5707963267948966) q[6];
rzz(1.5707963267948966) q[2], q[0];
tdg q[2];
u2(1.5707963267948966, 1.5707963267948966) q[6];
rzz(1.5707963267948966) q[0], q[2];
u3(0, 0, 1.5707963267948966) q[3];
rxx(0) q[1], q[7];
cu3(1.5707963267948966, 0, 0) q[4], q[6];
ch q[2], q[5];
u1(1.5707963267948966) q[3];
rz(1.5707963267948966) q[0];
h q[6];
u3(0, 0, 1.5707963267948966) q[3];
s q[6];
p(0) q[4];
crz(1.5707963267948966) q[5], q[4];
s q[1];
rz(1.5707963267948966) q[1];
h q[8];
rx(1.5707963267948966) q[2];
sdg q[1];
tdg q[3];
h q[5];
cz q[8], q[1];
cy q[6], q[3];
rx(1.5707963267948966) q[8];
crz(1.5707963267948966) q[8], q[1];
cx q[5], q[2];
ry(1.5707963267948966) q[5];
rx(1.5707963267948966) q[2];
cz q[0], q[7];
u1(1.5707963267948966) q[3];
cy q[7], q[4];
u1(1.5707963267948966) q[8];
p(0) q[7];
cy q[0], q[4];
p(0) q[3];
ch q[1], q[5];
h q[7];
cz q[6], q[2];
u2(1.5707963267948966, 1.5707963267948966) q[7];
ry(1.5707963267948966) q[6];