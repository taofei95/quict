OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
cz q[3], q[0];
rz(1.5707963267948966) q[1];
s q[7];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[7];
rx(1.5707963267948966) q[1];
cx q[5], q[1];
rxx(0) q[6], q[0];
id q[5];
rx(1.5707963267948966) q[0];
cz q[7], q[4];
ch q[6], q[1];
u3(0, 0, 1.5707963267948966) q[7];
h q[7];
t q[8];
u1(1.5707963267948966) q[8];
tdg q[9];
t q[3];
cu3(1.5707963267948966, 0, 0) q[4], q[9];
rx(1.5707963267948966) q[7];
s q[8];
u1(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[4];
cz q[4], q[8];
id q[7];
s q[1];
ry(1.5707963267948966) q[1];
tdg q[1];
sdg q[2];
rzz(1.5707963267948966) q[8], q[6];
rx(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[8];
ch q[3], q[2];
cy q[8], q[6];
rx(1.5707963267948966) q[7];
s q[1];
h q[9];
ry(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rz(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[7];
sdg q[9];
s q[1];
p(0) q[4];
u3(0, 0, 1.5707963267948966) q[6];
id q[1];
s q[3];
u2(1.5707963267948966, 1.5707963267948966) q[2];