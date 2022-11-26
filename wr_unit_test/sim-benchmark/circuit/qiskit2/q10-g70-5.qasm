OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
cu1(1.5707963267948966) q[4], q[3];
u3(0, 0, 1.5707963267948966) q[5];
cu1(1.5707963267948966) q[9], q[6];
rzz(1.5707963267948966) q[3], q[0];
t q[9];
cx q[8], q[1];
u2(1.5707963267948966, 1.5707963267948966) q[1];
cu1(1.5707963267948966) q[6], q[5];
h q[7];
cy q[6], q[5];
u1(1.5707963267948966) q[9];
p(0) q[4];
t q[3];
p(0) q[6];
rz(1.5707963267948966) q[6];
rzz(1.5707963267948966) q[9], q[6];
sdg q[3];
tdg q[5];
u2(1.5707963267948966, 1.5707963267948966) q[7];
id q[4];
id q[9];
h q[1];
cx q[5], q[8];
s q[7];
u1(1.5707963267948966) q[4];
sdg q[1];
tdg q[7];
u3(0, 0, 1.5707963267948966) q[0];
p(0) q[6];
h q[9];
u1(1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[4];
p(0) q[3];
p(0) q[5];
id q[1];
rx(1.5707963267948966) q[4];
cz q[4], q[0];
rz(1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[8];
s q[7];
cz q[9], q[2];
u3(0, 0, 1.5707963267948966) q[7];
id q[1];
ry(1.5707963267948966) q[7];
ry(1.5707963267948966) q[3];
u1(1.5707963267948966) q[8];
swap q[9], q[7];
rx(1.5707963267948966) q[3];
sdg q[1];
u2(1.5707963267948966, 1.5707963267948966) q[6];
tdg q[4];
cz q[4], q[1];
h q[2];
cx q[9], q[7];
ry(1.5707963267948966) q[8];
rz(1.5707963267948966) q[8];
h q[9];
u3(0, 0, 1.5707963267948966) q[3];
id q[0];
u1(1.5707963267948966) q[6];
ry(1.5707963267948966) q[5];
tdg q[4];
rzz(1.5707963267948966) q[7], q[2];
rx(1.5707963267948966) q[4];
h q[6];
ry(1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[5];
h q[0];
rz(1.5707963267948966) q[2];