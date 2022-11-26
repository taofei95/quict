OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
h q[7];
t q[2];
cu1(1.5707963267948966) q[4], q[2];
h q[9];
rx(1.5707963267948966) q[2];
rz(1.5707963267948966) q[8];
tdg q[3];
u2(1.5707963267948966, 1.5707963267948966) q[0];
swap q[2], q[5];
id q[1];
t q[9];
sdg q[0];
id q[1];
tdg q[0];
h q[8];
u2(1.5707963267948966, 1.5707963267948966) q[0];
rz(1.5707963267948966) q[9];
u1(1.5707963267948966) q[3];
p(0) q[1];
s q[5];
rx(1.5707963267948966) q[1];
cx q[3], q[0];
rz(1.5707963267948966) q[5];
cx q[6], q[2];
rzz(1.5707963267948966) q[5], q[7];
u3(0, 0, 1.5707963267948966) q[9];
tdg q[7];
id q[3];
ry(1.5707963267948966) q[5];
t q[7];
u3(0, 0, 1.5707963267948966) q[1];
u1(1.5707963267948966) q[1];
h q[0];
tdg q[2];
p(0) q[1];
t q[6];
u1(1.5707963267948966) q[3];
h q[5];
s q[3];
p(0) q[3];
cx q[5], q[3];
ry(1.5707963267948966) q[9];
cu1(1.5707963267948966) q[2], q[5];
t q[0];
rz(1.5707963267948966) q[9];
p(0) q[9];
tdg q[3];
id q[3];
s q[7];
cz q[4], q[2];
cz q[4], q[1];
sdg q[2];
u3(0, 0, 1.5707963267948966) q[1];
t q[7];
rz(1.5707963267948966) q[9];
cu3(1.5707963267948966, 0, 0) q[9], q[4];
rxx(0) q[8], q[6];
p(0) q[7];
s q[1];
tdg q[0];
tdg q[9];
id q[2];
id q[3];
u3(0, 0, 1.5707963267948966) q[8];
rzz(1.5707963267948966) q[4], q[3];
h q[1];
cu3(1.5707963267948966, 0, 0) q[9], q[3];
rx(1.5707963267948966) q[5];
id q[8];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u1(1.5707963267948966) q[3];
cu1(1.5707963267948966) q[7], q[3];
u3(0, 0, 1.5707963267948966) q[2];
cx q[7], q[0];
rx(1.5707963267948966) q[2];
t q[7];
s q[5];
rz(1.5707963267948966) q[7];
rx(1.5707963267948966) q[7];
cy q[6], q[0];
p(0) q[8];
rx(1.5707963267948966) q[3];
s q[5];
t q[3];
rzz(1.5707963267948966) q[6], q[3];
t q[6];
swap q[5], q[7];
u3(0, 0, 1.5707963267948966) q[4];
s q[3];
h q[0];