OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
ry(1.5707963267948966) q[3];
sdg q[6];
cx q[5], q[2];
x q[3];
p(0) q[3];
tdg q[9];
t q[2];
h q[7];
h q[0];
u3(0, 0, 1.5707963267948966) q[2];
ry(1.5707963267948966) q[2];
p(0) q[7];
h q[6];
ry(1.5707963267948966) q[5];
t q[9];
sdg q[7];
s q[1];
tdg q[2];
id q[3];
t q[8];
x q[6];
h q[7];
id q[1];
cu3(1.5707963267948966, 0, 0) q[6], q[0];
h q[3];
u3(0, 0, 1.5707963267948966) q[0];
rz(1.5707963267948966) q[4];
rx(1.5707963267948966) q[7];
rz(1.5707963267948966) q[7];
p(0) q[0];
swap q[5], q[9];
x q[5];
t q[8];
s q[6];
crz(1.5707963267948966) q[0], q[1];
u3(0, 0, 1.5707963267948966) q[7];
swap q[6], q[4];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[2];
t q[0];
tdg q[1];
tdg q[6];
tdg q[3];
tdg q[6];
p(0) q[7];
ryy(1.5707963267948966) q[2], q[0];
swap q[7], q[3];
rx(1.5707963267948966) q[3];
u1(1.5707963267948966) q[2];
cz q[4], q[1];
u2(1.5707963267948966, 1.5707963267948966) q[7];
ry(1.5707963267948966) q[4];
crz(1.5707963267948966) q[0], q[5];
p(0) q[9];
cu3(1.5707963267948966, 0, 0) q[7], q[4];
s q[7];
sdg q[9];
cy q[9], q[6];
p(0) q[0];
id q[8];
h q[3];
rz(1.5707963267948966) q[2];
t q[8];
id q[4];
p(0) q[2];
crz(1.5707963267948966) q[4], q[2];
sdg q[4];
tdg q[5];
h q[0];
x q[9];
p(0) q[7];
cy q[1], q[8];
id q[5];
swap q[1], q[5];
cz q[7], q[5];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[9];
x q[1];
crz(1.5707963267948966) q[1], q[2];
h q[8];
ch q[9], q[1];
rzz(1.5707963267948966) q[4], q[8];
cy q[6], q[0];
rx(1.5707963267948966) q[3];
x q[2];
rz(1.5707963267948966) q[1];
cx q[4], q[0];
rxx(0) q[2], q[6];
t q[7];