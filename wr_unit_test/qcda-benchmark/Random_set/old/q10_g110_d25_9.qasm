OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
ry(1.5707963267948966) q[0];
h q[6];
tdg q[5];
tdg q[4];
ry(1.5707963267948966) q[5];
sdg q[6];
t q[4];
cz q[5], q[8];
id q[3];
cx q[9], q[4];
u2(1.5707963267948966, 1.5707963267948966) q[2];
h q[5];
u1(1.5707963267948966) q[7];
x q[8];
p(0) q[1];
rz(1.5707963267948966) q[7];
h q[4];
rxx(0) q[9], q[4];
sdg q[3];
u1(1.5707963267948966) q[4];
s q[1];
ry(1.5707963267948966) q[7];
sdg q[5];
ry(1.5707963267948966) q[9];
sdg q[6];
u2(1.5707963267948966, 1.5707963267948966) q[2];
s q[2];
sdg q[9];
s q[6];
x q[4];
u1(1.5707963267948966) q[9];
x q[1];
cx q[6], q[8];
id q[7];
cu1(1.5707963267948966) q[7], q[4];
h q[0];
crz(1.5707963267948966) q[8], q[4];
u1(1.5707963267948966) q[1];
p(0) q[4];
sdg q[1];
u3(0, 0, 1.5707963267948966) q[5];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[6];
x q[1];
cu1(1.5707963267948966) q[6], q[8];
t q[6];
rx(1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[5];
cu1(1.5707963267948966) q[9], q[4];
tdg q[3];
cy q[0], q[9];
rx(1.5707963267948966) q[8];
h q[3];
u3(0, 0, 1.5707963267948966) q[3];
h q[1];
h q[0];
h q[6];
u2(1.5707963267948966, 1.5707963267948966) q[3];
h q[5];
h q[0];
rz(1.5707963267948966) q[2];
id q[9];
sdg q[8];
rx(1.5707963267948966) q[3];
rxx(0) q[9], q[5];
s q[0];
ry(1.5707963267948966) q[0];
ch q[4], q[7];
u1(1.5707963267948966) q[8];
rx(1.5707963267948966) q[7];
rz(1.5707963267948966) q[3];
t q[4];
sdg q[8];
rxx(0) q[4], q[5];
id q[0];
p(0) q[5];
id q[8];
p(0) q[6];
tdg q[5];
rzz(1.5707963267948966) q[8], q[9];
cx q[8], q[9];
s q[5];
p(0) q[6];
rx(1.5707963267948966) q[2];
h q[8];
s q[3];
id q[1];
rz(1.5707963267948966) q[7];
sdg q[0];
ryy(1.5707963267948966) q[9], q[5];
rzz(1.5707963267948966) q[2], q[5];
p(0) q[6];
x q[2];
s q[5];
id q[1];
id q[9];
tdg q[6];
swap q[1], q[2];
rx(1.5707963267948966) q[5];
t q[4];
h q[0];
u1(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[0];
rz(1.5707963267948966) q[8];
tdg q[4];
t q[2];
u3(0, 0, 1.5707963267948966) q[2];
crz(1.5707963267948966) q[9], q[3];
s q[6];
rz(1.5707963267948966) q[2];