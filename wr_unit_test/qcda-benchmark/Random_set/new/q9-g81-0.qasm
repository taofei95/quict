OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c[9];
ry(1.5707963267948966) q[3];
id q[8];
s q[5];
u2(1.5707963267948966, 1.5707963267948966) q[0];
ry(1.5707963267948966) q[8];
sdg q[3];
cx q[0], q[1];
tdg q[5];
u3(0, 0, 1.5707963267948966) q[3];
ry(1.5707963267948966) q[5];
crz(1.5707963267948966) q[5], q[3];
t q[3];
tdg q[3];
t q[1];
x q[3];
s q[1];
u3(0, 0, 1.5707963267948966) q[8];
u1(1.5707963267948966) q[8];
t q[7];
tdg q[5];
u2(1.5707963267948966, 1.5707963267948966) q[0];
t q[2];
tdg q[0];
id q[4];
p(0) q[3];
ry(1.5707963267948966) q[3];
cx q[3], q[4];
cz q[5], q[2];
u1(1.5707963267948966) q[4];
p(0) q[5];
u3(0, 0, 1.5707963267948966) q[3];
sdg q[0];
t q[2];
u3(0, 0, 1.5707963267948966) q[5];
sdg q[8];
rxx(0) q[1], q[3];
rxx(0) q[8], q[2];
u1(1.5707963267948966) q[1];
u1(1.5707963267948966) q[8];
cx q[5], q[6];
cz q[8], q[3];
rx(1.5707963267948966) q[2];
rz(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[3];
t q[7];
p(0) q[6];
x q[6];
sdg q[5];
tdg q[2];
sdg q[8];
cx q[3], q[4];
s q[7];
sdg q[6];
x q[3];
rx(1.5707963267948966) q[0];
cu3(1.5707963267948966, 0, 0) q[5], q[6];
ch q[2], q[7];
s q[5];
crz(1.5707963267948966) q[1], q[5];
tdg q[0];
u1(1.5707963267948966) q[1];
tdg q[7];
t q[8];
u3(0, 0, 1.5707963267948966) q[1];
id q[5];
ry(1.5707963267948966) q[8];
ryy(1.5707963267948966) q[4], q[5];
h q[3];
rz(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[5];
cx q[1], q[4];
u1(1.5707963267948966) q[6];
tdg q[0];
rzz(1.5707963267948966) q[3], q[8];
cu3(1.5707963267948966, 0, 0) q[2], q[6];
u1(1.5707963267948966) q[3];
s q[8];
t q[5];
ch q[8], q[1];
rz(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[7];