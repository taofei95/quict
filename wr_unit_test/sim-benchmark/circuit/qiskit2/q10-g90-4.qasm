OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
rz(1.5707963267948966) q[5];
h q[9];
tdg q[4];
rz(1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[1];
s q[3];
ry(1.5707963267948966) q[8];
cu1(1.5707963267948966) q[7], q[2];
u3(0, 0, 1.5707963267948966) q[1];
rz(1.5707963267948966) q[1];
sdg q[4];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[4];
tdg q[8];
t q[1];
tdg q[9];
rxx(0) q[7], q[3];
sdg q[4];
cx q[8], q[1];
tdg q[4];
ry(1.5707963267948966) q[6];
tdg q[1];
u1(1.5707963267948966) q[9];
ry(1.5707963267948966) q[9];
s q[9];
p(0) q[4];
rzz(1.5707963267948966) q[6], q[7];
h q[5];
tdg q[2];
cx q[1], q[4];
s q[5];
ry(1.5707963267948966) q[5];
p(0) q[4];
h q[9];
u3(0, 0, 1.5707963267948966) q[7];
h q[4];
u3(0, 0, 1.5707963267948966) q[3];
ry(1.5707963267948966) q[3];
id q[6];
rx(1.5707963267948966) q[6];
u1(1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[7];
p(0) q[7];
id q[1];
swap q[0], q[9];
s q[0];
tdg q[5];
tdg q[8];
id q[0];
h q[7];
rz(1.5707963267948966) q[6];
ry(1.5707963267948966) q[8];
u1(1.5707963267948966) q[5];
p(0) q[3];
t q[0];
t q[5];
h q[3];
id q[5];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[5];
ry(1.5707963267948966) q[8];
t q[6];
s q[1];
rz(1.5707963267948966) q[7];
cy q[4], q[8];
h q[0];
s q[5];
u2(1.5707963267948966, 1.5707963267948966) q[1];
id q[2];
tdg q[3];
tdg q[8];
s q[2];
rz(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[7];
sdg q[2];
sdg q[5];
s q[0];
u3(0, 0, 1.5707963267948966) q[7];
p(0) q[7];
rx(1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[3];
cz q[2], q[9];
t q[5];
id q[8];
u3(0, 0, 1.5707963267948966) q[7];
ry(1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[7];
s q[6];
p(0) q[9];