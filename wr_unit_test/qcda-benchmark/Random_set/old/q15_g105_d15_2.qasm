OPENQASM 2.0;
include "qelib1.inc";
qreg q[15];
creg c[15];
swap q[5], q[14];
cz q[10], q[7];
cz q[2], q[11];
u2(1.5707963267948966, 1.5707963267948966) q[13];
u2(1.5707963267948966, 1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[14];
rx(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[5];
ry(1.5707963267948966) q[14];
tdg q[6];
ry(1.5707963267948966) q[9];
p(0) q[8];
rx(1.5707963267948966) q[2];
cy q[10], q[0];
rz(1.5707963267948966) q[6];
h q[2];
u1(1.5707963267948966) q[8];
p(0) q[8];
ryy(1.5707963267948966) q[13], q[11];
t q[14];
tdg q[0];
tdg q[9];
u2(1.5707963267948966, 1.5707963267948966) q[2];
t q[1];
h q[4];
u1(1.5707963267948966) q[13];
swap q[14], q[10];
t q[12];
cx q[3], q[5];
p(0) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[3];
sdg q[7];
swap q[4], q[11];
t q[8];
s q[11];
id q[3];
rx(1.5707963267948966) q[9];
p(0) q[4];
ryy(1.5707963267948966) q[6], q[14];
cz q[1], q[5];
id q[10];
u1(1.5707963267948966) q[14];
tdg q[12];
p(0) q[1];
ry(1.5707963267948966) q[1];
h q[4];
s q[2];
ry(1.5707963267948966) q[1];
tdg q[6];
rz(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[8];
sdg q[8];
ry(1.5707963267948966) q[9];
t q[8];
u2(1.5707963267948966, 1.5707963267948966) q[12];
ch q[2], q[8];
ry(1.5707963267948966) q[6];
id q[8];
t q[1];
id q[1];
ryy(1.5707963267948966) q[0], q[12];
x q[7];
tdg q[10];
p(0) q[10];
u1(1.5707963267948966) q[4];
s q[9];
t q[7];
u1(1.5707963267948966) q[12];
u1(1.5707963267948966) q[5];
ry(1.5707963267948966) q[6];
id q[13];
rzz(1.5707963267948966) q[3], q[10];
rz(1.5707963267948966) q[12];
h q[12];
u1(1.5707963267948966) q[8];
rxx(0) q[9], q[2];
ryy(1.5707963267948966) q[10], q[11];
u1(1.5707963267948966) q[3];
u1(1.5707963267948966) q[5];
cu1(1.5707963267948966) q[12], q[14];
u1(1.5707963267948966) q[6];
s q[2];
u1(1.5707963267948966) q[7];
u1(1.5707963267948966) q[14];
cx q[11], q[6];
u3(0, 0, 1.5707963267948966) q[10];
t q[0];
rxx(0) q[11], q[13];
h q[1];
u2(1.5707963267948966, 1.5707963267948966) q[13];
id q[12];
u1(1.5707963267948966) q[8];
x q[8];
sdg q[5];
rzz(1.5707963267948966) q[6], q[11];
id q[8];
rx(1.5707963267948966) q[11];
ry(1.5707963267948966) q[10];
h q[7];
rz(1.5707963267948966) q[8];
rx(1.5707963267948966) q[2];
x q[4];
t q[1];
cy q[10], q[12];