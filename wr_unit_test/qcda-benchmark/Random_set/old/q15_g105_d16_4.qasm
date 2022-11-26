OPENQASM 2.0;
include "qelib1.inc";
qreg q[15];
creg c[15];
tdg q[2];
cy q[12], q[11];
rz(1.5707963267948966) q[9];
sdg q[8];
cu3(1.5707963267948966, 0, 0) q[14], q[7];
rz(1.5707963267948966) q[2];
x q[8];
u2(1.5707963267948966, 1.5707963267948966) q[6];
id q[10];
u2(1.5707963267948966, 1.5707963267948966) q[6];
sdg q[1];
u2(1.5707963267948966, 1.5707963267948966) q[9];
crz(1.5707963267948966) q[12], q[3];
x q[10];
u3(0, 0, 1.5707963267948966) q[3];
rx(1.5707963267948966) q[10];
h q[13];
h q[0];
sdg q[14];
tdg q[3];
t q[9];
u3(0, 0, 1.5707963267948966) q[3];
t q[3];
crz(1.5707963267948966) q[12], q[13];
u1(1.5707963267948966) q[9];
h q[6];
u2(1.5707963267948966, 1.5707963267948966) q[7];
cu3(1.5707963267948966, 0, 0) q[10], q[2];
tdg q[14];
x q[10];
cu3(1.5707963267948966, 0, 0) q[1], q[11];
id q[5];
rx(1.5707963267948966) q[7];
p(0) q[0];
x q[14];
x q[13];
cz q[13], q[9];
s q[13];
ry(1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[6];
t q[7];
id q[8];
ry(1.5707963267948966) q[9];
cy q[4], q[8];
p(0) q[7];
s q[3];
cz q[7], q[3];
cu3(1.5707963267948966, 0, 0) q[0], q[8];
x q[6];
cx q[10], q[0];
crz(1.5707963267948966) q[0], q[9];
cu1(1.5707963267948966) q[14], q[4];
s q[9];
swap q[8], q[13];
u3(0, 0, 1.5707963267948966) q[13];
x q[4];
u2(1.5707963267948966, 1.5707963267948966) q[8];
ry(1.5707963267948966) q[11];
h q[9];
id q[5];
t q[8];
u1(1.5707963267948966) q[10];
id q[8];
id q[12];
x q[3];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[6];
id q[6];
u1(1.5707963267948966) q[14];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[5];
tdg q[7];
cx q[14], q[3];
u3(0, 0, 1.5707963267948966) q[3];
x q[13];
u1(1.5707963267948966) q[8];
sdg q[12];
t q[10];
cz q[6], q[3];
p(0) q[2];
rz(1.5707963267948966) q[3];
t q[5];
u3(0, 0, 1.5707963267948966) q[0];
cz q[9], q[8];
h q[6];
cu1(1.5707963267948966) q[12], q[3];
t q[8];
id q[5];
p(0) q[14];
sdg q[0];
rzz(1.5707963267948966) q[13], q[0];
u2(1.5707963267948966, 1.5707963267948966) q[8];
p(0) q[0];
rz(1.5707963267948966) q[9];
p(0) q[10];
h q[12];
x q[13];
h q[9];
cy q[8], q[7];
tdg q[3];
t q[2];
sdg q[0];
x q[14];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[7];