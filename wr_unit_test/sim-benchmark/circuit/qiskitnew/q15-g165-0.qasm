OPENQASM 2.0;
include "qelib1.inc";
qreg q[15];
creg c[15];
rz(1.5707963267948966) q[12];
ry(1.5707963267948966) q[5];
id q[2];
h q[3];
rx(1.5707963267948966) q[12];
tdg q[5];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[0];
t q[7];
cz q[10], q[5];
h q[13];
u3(0, 0, 1.5707963267948966) q[13];
h q[10];
h q[5];
h q[3];
rz(1.5707963267948966) q[14];
h q[12];
cu3(1.5707963267948966, 0, 0) q[4], q[14];
rx(1.5707963267948966) q[1];
s q[4];
rxx(0) q[5], q[11];
rzz(1.5707963267948966) q[0], q[3];
id q[5];
rx(1.5707963267948966) q[0];
rzz(1.5707963267948966) q[1], q[14];
p(0) q[6];
p(0) q[11];
cy q[3], q[10];
h q[10];
u1(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[4];
id q[2];
u1(1.5707963267948966) q[5];
cz q[8], q[4];
t q[10];
cx q[13], q[6];
rxx(0) q[0], q[1];
cy q[8], q[3];
sdg q[4];
t q[5];
tdg q[3];
rxx(0) q[14], q[2];
rx(1.5707963267948966) q[13];
rx(1.5707963267948966) q[4];
id q[11];
u1(1.5707963267948966) q[11];
u3(0, 0, 1.5707963267948966) q[1];
id q[0];
p(0) q[12];
rx(1.5707963267948966) q[11];
cy q[4], q[14];
u2(1.5707963267948966, 1.5707963267948966) q[7];
t q[1];
rzz(1.5707963267948966) q[3], q[8];
sdg q[1];
s q[11];
ry(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[8];
rz(1.5707963267948966) q[0];
cu1(1.5707963267948966) q[11], q[2];
sdg q[14];
u3(0, 0, 1.5707963267948966) q[1];
t q[8];
sdg q[9];
t q[12];
ry(1.5707963267948966) q[9];
cy q[0], q[13];
p(0) q[4];
cu1(1.5707963267948966) q[3], q[11];
h q[4];
rx(1.5707963267948966) q[9];
rx(1.5707963267948966) q[1];
h q[2];
t q[3];
h q[13];
sdg q[1];
u2(1.5707963267948966, 1.5707963267948966) q[4];
cz q[12], q[8];
ry(1.5707963267948966) q[13];
ry(1.5707963267948966) q[12];
cx q[1], q[4];
cz q[6], q[5];
cy q[0], q[2];
rx(1.5707963267948966) q[3];
u1(1.5707963267948966) q[5];
cz q[4], q[13];
t q[12];
u2(1.5707963267948966, 1.5707963267948966) q[5];
cy q[7], q[9];
s q[5];
swap q[0], q[7];
ry(1.5707963267948966) q[10];
u1(1.5707963267948966) q[8];
sdg q[11];
u2(1.5707963267948966, 1.5707963267948966) q[13];
h q[9];
tdg q[10];
sdg q[0];
u3(0, 0, 1.5707963267948966) q[8];
rz(1.5707963267948966) q[12];
u3(0, 0, 1.5707963267948966) q[7];
cz q[10], q[2];
cx q[8], q[9];
h q[14];
sdg q[12];
s q[10];
u3(0, 0, 1.5707963267948966) q[8];
cz q[1], q[4];
t q[2];
u1(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[4];
cz q[7], q[12];
tdg q[0];
cz q[14], q[9];
t q[10];
p(0) q[0];
tdg q[1];
tdg q[13];
sdg q[2];
u1(1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[9];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[6];
swap q[13], q[9];
id q[12];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[12];
u3(0, 0, 1.5707963267948966) q[10];
p(0) q[2];
s q[8];
tdg q[3];
cy q[7], q[0];
rx(1.5707963267948966) q[5];
s q[1];
u3(0, 0, 1.5707963267948966) q[6];
rz(1.5707963267948966) q[4];
rx(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[6];
rx(1.5707963267948966) q[14];
u3(0, 0, 1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[6];
cu1(1.5707963267948966) q[8], q[4];
tdg q[4];
id q[14];
sdg q[8];
ry(1.5707963267948966) q[8];
h q[14];
h q[14];
rz(1.5707963267948966) q[13];
ry(1.5707963267948966) q[12];
s q[7];
u2(1.5707963267948966, 1.5707963267948966) q[10];
id q[5];
id q[6];
tdg q[3];
u2(1.5707963267948966, 1.5707963267948966) q[1];
p(0) q[5];
s q[3];
t q[9];
cu3(1.5707963267948966, 0, 0) q[1], q[7];
rz(1.5707963267948966) q[5];
p(0) q[12];
h q[5];
t q[7];
h q[5];