OPENQASM 2.0;
include "qelib1.inc";
qreg q[15];
creg c[15];
tdg q[13];
u2(1.5707963267948966, 1.5707963267948966) q[6];
s q[4];
rz(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u1(1.5707963267948966) q[10];
id q[13];
p(0) q[7];
sdg q[7];
u1(1.5707963267948966) q[0];
u1(1.5707963267948966) q[4];
tdg q[4];
u2(1.5707963267948966, 1.5707963267948966) q[2];
sdg q[1];
cy q[11], q[5];
t q[7];
u1(1.5707963267948966) q[7];
rz(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[13];
u2(1.5707963267948966, 1.5707963267948966) q[8];
s q[13];
u2(1.5707963267948966, 1.5707963267948966) q[10];
cu3(1.5707963267948966, 0, 0) q[9], q[0];
id q[5];
rxx(0) q[0], q[5];
sdg q[2];
swap q[13], q[5];
s q[1];
u2(1.5707963267948966, 1.5707963267948966) q[3];
p(0) q[7];
u3(0, 0, 1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[6];
cy q[10], q[5];
u1(1.5707963267948966) q[7];
t q[13];
rxx(0) q[14], q[12];
ry(1.5707963267948966) q[3];
t q[1];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[10];
rx(1.5707963267948966) q[14];
h q[1];
cz q[4], q[11];
cx q[2], q[11];
rx(1.5707963267948966) q[9];
cy q[8], q[6];
ry(1.5707963267948966) q[13];
cx q[6], q[0];
rx(1.5707963267948966) q[14];
cy q[13], q[6];
cy q[7], q[14];
rz(1.5707963267948966) q[6];
h q[6];
rz(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[10];
u1(1.5707963267948966) q[3];
cu3(1.5707963267948966, 0, 0) q[5], q[0];
cx q[7], q[2];
id q[7];
u2(1.5707963267948966, 1.5707963267948966) q[11];
id q[13];
rx(1.5707963267948966) q[5];
rzz(1.5707963267948966) q[6], q[1];
u1(1.5707963267948966) q[13];
rz(1.5707963267948966) q[0];
t q[7];
sdg q[8];
p(0) q[14];
rx(1.5707963267948966) q[11];
rzz(1.5707963267948966) q[3], q[6];
id q[7];
sdg q[13];
tdg q[14];
id q[3];
rz(1.5707963267948966) q[11];
u1(1.5707963267948966) q[13];
rz(1.5707963267948966) q[7];
rxx(0) q[14], q[5];
h q[14];
t q[3];
h q[4];
u1(1.5707963267948966) q[10];
rxx(0) q[8], q[9];
u3(0, 0, 1.5707963267948966) q[14];
rx(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[8];
u1(1.5707963267948966) q[13];
cx q[13], q[0];
u2(1.5707963267948966, 1.5707963267948966) q[4];
cz q[3], q[0];
tdg q[7];
ry(1.5707963267948966) q[12];
rz(1.5707963267948966) q[14];
u1(1.5707963267948966) q[12];
tdg q[0];
rx(1.5707963267948966) q[10];
rz(1.5707963267948966) q[13];
p(0) q[12];
u3(0, 0, 1.5707963267948966) q[14];
ry(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[8];
rz(1.5707963267948966) q[13];
rx(1.5707963267948966) q[14];
ry(1.5707963267948966) q[4];
s q[6];
cu1(1.5707963267948966) q[12], q[6];
cu3(1.5707963267948966, 0, 0) q[12], q[11];
s q[10];
ry(1.5707963267948966) q[13];
rz(1.5707963267948966) q[7];
u1(1.5707963267948966) q[8];
tdg q[8];
s q[9];
u2(1.5707963267948966, 1.5707963267948966) q[2];
t q[13];
tdg q[8];
sdg q[10];
u1(1.5707963267948966) q[11];
u3(0, 0, 1.5707963267948966) q[12];
cu3(1.5707963267948966, 0, 0) q[13], q[5];
cu3(1.5707963267948966, 0, 0) q[8], q[10];
cx q[3], q[5];
u2(1.5707963267948966, 1.5707963267948966) q[6];
s q[14];
s q[0];
sdg q[13];
sdg q[12];
swap q[0], q[4];
swap q[9], q[12];
rz(1.5707963267948966) q[13];
s q[13];
rx(1.5707963267948966) q[9];
cu1(1.5707963267948966) q[1], q[0];
s q[0];
t q[1];
tdg q[4];
s q[13];
ry(1.5707963267948966) q[3];
cy q[3], q[6];
u2(1.5707963267948966, 1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[4];
cy q[13], q[12];
s q[7];
rxx(0) q[1], q[0];
cy q[2], q[14];
id q[0];
cu3(1.5707963267948966, 0, 0) q[3], q[2];
id q[2];
ry(1.5707963267948966) q[7];
tdg q[8];
u1(1.5707963267948966) q[5];
id q[5];
tdg q[11];
rz(1.5707963267948966) q[6];
rx(1.5707963267948966) q[6];
t q[0];
h q[9];
rz(1.5707963267948966) q[10];
cy q[10], q[1];
rz(1.5707963267948966) q[4];
t q[7];
h q[1];
u3(0, 0, 1.5707963267948966) q[12];
id q[12];
rzz(1.5707963267948966) q[14], q[4];