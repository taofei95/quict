OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
rz(1.5707963267948966) q[2];
cu1(1.5707963267948966) q[11], q[5];
x q[9];
x q[14];
u2(1.5707963267948966, 1.5707963267948966) q[11];
t q[13];
sdg q[6];
h q[11];
crz(1.5707963267948966) q[3], q[9];
id q[11];
cz q[3], q[6];
rzz(1.5707963267948966) q[7], q[6];
u1(1.5707963267948966) q[14];
t q[16];
h q[7];
rz(1.5707963267948966) q[6];
s q[0];
rx(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[11];
sdg q[13];
crz(1.5707963267948966) q[12], q[17];
p(0) q[7];
id q[4];
rxx(0) q[6], q[14];
tdg q[6];
rz(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[4];
tdg q[11];
u2(1.5707963267948966, 1.5707963267948966) q[17];
swap q[2], q[0];
u3(0, 0, 1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[16];
s q[8];
rx(1.5707963267948966) q[16];
u2(1.5707963267948966, 1.5707963267948966) q[19];
u2(1.5707963267948966, 1.5707963267948966) q[19];
ryy(1.5707963267948966) q[16], q[1];
cz q[8], q[7];
rx(1.5707963267948966) q[0];
tdg q[9];
tdg q[13];
cu3(1.5707963267948966, 0, 0) q[12], q[13];
sdg q[12];
s q[1];
cz q[16], q[15];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[2];
t q[9];
x q[1];
cy q[5], q[8];
t q[18];
tdg q[4];
cx q[10], q[8];
tdg q[11];
sdg q[16];
rxx(0) q[11], q[3];
swap q[14], q[4];
u1(1.5707963267948966) q[4];
ch q[7], q[8];
u3(0, 0, 1.5707963267948966) q[12];
sdg q[17];
id q[18];
h q[18];
u2(1.5707963267948966, 1.5707963267948966) q[19];
t q[2];
cu3(1.5707963267948966, 0, 0) q[1], q[8];
ry(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u1(1.5707963267948966) q[8];
id q[13];
u2(1.5707963267948966, 1.5707963267948966) q[8];
s q[9];
u3(0, 0, 1.5707963267948966) q[6];
s q[18];
t q[19];
cu1(1.5707963267948966) q[6], q[15];
h q[6];
crz(1.5707963267948966) q[9], q[15];
sdg q[18];
h q[19];
sdg q[13];
rz(1.5707963267948966) q[15];
u3(0, 0, 1.5707963267948966) q[8];
tdg q[13];
u1(1.5707963267948966) q[14];
u1(1.5707963267948966) q[1];
cx q[6], q[16];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[5];
x q[0];
x q[12];
p(0) q[16];
id q[8];
u3(0, 0, 1.5707963267948966) q[17];
sdg q[14];
u2(1.5707963267948966, 1.5707963267948966) q[1];
sdg q[11];
u1(1.5707963267948966) q[7];
t q[7];
ry(1.5707963267948966) q[3];
sdg q[5];
rz(1.5707963267948966) q[11];
p(0) q[5];
u1(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[17];
h q[2];
rz(1.5707963267948966) q[8];
sdg q[13];
x q[7];
t q[19];
t q[14];
u2(1.5707963267948966, 1.5707963267948966) q[16];
rx(1.5707963267948966) q[13];
t q[9];
cz q[15], q[19];
cx q[3], q[6];
x q[9];
rz(1.5707963267948966) q[16];
tdg q[17];
t q[16];
t q[10];
s q[10];
ry(1.5707963267948966) q[18];
u1(1.5707963267948966) q[7];
sdg q[7];
ry(1.5707963267948966) q[1];
t q[9];
u1(1.5707963267948966) q[10];
p(0) q[15];
cx q[14], q[8];
x q[18];
s q[4];
ry(1.5707963267948966) q[0];
h q[16];
ch q[15], q[4];
s q[5];
ry(1.5707963267948966) q[11];
crz(1.5707963267948966) q[5], q[8];
tdg q[8];
h q[3];