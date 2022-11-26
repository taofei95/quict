OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
id q[13];
cu1(1.5707963267948966) q[5], q[13];
t q[7];
rz(1.5707963267948966) q[14];
id q[12];
u2(1.5707963267948966, 1.5707963267948966) q[16];
ry(1.5707963267948966) q[11];
sdg q[6];
u3(0, 0, 1.5707963267948966) q[13];
cu3(1.5707963267948966, 0, 0) q[6], q[2];
h q[1];
cz q[7], q[16];
u1(1.5707963267948966) q[4];
t q[3];
u1(1.5707963267948966) q[13];
cz q[10], q[12];
s q[3];
rxx(0) q[15], q[16];
h q[17];
id q[1];
t q[2];
tdg q[10];
cy q[0], q[5];
p(0) q[18];
u3(0, 0, 1.5707963267948966) q[13];
u3(0, 0, 1.5707963267948966) q[18];
p(0) q[7];
u1(1.5707963267948966) q[0];
id q[18];
u3(0, 0, 1.5707963267948966) q[7];
t q[12];
h q[16];
t q[8];
u2(1.5707963267948966, 1.5707963267948966) q[17];
sdg q[19];
u3(0, 0, 1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[1];
sdg q[19];
tdg q[4];
ry(1.5707963267948966) q[16];
p(0) q[8];
rz(1.5707963267948966) q[18];
swap q[19], q[0];
u3(0, 0, 1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[15];
u1(1.5707963267948966) q[15];
id q[0];
swap q[17], q[1];
u1(1.5707963267948966) q[12];
rx(1.5707963267948966) q[11];
s q[19];
s q[16];
p(0) q[3];
t q[2];
tdg q[12];
cz q[2], q[4];
t q[5];
u2(1.5707963267948966, 1.5707963267948966) q[5];
h q[19];
tdg q[12];
h q[8];
h q[19];
rxx(0) q[8], q[4];
cz q[7], q[6];
rz(1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[12];
cx q[3], q[1];
u3(0, 0, 1.5707963267948966) q[10];
ry(1.5707963267948966) q[12];
rz(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[1];
ry(1.5707963267948966) q[5];
rx(1.5707963267948966) q[16];
id q[10];
cu3(1.5707963267948966, 0, 0) q[0], q[2];
rz(1.5707963267948966) q[18];
u2(1.5707963267948966, 1.5707963267948966) q[1];
cy q[9], q[16];
tdg q[19];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[7];
t q[3];
s q[3];
rz(1.5707963267948966) q[9];
ry(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[6];
id q[7];
cu1(1.5707963267948966) q[2], q[10];
u2(1.5707963267948966, 1.5707963267948966) q[16];
tdg q[10];
h q[19];
tdg q[2];
rx(1.5707963267948966) q[18];
s q[14];
rx(1.5707963267948966) q[10];
t q[18];
p(0) q[1];
p(0) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[11];
ry(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[11];
rz(1.5707963267948966) q[2];
tdg q[0];
t q[7];
h q[5];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[5];
rxx(0) q[6], q[17];
id q[7];
rzz(1.5707963267948966) q[9], q[0];
cz q[6], q[9];
tdg q[0];
u2(1.5707963267948966, 1.5707963267948966) q[1];
cz q[19], q[17];
rx(1.5707963267948966) q[19];
rx(1.5707963267948966) q[15];
tdg q[11];
cx q[15], q[18];
ry(1.5707963267948966) q[17];
u2(1.5707963267948966, 1.5707963267948966) q[2];
h q[15];
s q[3];
cz q[12], q[4];
rxx(0) q[7], q[19];
rzz(1.5707963267948966) q[0], q[15];
swap q[19], q[4];
h q[12];
rz(1.5707963267948966) q[6];
id q[16];
u1(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[11];
rxx(0) q[10], q[7];
p(0) q[13];
p(0) q[7];
t q[16];
u2(1.5707963267948966, 1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[7];
id q[6];
rx(1.5707963267948966) q[0];
swap q[17], q[4];
sdg q[6];
u1(1.5707963267948966) q[19];
u3(0, 0, 1.5707963267948966) q[10];
s q[5];
cy q[14], q[0];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[0];
rz(1.5707963267948966) q[12];
h q[17];
u1(1.5707963267948966) q[0];
t q[8];
cu3(1.5707963267948966, 0, 0) q[1], q[16];
h q[9];
cu1(1.5707963267948966) q[2], q[9];
h q[6];
id q[14];
u2(1.5707963267948966, 1.5707963267948966) q[8];
rzz(1.5707963267948966) q[3], q[0];
s q[12];
u3(0, 0, 1.5707963267948966) q[6];
s q[18];
u3(0, 0, 1.5707963267948966) q[8];
tdg q[7];
cu3(1.5707963267948966, 0, 0) q[3], q[14];
rx(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[6];
cy q[3], q[0];
cu3(1.5707963267948966, 0, 0) q[0], q[11];
sdg q[1];
u1(1.5707963267948966) q[17];
ry(1.5707963267948966) q[12];
rzz(1.5707963267948966) q[2], q[6];
s q[7];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[18];
rxx(0) q[3], q[9];
tdg q[4];
p(0) q[14];
s q[11];
u1(1.5707963267948966) q[4];
s q[4];
s q[15];
t q[5];
rzz(1.5707963267948966) q[12], q[1];
sdg q[4];
h q[9];
rx(1.5707963267948966) q[12];
ry(1.5707963267948966) q[10];
cz q[18], q[17];
u3(0, 0, 1.5707963267948966) q[18];
rz(1.5707963267948966) q[10];
u3(0, 0, 1.5707963267948966) q[9];
h q[15];
u1(1.5707963267948966) q[12];
p(0) q[17];
rzz(1.5707963267948966) q[16], q[10];
u1(1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[17];
cy q[10], q[11];
u1(1.5707963267948966) q[12];
sdg q[8];
u3(0, 0, 1.5707963267948966) q[0];
ry(1.5707963267948966) q[1];
rzz(1.5707963267948966) q[5], q[8];
rx(1.5707963267948966) q[3];
cz q[5], q[19];
t q[4];
tdg q[17];
u3(0, 0, 1.5707963267948966) q[2];
s q[4];
h q[19];
u2(1.5707963267948966, 1.5707963267948966) q[17];
s q[4];
rz(1.5707963267948966) q[18];
u1(1.5707963267948966) q[2];
rxx(0) q[2], q[19];
u3(0, 0, 1.5707963267948966) q[0];
rz(1.5707963267948966) q[12];
t q[8];
rx(1.5707963267948966) q[11];