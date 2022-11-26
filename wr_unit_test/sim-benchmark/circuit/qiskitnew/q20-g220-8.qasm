OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
tdg q[7];
cy q[12], q[0];
h q[6];
id q[17];
u2(1.5707963267948966, 1.5707963267948966) q[14];
ry(1.5707963267948966) q[9];
id q[7];
ry(1.5707963267948966) q[2];
sdg q[2];
id q[16];
p(0) q[10];
rzz(1.5707963267948966) q[8], q[2];
sdg q[19];
ry(1.5707963267948966) q[2];
s q[0];
sdg q[14];
id q[11];
h q[2];
ry(1.5707963267948966) q[5];
cx q[18], q[13];
id q[12];
sdg q[19];
tdg q[10];
u3(0, 0, 1.5707963267948966) q[10];
sdg q[8];
u3(0, 0, 1.5707963267948966) q[2];
sdg q[13];
rz(1.5707963267948966) q[9];
h q[3];
tdg q[19];
rz(1.5707963267948966) q[13];
id q[1];
u1(1.5707963267948966) q[10];
sdg q[16];
sdg q[16];
cz q[2], q[11];
rz(1.5707963267948966) q[12];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[8];
id q[19];
id q[9];
rx(1.5707963267948966) q[7];
t q[15];
rx(1.5707963267948966) q[18];
s q[12];
t q[13];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[16];
s q[3];
u3(0, 0, 1.5707963267948966) q[11];
u3(0, 0, 1.5707963267948966) q[14];
tdg q[7];
p(0) q[7];
rx(1.5707963267948966) q[17];
u2(1.5707963267948966, 1.5707963267948966) q[10];
ry(1.5707963267948966) q[1];
sdg q[19];
s q[6];
rxx(0) q[16], q[5];
h q[14];
cu1(1.5707963267948966) q[7], q[1];
u3(0, 0, 1.5707963267948966) q[18];
t q[4];
sdg q[17];
tdg q[17];
p(0) q[4];
cz q[1], q[6];
p(0) q[19];
sdg q[4];
p(0) q[0];
u3(0, 0, 1.5707963267948966) q[8];
rx(1.5707963267948966) q[10];
p(0) q[10];
p(0) q[12];
s q[2];
h q[8];
s q[7];
u1(1.5707963267948966) q[19];
rz(1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[13];
u1(1.5707963267948966) q[4];
cu3(1.5707963267948966, 0, 0) q[2], q[4];
h q[0];
sdg q[15];
cu3(1.5707963267948966, 0, 0) q[3], q[4];
id q[17];
p(0) q[15];
u2(1.5707963267948966, 1.5707963267948966) q[10];
p(0) q[2];
u1(1.5707963267948966) q[14];
rz(1.5707963267948966) q[12];
rzz(1.5707963267948966) q[11], q[1];
cu1(1.5707963267948966) q[5], q[4];
rx(1.5707963267948966) q[4];
sdg q[1];
u1(1.5707963267948966) q[8];
cz q[2], q[13];
t q[11];
u3(0, 0, 1.5707963267948966) q[10];
cx q[12], q[18];
rz(1.5707963267948966) q[15];
rz(1.5707963267948966) q[17];
cu3(1.5707963267948966, 0, 0) q[4], q[14];
t q[8];
u2(1.5707963267948966, 1.5707963267948966) q[2];
rz(1.5707963267948966) q[12];
swap q[6], q[14];
id q[16];
cu3(1.5707963267948966, 0, 0) q[1], q[11];
t q[11];
u2(1.5707963267948966, 1.5707963267948966) q[14];
u1(1.5707963267948966) q[8];
tdg q[7];
cu3(1.5707963267948966, 0, 0) q[8], q[2];
tdg q[6];
rx(1.5707963267948966) q[10];
t q[9];
rx(1.5707963267948966) q[9];
s q[5];
rzz(1.5707963267948966) q[12], q[3];
tdg q[7];
u1(1.5707963267948966) q[12];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[11];
id q[11];
rx(1.5707963267948966) q[10];
cu1(1.5707963267948966) q[9], q[8];
tdg q[1];
rxx(0) q[6], q[13];
s q[17];
t q[5];
rx(1.5707963267948966) q[18];
sdg q[19];
id q[10];
u2(1.5707963267948966, 1.5707963267948966) q[9];
cy q[14], q[5];
s q[0];
rx(1.5707963267948966) q[5];
u1(1.5707963267948966) q[9];
t q[17];
u1(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[14];
p(0) q[3];
id q[12];
id q[12];
swap q[8], q[11];
cx q[11], q[6];
id q[12];
cu1(1.5707963267948966) q[14], q[13];
rz(1.5707963267948966) q[15];
u1(1.5707963267948966) q[2];
s q[16];
swap q[9], q[8];
rxx(0) q[12], q[4];
sdg q[16];
p(0) q[15];
rz(1.5707963267948966) q[19];
rzz(1.5707963267948966) q[8], q[18];
ry(1.5707963267948966) q[16];
cx q[1], q[6];
u2(1.5707963267948966, 1.5707963267948966) q[12];
u2(1.5707963267948966, 1.5707963267948966) q[4];
cy q[6], q[7];
id q[3];
s q[19];
rx(1.5707963267948966) q[5];
sdg q[3];
rx(1.5707963267948966) q[9];
u1(1.5707963267948966) q[18];
sdg q[0];
p(0) q[11];
id q[15];
rz(1.5707963267948966) q[3];
p(0) q[0];
t q[13];
rz(1.5707963267948966) q[0];
h q[4];
h q[2];
h q[6];
s q[16];
rz(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[14];
u2(1.5707963267948966, 1.5707963267948966) q[9];
rxx(0) q[11], q[1];
u1(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[14];
u1(1.5707963267948966) q[14];
cx q[18], q[6];
t q[8];
cu3(1.5707963267948966, 0, 0) q[19], q[11];
ry(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[16];
t q[10];
cu1(1.5707963267948966) q[16], q[15];
tdg q[4];
p(0) q[18];
p(0) q[13];
cu3(1.5707963267948966, 0, 0) q[13], q[1];
u2(1.5707963267948966, 1.5707963267948966) q[0];
p(0) q[10];
sdg q[14];
ry(1.5707963267948966) q[10];
p(0) q[5];
id q[18];
u3(0, 0, 1.5707963267948966) q[16];
h q[14];
t q[9];
cy q[18], q[15];
h q[18];
rxx(0) q[5], q[16];
rx(1.5707963267948966) q[0];
id q[13];
u3(0, 0, 1.5707963267948966) q[9];
rzz(1.5707963267948966) q[11], q[16];
id q[8];
t q[9];
id q[15];
t q[0];
h q[10];