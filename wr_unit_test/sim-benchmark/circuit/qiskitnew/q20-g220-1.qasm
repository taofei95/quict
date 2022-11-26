OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
cx q[16], q[12];
u3(0, 0, 1.5707963267948966) q[6];
swap q[16], q[9];
rx(1.5707963267948966) q[10];
sdg q[12];
u2(1.5707963267948966, 1.5707963267948966) q[13];
rz(1.5707963267948966) q[6];
tdg q[19];
t q[10];
u1(1.5707963267948966) q[16];
u2(1.5707963267948966, 1.5707963267948966) q[8];
h q[14];
id q[9];
u1(1.5707963267948966) q[19];
id q[15];
sdg q[1];
cu3(1.5707963267948966, 0, 0) q[16], q[18];
tdg q[19];
u2(1.5707963267948966, 1.5707963267948966) q[12];
ry(1.5707963267948966) q[5];
cu3(1.5707963267948966, 0, 0) q[16], q[14];
rzz(1.5707963267948966) q[12], q[14];
p(0) q[9];
u1(1.5707963267948966) q[2];
cu1(1.5707963267948966) q[15], q[13];
u1(1.5707963267948966) q[9];
id q[9];
u3(0, 0, 1.5707963267948966) q[10];
id q[0];
p(0) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rx(1.5707963267948966) q[8];
sdg q[16];
ry(1.5707963267948966) q[13];
tdg q[3];
cu1(1.5707963267948966) q[13], q[11];
id q[1];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[13];
p(0) q[9];
rxx(0) q[13], q[9];
sdg q[8];
t q[1];
p(0) q[4];
p(0) q[9];
t q[11];
cx q[2], q[16];
p(0) q[13];
u2(1.5707963267948966, 1.5707963267948966) q[19];
u1(1.5707963267948966) q[18];
u2(1.5707963267948966, 1.5707963267948966) q[1];
cy q[17], q[14];
u2(1.5707963267948966, 1.5707963267948966) q[1];
id q[9];
cy q[19], q[5];
u2(1.5707963267948966, 1.5707963267948966) q[11];
h q[4];
tdg q[6];
u2(1.5707963267948966, 1.5707963267948966) q[18];
ry(1.5707963267948966) q[19];
cz q[4], q[12];
p(0) q[15];
t q[13];
t q[5];
cu1(1.5707963267948966) q[4], q[17];
p(0) q[3];
rz(1.5707963267948966) q[0];
rzz(1.5707963267948966) q[11], q[10];
h q[15];
cu3(1.5707963267948966, 0, 0) q[6], q[0];
id q[2];
id q[5];
swap q[5], q[4];
rx(1.5707963267948966) q[9];
u1(1.5707963267948966) q[19];
h q[10];
rx(1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[5];
p(0) q[16];
ry(1.5707963267948966) q[10];
tdg q[8];
p(0) q[12];
cy q[9], q[15];
sdg q[0];
u2(1.5707963267948966, 1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[18];
rz(1.5707963267948966) q[19];
rz(1.5707963267948966) q[12];
id q[11];
cz q[18], q[13];
rx(1.5707963267948966) q[4];
swap q[17], q[9];
u2(1.5707963267948966, 1.5707963267948966) q[15];
t q[2];
id q[3];
u2(1.5707963267948966, 1.5707963267948966) q[12];
rx(1.5707963267948966) q[18];
rzz(1.5707963267948966) q[7], q[15];
s q[13];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[15];
u1(1.5707963267948966) q[6];
ry(1.5707963267948966) q[19];
ry(1.5707963267948966) q[1];
id q[13];
u1(1.5707963267948966) q[2];
sdg q[19];
t q[13];
cu1(1.5707963267948966) q[13], q[19];
cx q[16], q[5];
cu1(1.5707963267948966) q[1], q[3];
tdg q[8];
u3(0, 0, 1.5707963267948966) q[12];
p(0) q[1];
p(0) q[14];
id q[1];
u1(1.5707963267948966) q[19];
id q[5];
id q[9];
cu3(1.5707963267948966, 0, 0) q[8], q[19];
cz q[4], q[12];
t q[3];
swap q[17], q[14];
rz(1.5707963267948966) q[8];
t q[14];
u3(0, 0, 1.5707963267948966) q[13];
u3(0, 0, 1.5707963267948966) q[18];
sdg q[6];
u1(1.5707963267948966) q[12];
u1(1.5707963267948966) q[8];
t q[4];
id q[2];
id q[16];
u1(1.5707963267948966) q[6];
h q[13];
t q[17];
s q[13];
u2(1.5707963267948966, 1.5707963267948966) q[17];
id q[5];
rzz(1.5707963267948966) q[2], q[17];
s q[16];
t q[15];
cu3(1.5707963267948966, 0, 0) q[8], q[18];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[4];
s q[19];
u1(1.5707963267948966) q[1];
sdg q[14];
u2(1.5707963267948966, 1.5707963267948966) q[14];
t q[16];
h q[18];
swap q[6], q[10];
p(0) q[16];
u2(1.5707963267948966, 1.5707963267948966) q[7];
rx(1.5707963267948966) q[8];
t q[13];
cu1(1.5707963267948966) q[4], q[3];
cu3(1.5707963267948966, 0, 0) q[6], q[19];
rzz(1.5707963267948966) q[9], q[13];
h q[4];
rz(1.5707963267948966) q[12];
cu1(1.5707963267948966) q[4], q[17];
u1(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[13];
rz(1.5707963267948966) q[18];
cu1(1.5707963267948966) q[18], q[0];
rxx(0) q[13], q[12];
sdg q[18];
ry(1.5707963267948966) q[19];
cu1(1.5707963267948966) q[17], q[9];
cy q[11], q[3];
cz q[0], q[17];
rx(1.5707963267948966) q[15];
swap q[2], q[11];
rx(1.5707963267948966) q[4];
p(0) q[17];
cu1(1.5707963267948966) q[8], q[13];
cu1(1.5707963267948966) q[2], q[13];
cu1(1.5707963267948966) q[12], q[8];
t q[9];
cu1(1.5707963267948966) q[2], q[15];
rxx(0) q[3], q[15];
u2(1.5707963267948966, 1.5707963267948966) q[13];
rz(1.5707963267948966) q[14];
ry(1.5707963267948966) q[9];
tdg q[14];
u3(0, 0, 1.5707963267948966) q[9];
id q[8];
sdg q[18];
ry(1.5707963267948966) q[19];
rxx(0) q[14], q[8];
h q[0];
ry(1.5707963267948966) q[16];
id q[3];
tdg q[7];
sdg q[7];
tdg q[18];
ry(1.5707963267948966) q[0];
s q[7];
ry(1.5707963267948966) q[1];
s q[11];
u1(1.5707963267948966) q[0];
u1(1.5707963267948966) q[19];
h q[3];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[5];
tdg q[4];
p(0) q[18];
u1(1.5707963267948966) q[6];
t q[7];
cy q[12], q[14];
sdg q[15];
rz(1.5707963267948966) q[15];
rx(1.5707963267948966) q[18];
h q[17];
u3(0, 0, 1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[14];
rx(1.5707963267948966) q[1];
sdg q[15];
cx q[2], q[9];