OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
creg c[25];
t q[2];
cz q[2], q[17];
u3(0, 0, 1.5707963267948966) q[7];
cu1(1.5707963267948966) q[12], q[7];
t q[0];
u2(1.5707963267948966, 1.5707963267948966) q[19];
u1(1.5707963267948966) q[1];
id q[12];
h q[18];
u2(1.5707963267948966, 1.5707963267948966) q[9];
h q[23];
ry(1.5707963267948966) q[10];
t q[11];
x q[6];
sdg q[14];
rx(1.5707963267948966) q[20];
p(0) q[22];
ry(1.5707963267948966) q[10];
u1(1.5707963267948966) q[22];
x q[11];
ch q[11], q[23];
tdg q[17];
rz(1.5707963267948966) q[23];
rzz(1.5707963267948966) q[9], q[12];
h q[12];
ry(1.5707963267948966) q[19];
u3(0, 0, 1.5707963267948966) q[9];
tdg q[21];
x q[8];
rx(1.5707963267948966) q[0];
t q[3];
s q[18];
t q[21];
sdg q[13];
sdg q[6];
rz(1.5707963267948966) q[3];
cx q[8], q[15];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[24];
id q[1];
rx(1.5707963267948966) q[12];
rxx(0) q[22], q[17];
t q[6];
u3(0, 0, 1.5707963267948966) q[23];
s q[16];
u2(1.5707963267948966, 1.5707963267948966) q[20];
id q[2];
cx q[16], q[14];
tdg q[4];
u1(1.5707963267948966) q[4];
x q[12];
x q[1];
u2(1.5707963267948966, 1.5707963267948966) q[1];
h q[19];
s q[10];
cu3(1.5707963267948966, 0, 0) q[1], q[21];
s q[13];
u3(0, 0, 1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[15];
s q[0];
cz q[19], q[15];
ry(1.5707963267948966) q[9];
ryy(1.5707963267948966) q[5], q[13];
tdg q[9];
ry(1.5707963267948966) q[23];
t q[14];
u3(0, 0, 1.5707963267948966) q[18];
x q[19];
u1(1.5707963267948966) q[14];
t q[22];
u1(1.5707963267948966) q[12];
ryy(1.5707963267948966) q[6], q[14];
ry(1.5707963267948966) q[11];
cy q[23], q[8];
u2(1.5707963267948966, 1.5707963267948966) q[20];
h q[18];
cz q[2], q[16];
u1(1.5707963267948966) q[17];
u2(1.5707963267948966, 1.5707963267948966) q[1];
p(0) q[23];
crz(1.5707963267948966) q[16], q[23];
u3(0, 0, 1.5707963267948966) q[12];
t q[4];
tdg q[10];
x q[22];
u1(1.5707963267948966) q[9];
h q[7];
rx(1.5707963267948966) q[21];
rzz(1.5707963267948966) q[9], q[0];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rz(1.5707963267948966) q[19];
id q[15];
u1(1.5707963267948966) q[4];
cy q[13], q[14];
u2(1.5707963267948966, 1.5707963267948966) q[23];
p(0) q[13];
u3(0, 0, 1.5707963267948966) q[9];
cx q[6], q[12];
u3(0, 0, 1.5707963267948966) q[6];
p(0) q[3];
cu1(1.5707963267948966) q[24], q[20];
sdg q[10];
p(0) q[1];
tdg q[23];
id q[3];
sdg q[24];
t q[5];
tdg q[22];
rxx(0) q[22], q[11];
id q[14];
u1(1.5707963267948966) q[4];
ry(1.5707963267948966) q[1];
x q[16];
cx q[3], q[20];
cu1(1.5707963267948966) q[16], q[4];
u1(1.5707963267948966) q[17];
p(0) q[17];
p(0) q[17];
sdg q[14];
sdg q[14];
u2(1.5707963267948966, 1.5707963267948966) q[16];
t q[12];
cx q[13], q[22];
crz(1.5707963267948966) q[22], q[9];
cu3(1.5707963267948966, 0, 0) q[22], q[5];
s q[19];
u2(1.5707963267948966, 1.5707963267948966) q[19];
tdg q[11];
cu1(1.5707963267948966) q[4], q[16];
sdg q[2];
h q[1];
u3(0, 0, 1.5707963267948966) q[3];
ry(1.5707963267948966) q[20];
rxx(0) q[14], q[20];
cu3(1.5707963267948966, 0, 0) q[17], q[0];
cx q[3], q[21];
u1(1.5707963267948966) q[11];
tdg q[2];
t q[10];
tdg q[1];
id q[4];
u1(1.5707963267948966) q[23];
h q[15];
u2(1.5707963267948966, 1.5707963267948966) q[18];
sdg q[23];
cy q[21], q[4];
s q[3];
rx(1.5707963267948966) q[13];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[22];
u2(1.5707963267948966, 1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[6];
t q[9];
cu1(1.5707963267948966) q[23], q[13];
rxx(0) q[8], q[24];
rz(1.5707963267948966) q[14];
u1(1.5707963267948966) q[5];
rz(1.5707963267948966) q[20];
u3(0, 0, 1.5707963267948966) q[16];
s q[15];
rz(1.5707963267948966) q[12];
t q[6];
x q[9];
cz q[16], q[22];
tdg q[9];
t q[5];
u1(1.5707963267948966) q[7];
rzz(1.5707963267948966) q[18], q[15];
t q[1];
h q[5];
s q[12];
rx(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[8];
ch q[22], q[7];
rxx(0) q[20], q[0];
u3(0, 0, 1.5707963267948966) q[21];
p(0) q[19];
ry(1.5707963267948966) q[14];
ry(1.5707963267948966) q[3];
u1(1.5707963267948966) q[11];
rzz(1.5707963267948966) q[20], q[8];
u3(0, 0, 1.5707963267948966) q[20];
rzz(1.5707963267948966) q[16], q[20];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[23];
u1(1.5707963267948966) q[24];
t q[23];
swap q[4], q[0];
ch q[21], q[3];
id q[5];
rx(1.5707963267948966) q[24];
u1(1.5707963267948966) q[8];
id q[11];
x q[20];
p(0) q[0];
crz(1.5707963267948966) q[4], q[10];
cu1(1.5707963267948966) q[18], q[6];
cz q[7], q[24];
u2(1.5707963267948966, 1.5707963267948966) q[21];
id q[18];
rzz(1.5707963267948966) q[19], q[10];
id q[11];
h q[20];
cu1(1.5707963267948966) q[10], q[17];
u1(1.5707963267948966) q[18];
u2(1.5707963267948966, 1.5707963267948966) q[18];
tdg q[15];
u2(1.5707963267948966, 1.5707963267948966) q[23];
sdg q[6];
tdg q[23];
h q[20];
u3(0, 0, 1.5707963267948966) q[7];
t q[2];
x q[7];
h q[9];
ch q[23], q[11];
t q[16];
swap q[1], q[22];
id q[17];
u1(1.5707963267948966) q[16];
ry(1.5707963267948966) q[19];
t q[15];
h q[15];
rx(1.5707963267948966) q[24];
s q[6];