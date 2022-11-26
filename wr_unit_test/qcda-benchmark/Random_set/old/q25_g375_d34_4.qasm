OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
creg c[25];
tdg q[21];
ry(1.5707963267948966) q[9];
id q[11];
swap q[17], q[4];
x q[6];
crz(1.5707963267948966) q[16], q[22];
s q[10];
u2(1.5707963267948966, 1.5707963267948966) q[13];
rzz(1.5707963267948966) q[19], q[23];
rz(1.5707963267948966) q[2];
t q[6];
u3(0, 0, 1.5707963267948966) q[24];
h q[11];
h q[3];
u1(1.5707963267948966) q[5];
s q[1];
rx(1.5707963267948966) q[13];
u1(1.5707963267948966) q[0];
rx(1.5707963267948966) q[9];
x q[4];
cx q[10], q[16];
cu1(1.5707963267948966) q[19], q[18];
tdg q[10];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[19];
id q[3];
rzz(1.5707963267948966) q[10], q[2];
id q[9];
u1(1.5707963267948966) q[13];
tdg q[0];
sdg q[21];
u3(0, 0, 1.5707963267948966) q[7];
h q[17];
sdg q[16];
rx(1.5707963267948966) q[22];
cz q[24], q[14];
cx q[20], q[21];
s q[0];
x q[14];
x q[5];
sdg q[22];
u1(1.5707963267948966) q[7];
s q[9];
u2(1.5707963267948966, 1.5707963267948966) q[21];
ryy(1.5707963267948966) q[15], q[24];
rxx(0) q[7], q[4];
ry(1.5707963267948966) q[8];
rxx(0) q[11], q[18];
ryy(1.5707963267948966) q[20], q[9];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[22];
cu1(1.5707963267948966) q[11], q[16];
h q[13];
sdg q[5];
u2(1.5707963267948966, 1.5707963267948966) q[13];
rz(1.5707963267948966) q[19];
u2(1.5707963267948966, 1.5707963267948966) q[17];
s q[17];
cx q[4], q[22];
x q[21];
ry(1.5707963267948966) q[20];
cu3(1.5707963267948966, 0, 0) q[17], q[13];
ry(1.5707963267948966) q[4];
sdg q[5];
u2(1.5707963267948966, 1.5707963267948966) q[3];
ry(1.5707963267948966) q[16];
tdg q[5];
rxx(0) q[24], q[11];
x q[21];
swap q[11], q[5];
tdg q[21];
s q[10];
ry(1.5707963267948966) q[1];
h q[17];
u3(0, 0, 1.5707963267948966) q[20];
ryy(1.5707963267948966) q[21], q[11];
cx q[11], q[0];
t q[13];
s q[22];
cz q[8], q[24];
tdg q[1];
crz(1.5707963267948966) q[7], q[6];
sdg q[5];
s q[11];
swap q[8], q[0];
id q[9];
id q[2];
t q[14];
u2(1.5707963267948966, 1.5707963267948966) q[13];
x q[13];
swap q[3], q[2];
rx(1.5707963267948966) q[12];
s q[8];
rz(1.5707963267948966) q[23];
s q[19];
h q[14];
s q[21];
u2(1.5707963267948966, 1.5707963267948966) q[19];
rx(1.5707963267948966) q[21];
rzz(1.5707963267948966) q[20], q[5];
x q[2];
x q[21];
u1(1.5707963267948966) q[4];
s q[3];
t q[1];
u3(0, 0, 1.5707963267948966) q[11];
rzz(1.5707963267948966) q[13], q[14];
id q[2];
rx(1.5707963267948966) q[16];
u2(1.5707963267948966, 1.5707963267948966) q[1];
x q[11];
p(0) q[3];
cx q[12], q[20];
ryy(1.5707963267948966) q[15], q[1];
id q[20];
sdg q[3];
tdg q[5];
s q[21];
rz(1.5707963267948966) q[15];
cu3(1.5707963267948966, 0, 0) q[1], q[14];
cu1(1.5707963267948966) q[2], q[21];
u1(1.5707963267948966) q[20];
p(0) q[11];
rxx(0) q[4], q[2];
t q[18];
rz(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[10];
u1(1.5707963267948966) q[21];
cu1(1.5707963267948966) q[5], q[12];
u1(1.5707963267948966) q[20];
tdg q[20];
cy q[3], q[8];
sdg q[19];
t q[18];
p(0) q[22];
id q[17];
tdg q[22];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[14];
t q[10];
rx(1.5707963267948966) q[18];
u2(1.5707963267948966, 1.5707963267948966) q[16];
cu1(1.5707963267948966) q[15], q[19];
rx(1.5707963267948966) q[15];
id q[16];
sdg q[20];
cx q[12], q[21];
u1(1.5707963267948966) q[21];
h q[23];
swap q[17], q[12];
ch q[20], q[24];
swap q[1], q[19];
x q[14];
ry(1.5707963267948966) q[18];
rx(1.5707963267948966) q[2];
t q[18];
tdg q[24];
rx(1.5707963267948966) q[5];
cy q[13], q[10];
h q[11];
ch q[3], q[1];
rz(1.5707963267948966) q[24];
x q[20];
ch q[7], q[18];
rx(1.5707963267948966) q[14];
ryy(1.5707963267948966) q[5], q[10];
id q[8];
cu1(1.5707963267948966) q[0], q[22];
rz(1.5707963267948966) q[15];
ryy(1.5707963267948966) q[13], q[5];
ch q[3], q[0];
swap q[8], q[22];
s q[13];
rz(1.5707963267948966) q[19];
u3(0, 0, 1.5707963267948966) q[8];
sdg q[1];
rx(1.5707963267948966) q[19];
t q[11];
p(0) q[11];
cy q[16], q[24];
s q[18];
u1(1.5707963267948966) q[15];
sdg q[22];
sdg q[4];
sdg q[20];
tdg q[5];
rx(1.5707963267948966) q[0];
u1(1.5707963267948966) q[20];
id q[6];
rz(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[11];
rx(1.5707963267948966) q[20];
cz q[24], q[13];
p(0) q[5];
swap q[17], q[8];
h q[12];
tdg q[9];
id q[20];
ryy(1.5707963267948966) q[3], q[23];
rx(1.5707963267948966) q[8];
rx(1.5707963267948966) q[15];
u3(0, 0, 1.5707963267948966) q[17];
sdg q[23];
rzz(1.5707963267948966) q[14], q[10];
h q[23];
h q[0];
p(0) q[22];
id q[9];
ch q[24], q[5];
id q[16];
ry(1.5707963267948966) q[11];
cz q[9], q[21];
rx(1.5707963267948966) q[19];
x q[7];
x q[13];
u2(1.5707963267948966, 1.5707963267948966) q[19];
u3(0, 0, 1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[21];
ry(1.5707963267948966) q[8];
p(0) q[4];
rzz(1.5707963267948966) q[18], q[10];
p(0) q[2];
sdg q[15];
ryy(1.5707963267948966) q[18], q[21];
cx q[10], q[0];
rxx(0) q[18], q[8];
ry(1.5707963267948966) q[3];
t q[6];
id q[14];
rxx(0) q[8], q[13];
ry(1.5707963267948966) q[19];
id q[22];
u1(1.5707963267948966) q[14];
ry(1.5707963267948966) q[22];
p(0) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[8];
rx(1.5707963267948966) q[5];
ryy(1.5707963267948966) q[12], q[21];
rz(1.5707963267948966) q[17];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[17];
s q[1];
u3(0, 0, 1.5707963267948966) q[8];
id q[4];
u2(1.5707963267948966, 1.5707963267948966) q[14];
h q[18];
cx q[23], q[21];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[2];
ry(1.5707963267948966) q[16];
rz(1.5707963267948966) q[13];
id q[18];
u1(1.5707963267948966) q[13];
sdg q[15];
ry(1.5707963267948966) q[12];
swap q[16], q[0];
s q[11];
h q[3];
cx q[2], q[8];
id q[5];
u2(1.5707963267948966, 1.5707963267948966) q[15];
crz(1.5707963267948966) q[14], q[17];
cu3(1.5707963267948966, 0, 0) q[24], q[13];
s q[1];
s q[11];
id q[8];
u3(0, 0, 1.5707963267948966) q[19];
s q[4];
crz(1.5707963267948966) q[20], q[5];
u3(0, 0, 1.5707963267948966) q[1];
tdg q[10];
h q[6];
u3(0, 0, 1.5707963267948966) q[4];
cx q[22], q[17];
crz(1.5707963267948966) q[15], q[9];
sdg q[20];
sdg q[15];
ry(1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[3];
swap q[11], q[3];
cu1(1.5707963267948966) q[3], q[19];
u2(1.5707963267948966, 1.5707963267948966) q[16];
ry(1.5707963267948966) q[16];
u1(1.5707963267948966) q[3];
cu1(1.5707963267948966) q[16], q[2];
rxx(0) q[15], q[12];
crz(1.5707963267948966) q[23], q[22];
id q[6];
id q[1];
u1(1.5707963267948966) q[11];
rxx(0) q[2], q[23];
p(0) q[19];
cu3(1.5707963267948966, 0, 0) q[4], q[6];
p(0) q[20];
swap q[18], q[0];
cy q[21], q[23];
u1(1.5707963267948966) q[0];
cz q[12], q[9];
crz(1.5707963267948966) q[15], q[8];
ryy(1.5707963267948966) q[7], q[3];
p(0) q[14];
x q[17];
id q[20];
t q[16];
cu1(1.5707963267948966) q[7], q[12];
s q[11];
u3(0, 0, 1.5707963267948966) q[4];
x q[23];
id q[2];
rz(1.5707963267948966) q[20];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rx(1.5707963267948966) q[0];
h q[9];
rx(1.5707963267948966) q[15];
u2(1.5707963267948966, 1.5707963267948966) q[1];
t q[1];
rz(1.5707963267948966) q[22];
swap q[10], q[14];
rx(1.5707963267948966) q[23];
u2(1.5707963267948966, 1.5707963267948966) q[18];
t q[8];
rx(1.5707963267948966) q[15];
u3(0, 0, 1.5707963267948966) q[10];
p(0) q[4];
rz(1.5707963267948966) q[17];
rz(1.5707963267948966) q[7];
s q[21];
sdg q[17];
ry(1.5707963267948966) q[14];
tdg q[12];
u1(1.5707963267948966) q[0];
h q[13];
ry(1.5707963267948966) q[20];
ch q[5], q[1];
cu3(1.5707963267948966, 0, 0) q[18], q[20];
h q[12];
u1(1.5707963267948966) q[9];
rxx(0) q[18], q[1];
p(0) q[5];
t q[22];
tdg q[9];
id q[18];
id q[13];
cu1(1.5707963267948966) q[4], q[7];
id q[8];
s q[24];
rz(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[18];
u2(1.5707963267948966, 1.5707963267948966) q[13];
p(0) q[18];
s q[14];
cu3(1.5707963267948966, 0, 0) q[1], q[12];
u1(1.5707963267948966) q[3];
rx(1.5707963267948966) q[8];
s q[22];
u2(1.5707963267948966, 1.5707963267948966) q[22];
sdg q[9];
h q[14];
u2(1.5707963267948966, 1.5707963267948966) q[20];
cu1(1.5707963267948966) q[15], q[7];
p(0) q[12];
t q[7];
u3(0, 0, 1.5707963267948966) q[22];
x q[19];
p(0) q[0];
cy q[23], q[5];
rx(1.5707963267948966) q[6];
p(0) q[7];
p(0) q[7];
h q[1];
p(0) q[16];
tdg q[12];
cu1(1.5707963267948966) q[15], q[9];
ry(1.5707963267948966) q[4];