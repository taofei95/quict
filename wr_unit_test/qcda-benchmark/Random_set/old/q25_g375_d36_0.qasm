OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
creg c[25];
p(0) q[12];
cu1(1.5707963267948966) q[16], q[20];
cx q[16], q[8];
swap q[13], q[19];
id q[1];
h q[6];
t q[23];
cy q[24], q[10];
id q[16];
cz q[21], q[14];
rzz(1.5707963267948966) q[21], q[22];
id q[12];
cz q[6], q[22];
swap q[6], q[12];
sdg q[21];
rxx(0) q[6], q[18];
u1(1.5707963267948966) q[13];
sdg q[0];
cu3(1.5707963267948966, 0, 0) q[1], q[19];
ryy(1.5707963267948966) q[20], q[13];
u2(1.5707963267948966, 1.5707963267948966) q[17];
p(0) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[21];
u1(1.5707963267948966) q[5];
x q[3];
cz q[24], q[15];
sdg q[10];
ch q[21], q[18];
u1(1.5707963267948966) q[11];
ry(1.5707963267948966) q[22];
x q[16];
ch q[9], q[24];
rx(1.5707963267948966) q[22];
p(0) q[3];
id q[3];
id q[18];
rz(1.5707963267948966) q[18];
rx(1.5707963267948966) q[1];
rxx(0) q[2], q[10];
u2(1.5707963267948966, 1.5707963267948966) q[10];
u3(0, 0, 1.5707963267948966) q[20];
u3(0, 0, 1.5707963267948966) q[4];
cu3(1.5707963267948966, 0, 0) q[2], q[9];
crz(1.5707963267948966) q[8], q[2];
ch q[21], q[17];
id q[10];
u2(1.5707963267948966, 1.5707963267948966) q[24];
sdg q[13];
s q[7];
s q[16];
rx(1.5707963267948966) q[8];
x q[7];
u1(1.5707963267948966) q[4];
h q[2];
sdg q[18];
cz q[9], q[11];
u1(1.5707963267948966) q[1];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[11];
ch q[19], q[12];
id q[6];
ryy(1.5707963267948966) q[16], q[10];
tdg q[15];
rz(1.5707963267948966) q[17];
sdg q[23];
id q[5];
rxx(0) q[2], q[7];
u2(1.5707963267948966, 1.5707963267948966) q[6];
x q[23];
cu3(1.5707963267948966, 0, 0) q[23], q[1];
x q[21];
id q[9];
u3(0, 0, 1.5707963267948966) q[4];
swap q[3], q[17];
ry(1.5707963267948966) q[10];
swap q[1], q[4];
s q[4];
tdg q[6];
p(0) q[4];
t q[20];
ch q[22], q[14];
id q[2];
ryy(1.5707963267948966) q[18], q[19];
ry(1.5707963267948966) q[16];
t q[13];
sdg q[1];
rx(1.5707963267948966) q[5];
s q[2];
h q[21];
ry(1.5707963267948966) q[10];
swap q[21], q[6];
u3(0, 0, 1.5707963267948966) q[14];
s q[2];
h q[18];
t q[4];
h q[1];
s q[1];
id q[21];
tdg q[21];
cu1(1.5707963267948966) q[5], q[16];
cz q[22], q[6];
crz(1.5707963267948966) q[10], q[15];
p(0) q[20];
p(0) q[17];
h q[17];
u3(0, 0, 1.5707963267948966) q[18];
rx(1.5707963267948966) q[22];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[16];
id q[10];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rz(1.5707963267948966) q[24];
p(0) q[13];
tdg q[24];
x q[9];
rz(1.5707963267948966) q[19];
s q[10];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[6];
ry(1.5707963267948966) q[5];
cu1(1.5707963267948966) q[2], q[17];
rz(1.5707963267948966) q[8];
h q[1];
id q[10];
x q[24];
t q[16];
s q[14];
h q[9];
cz q[14], q[22];
ry(1.5707963267948966) q[23];
id q[9];
rx(1.5707963267948966) q[19];
u3(0, 0, 1.5707963267948966) q[0];
sdg q[12];
x q[6];
s q[4];
swap q[10], q[1];
t q[11];
ry(1.5707963267948966) q[13];
x q[18];
cz q[9], q[7];
p(0) q[18];
id q[21];
h q[14];
u1(1.5707963267948966) q[14];
rxx(0) q[23], q[22];
h q[9];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[6];
cz q[18], q[21];
ch q[13], q[0];
t q[7];
ry(1.5707963267948966) q[9];
x q[23];
rx(1.5707963267948966) q[20];
h q[2];
ryy(1.5707963267948966) q[7], q[8];
ry(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[18];
ry(1.5707963267948966) q[13];
cu3(1.5707963267948966, 0, 0) q[24], q[22];
ch q[21], q[10];
sdg q[23];
sdg q[6];
cu3(1.5707963267948966, 0, 0) q[13], q[2];
sdg q[6];
s q[17];
h q[17];
cy q[8], q[9];
rz(1.5707963267948966) q[20];
sdg q[17];
id q[3];
u3(0, 0, 1.5707963267948966) q[15];
p(0) q[1];
rz(1.5707963267948966) q[22];
cx q[23], q[13];
ch q[21], q[16];
u1(1.5707963267948966) q[13];
rxx(0) q[14], q[3];
h q[1];
x q[22];
u2(1.5707963267948966, 1.5707963267948966) q[11];
rz(1.5707963267948966) q[8];
s q[8];
crz(1.5707963267948966) q[22], q[3];
crz(1.5707963267948966) q[10], q[22];
u3(0, 0, 1.5707963267948966) q[16];
rzz(1.5707963267948966) q[7], q[6];
p(0) q[3];
rz(1.5707963267948966) q[15];
h q[20];
p(0) q[10];
tdg q[0];
cu1(1.5707963267948966) q[20], q[18];
u1(1.5707963267948966) q[16];
u2(1.5707963267948966, 1.5707963267948966) q[10];
s q[20];
sdg q[16];
u2(1.5707963267948966, 1.5707963267948966) q[1];
ry(1.5707963267948966) q[7];
id q[20];
h q[18];
sdg q[21];
ch q[23], q[0];
cy q[22], q[24];
t q[8];
cz q[0], q[20];
tdg q[12];
s q[4];
tdg q[24];
id q[4];
u1(1.5707963267948966) q[10];
s q[4];
x q[9];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[19];
u2(1.5707963267948966, 1.5707963267948966) q[10];
id q[8];
rz(1.5707963267948966) q[23];
h q[13];
u2(1.5707963267948966, 1.5707963267948966) q[19];
cz q[18], q[24];
s q[2];
sdg q[20];
t q[4];
cz q[21], q[17];
ry(1.5707963267948966) q[15];
rx(1.5707963267948966) q[23];
sdg q[24];
s q[3];
cz q[2], q[5];
t q[24];
ry(1.5707963267948966) q[4];
sdg q[2];
rxx(0) q[4], q[10];
id q[11];
x q[5];
cx q[8], q[14];
swap q[5], q[22];
u1(1.5707963267948966) q[12];
s q[11];
u3(0, 0, 1.5707963267948966) q[22];
p(0) q[16];
t q[18];
rx(1.5707963267948966) q[24];
ryy(1.5707963267948966) q[15], q[0];
u1(1.5707963267948966) q[22];
rx(1.5707963267948966) q[15];
tdg q[10];
cu3(1.5707963267948966, 0, 0) q[18], q[13];
cz q[2], q[8];
rx(1.5707963267948966) q[11];
rz(1.5707963267948966) q[14];
u2(1.5707963267948966, 1.5707963267948966) q[15];
ch q[15], q[24];
t q[11];
rz(1.5707963267948966) q[5];
sdg q[9];
h q[21];
cx q[14], q[15];
crz(1.5707963267948966) q[20], q[1];
ry(1.5707963267948966) q[5];
sdg q[3];
s q[18];
tdg q[9];
id q[4];
rxx(0) q[3], q[17];
cx q[9], q[3];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[10];
u3(0, 0, 1.5707963267948966) q[11];
s q[0];
s q[4];
h q[18];
h q[1];
h q[4];
rz(1.5707963267948966) q[16];
p(0) q[15];
s q[0];
t q[0];
rzz(1.5707963267948966) q[6], q[8];
u3(0, 0, 1.5707963267948966) q[0];
id q[13];
rxx(0) q[16], q[15];
p(0) q[3];
s q[5];
rz(1.5707963267948966) q[0];
u1(1.5707963267948966) q[24];
rx(1.5707963267948966) q[13];
s q[24];
u3(0, 0, 1.5707963267948966) q[12];
u2(1.5707963267948966, 1.5707963267948966) q[9];
x q[11];
x q[1];
tdg q[3];
s q[5];
u2(1.5707963267948966, 1.5707963267948966) q[22];
sdg q[16];
cx q[10], q[22];
tdg q[19];
tdg q[5];
p(0) q[18];
u3(0, 0, 1.5707963267948966) q[10];
t q[7];
h q[4];
cu3(1.5707963267948966, 0, 0) q[0], q[5];
rx(1.5707963267948966) q[10];
h q[15];
u1(1.5707963267948966) q[12];
ry(1.5707963267948966) q[12];
id q[2];
rz(1.5707963267948966) q[24];
cu3(1.5707963267948966, 0, 0) q[20], q[2];
x q[9];
sdg q[21];
rx(1.5707963267948966) q[12];
u3(0, 0, 1.5707963267948966) q[17];
p(0) q[12];
u1(1.5707963267948966) q[4];
ry(1.5707963267948966) q[7];
cu1(1.5707963267948966) q[8], q[18];
sdg q[19];
s q[8];
u1(1.5707963267948966) q[16];
rz(1.5707963267948966) q[24];
sdg q[22];
t q[10];
id q[3];
p(0) q[6];
rx(1.5707963267948966) q[11];
ch q[9], q[21];
swap q[1], q[11];
u1(1.5707963267948966) q[7];
ryy(1.5707963267948966) q[17], q[24];
cy q[19], q[4];
rz(1.5707963267948966) q[19];
u3(0, 0, 1.5707963267948966) q[2];
ry(1.5707963267948966) q[8];
id q[9];
h q[18];
rxx(0) q[4], q[21];
sdg q[15];
t q[8];
t q[14];
ry(1.5707963267948966) q[12];
rz(1.5707963267948966) q[23];
ryy(1.5707963267948966) q[22], q[7];
t q[9];
rx(1.5707963267948966) q[20];
cu3(1.5707963267948966, 0, 0) q[7], q[9];
rz(1.5707963267948966) q[9];
ry(1.5707963267948966) q[9];
cu1(1.5707963267948966) q[3], q[9];
u1(1.5707963267948966) q[21];
h q[23];
u1(1.5707963267948966) q[21];
u2(1.5707963267948966, 1.5707963267948966) q[21];
rz(1.5707963267948966) q[18];
swap q[4], q[2];
ry(1.5707963267948966) q[6];
cu3(1.5707963267948966, 0, 0) q[23], q[15];
ryy(1.5707963267948966) q[5], q[20];
id q[10];
u3(0, 0, 1.5707963267948966) q[15];
cy q[17], q[13];
p(0) q[13];
u1(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[2];
cz q[15], q[8];
rx(1.5707963267948966) q[12];
x q[18];
id q[8];
cu1(1.5707963267948966) q[14], q[21];
h q[8];