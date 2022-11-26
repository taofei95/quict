OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
creg c[25];
u1(1.5707963267948966) q[22];
p(0) q[24];
s q[10];
cy q[8], q[22];
cu1(1.5707963267948966) q[9], q[18];
u1(1.5707963267948966) q[14];
u3(0, 0, 1.5707963267948966) q[2];
id q[21];
swap q[18], q[4];
tdg q[0];
rx(1.5707963267948966) q[20];
cu1(1.5707963267948966) q[20], q[18];
p(0) q[24];
ch q[11], q[1];
h q[18];
p(0) q[13];
crz(1.5707963267948966) q[24], q[4];
h q[24];
p(0) q[10];
swap q[13], q[20];
u3(0, 0, 1.5707963267948966) q[16];
x q[24];
u3(0, 0, 1.5707963267948966) q[6];
rz(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[10];
ry(1.5707963267948966) q[23];
u3(0, 0, 1.5707963267948966) q[10];
tdg q[13];
rx(1.5707963267948966) q[19];
cu3(1.5707963267948966, 0, 0) q[18], q[22];
cu3(1.5707963267948966, 0, 0) q[10], q[6];
sdg q[13];
cu1(1.5707963267948966) q[7], q[0];
id q[7];
u3(0, 0, 1.5707963267948966) q[14];
u1(1.5707963267948966) q[16];
rzz(1.5707963267948966) q[0], q[21];
x q[5];
tdg q[22];
u3(0, 0, 1.5707963267948966) q[8];
p(0) q[21];
h q[9];
rz(1.5707963267948966) q[21];
x q[7];
p(0) q[3];
rz(1.5707963267948966) q[10];
h q[14];
x q[1];
p(0) q[22];
s q[5];
cu3(1.5707963267948966, 0, 0) q[10], q[8];
rxx(0) q[2], q[15];
rz(1.5707963267948966) q[17];
u1(1.5707963267948966) q[2];
swap q[3], q[2];
s q[15];
s q[7];
t q[17];
t q[12];
rz(1.5707963267948966) q[21];
p(0) q[12];
s q[17];
cz q[21], q[0];
s q[23];
id q[20];
crz(1.5707963267948966) q[13], q[3];
ry(1.5707963267948966) q[9];
id q[21];
cx q[0], q[1];
ry(1.5707963267948966) q[13];
ry(1.5707963267948966) q[8];
tdg q[23];
tdg q[20];
id q[20];
crz(1.5707963267948966) q[14], q[1];
u2(1.5707963267948966, 1.5707963267948966) q[21];
s q[22];
u1(1.5707963267948966) q[9];
tdg q[14];
x q[20];
rx(1.5707963267948966) q[5];
cu3(1.5707963267948966, 0, 0) q[15], q[21];
ryy(1.5707963267948966) q[2], q[0];
sdg q[14];
s q[0];
rxx(0) q[3], q[15];
rx(1.5707963267948966) q[15];
u1(1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[22];
id q[18];
u2(1.5707963267948966, 1.5707963267948966) q[17];
s q[24];
x q[7];
s q[0];
tdg q[2];
u3(0, 0, 1.5707963267948966) q[10];
t q[9];
ry(1.5707963267948966) q[5];
x q[6];
cz q[11], q[9];
u3(0, 0, 1.5707963267948966) q[15];
u2(1.5707963267948966, 1.5707963267948966) q[13];
ryy(1.5707963267948966) q[10], q[8];
ry(1.5707963267948966) q[24];
id q[0];
u3(0, 0, 1.5707963267948966) q[13];
u3(0, 0, 1.5707963267948966) q[3];
cu1(1.5707963267948966) q[1], q[19];
id q[4];
ch q[17], q[19];
t q[0];
sdg q[16];
u2(1.5707963267948966, 1.5707963267948966) q[13];
u1(1.5707963267948966) q[17];
id q[16];
cy q[19], q[6];
x q[21];
ry(1.5707963267948966) q[19];
tdg q[22];
sdg q[16];
u2(1.5707963267948966, 1.5707963267948966) q[9];
x q[4];
tdg q[5];
p(0) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[11];
u1(1.5707963267948966) q[12];
cz q[9], q[16];
cx q[23], q[16];
cu3(1.5707963267948966, 0, 0) q[20], q[3];
rz(1.5707963267948966) q[14];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[9];
cx q[23], q[15];
sdg q[6];
u2(1.5707963267948966, 1.5707963267948966) q[24];
ry(1.5707963267948966) q[23];
ry(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[16];
rz(1.5707963267948966) q[6];
id q[1];
ryy(1.5707963267948966) q[7], q[10];
cz q[9], q[14];
h q[17];
h q[0];
h q[3];
ry(1.5707963267948966) q[1];
id q[6];
rx(1.5707963267948966) q[16];
cz q[11], q[0];
u3(0, 0, 1.5707963267948966) q[7];
s q[11];
u1(1.5707963267948966) q[19];
ch q[4], q[23];
t q[19];
rz(1.5707963267948966) q[8];
u1(1.5707963267948966) q[10];
rz(1.5707963267948966) q[22];
s q[4];
p(0) q[20];
ry(1.5707963267948966) q[16];
tdg q[11];
x q[21];
sdg q[6];
sdg q[6];
rx(1.5707963267948966) q[15];
rz(1.5707963267948966) q[0];
swap q[7], q[5];
rxx(0) q[8], q[3];
u1(1.5707963267948966) q[4];
s q[5];
t q[7];
rzz(1.5707963267948966) q[3], q[12];
sdg q[21];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[10];
sdg q[7];
sdg q[7];
tdg q[13];
u3(0, 0, 1.5707963267948966) q[21];
ryy(1.5707963267948966) q[20], q[14];
u1(1.5707963267948966) q[3];
rx(1.5707963267948966) q[17];
h q[14];
p(0) q[19];
u1(1.5707963267948966) q[1];
h q[1];
s q[20];
ry(1.5707963267948966) q[16];
rx(1.5707963267948966) q[1];
cy q[22], q[16];
x q[22];
s q[15];
s q[2];
u2(1.5707963267948966, 1.5707963267948966) q[8];
t q[23];
h q[15];
cu3(1.5707963267948966, 0, 0) q[24], q[17];
rx(1.5707963267948966) q[7];
rz(1.5707963267948966) q[9];
x q[13];
id q[5];
rx(1.5707963267948966) q[19];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[12];
sdg q[9];
u1(1.5707963267948966) q[7];
rzz(1.5707963267948966) q[0], q[11];
p(0) q[6];
swap q[21], q[23];
rz(1.5707963267948966) q[22];
tdg q[8];
ch q[2], q[4];
swap q[5], q[20];
u3(0, 0, 1.5707963267948966) q[3];
cu3(1.5707963267948966, 0, 0) q[6], q[24];
x q[6];
id q[12];
h q[2];
tdg q[9];
id q[17];
u1(1.5707963267948966) q[22];
x q[1];
rzz(1.5707963267948966) q[21], q[5];
swap q[13], q[24];
u2(1.5707963267948966, 1.5707963267948966) q[6];
h q[5];
cu1(1.5707963267948966) q[17], q[3];
id q[23];
p(0) q[13];
h q[22];
u3(0, 0, 1.5707963267948966) q[3];
cu1(1.5707963267948966) q[17], q[19];
u3(0, 0, 1.5707963267948966) q[14];
rx(1.5707963267948966) q[24];
x q[5];
u2(1.5707963267948966, 1.5707963267948966) q[24];
id q[10];
rz(1.5707963267948966) q[14];
rx(1.5707963267948966) q[6];
u1(1.5707963267948966) q[24];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[16];
ry(1.5707963267948966) q[18];
u3(0, 0, 1.5707963267948966) q[8];
x q[23];
tdg q[2];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[17];
u3(0, 0, 1.5707963267948966) q[21];
cu3(1.5707963267948966, 0, 0) q[13], q[14];
sdg q[14];
cy q[21], q[7];
u1(1.5707963267948966) q[11];
u3(0, 0, 1.5707963267948966) q[23];
s q[10];
rz(1.5707963267948966) q[12];
u2(1.5707963267948966, 1.5707963267948966) q[22];
rzz(1.5707963267948966) q[18], q[23];
rz(1.5707963267948966) q[17];
t q[8];
t q[9];
rx(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[5];
p(0) q[21];
u1(1.5707963267948966) q[20];
u3(0, 0, 1.5707963267948966) q[15];
p(0) q[0];
s q[17];
sdg q[5];
id q[8];
h q[9];
h q[20];
rx(1.5707963267948966) q[9];