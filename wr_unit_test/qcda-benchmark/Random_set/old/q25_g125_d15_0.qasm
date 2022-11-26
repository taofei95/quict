OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
creg c[25];
u1(1.5707963267948966) q[1];
sdg q[5];
x q[4];
x q[22];
t q[8];
rxx(0) q[0], q[18];
rz(1.5707963267948966) q[17];
u3(0, 0, 1.5707963267948966) q[6];
x q[7];
rx(1.5707963267948966) q[12];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rx(1.5707963267948966) q[2];
id q[0];
u1(1.5707963267948966) q[17];
x q[11];
u1(1.5707963267948966) q[20];
p(0) q[7];
tdg q[10];
x q[19];
u1(1.5707963267948966) q[0];
cu3(1.5707963267948966, 0, 0) q[1], q[18];
cu1(1.5707963267948966) q[14], q[15];
p(0) q[13];
rx(1.5707963267948966) q[0];
sdg q[19];
u1(1.5707963267948966) q[4];
s q[3];
rx(1.5707963267948966) q[14];
rx(1.5707963267948966) q[20];
tdg q[12];
ry(1.5707963267948966) q[20];
h q[6];
h q[12];
u3(0, 0, 1.5707963267948966) q[9];
p(0) q[7];
cu3(1.5707963267948966, 0, 0) q[9], q[24];
rz(1.5707963267948966) q[23];
x q[12];
cu3(1.5707963267948966, 0, 0) q[3], q[16];
h q[24];
x q[21];
cu3(1.5707963267948966, 0, 0) q[5], q[4];
rz(1.5707963267948966) q[24];
id q[0];
rz(1.5707963267948966) q[21];
sdg q[9];
h q[16];
ry(1.5707963267948966) q[19];
cz q[11], q[9];
h q[3];
h q[17];
p(0) q[10];
s q[18];
rx(1.5707963267948966) q[12];
h q[13];
cu1(1.5707963267948966) q[2], q[0];
p(0) q[24];
u2(1.5707963267948966, 1.5707963267948966) q[15];
u3(0, 0, 1.5707963267948966) q[20];
cx q[8], q[0];
ry(1.5707963267948966) q[1];
cx q[9], q[10];
s q[16];
u2(1.5707963267948966, 1.5707963267948966) q[0];
cu3(1.5707963267948966, 0, 0) q[0], q[20];
s q[7];
ch q[15], q[5];
ry(1.5707963267948966) q[17];
cy q[15], q[20];
ry(1.5707963267948966) q[17];
rz(1.5707963267948966) q[14];
u1(1.5707963267948966) q[0];
rx(1.5707963267948966) q[21];
rxx(0) q[7], q[22];
s q[6];
u2(1.5707963267948966, 1.5707963267948966) q[11];
h q[9];
u3(0, 0, 1.5707963267948966) q[20];
ch q[11], q[7];
cz q[22], q[6];
h q[5];
sdg q[4];
h q[0];
cu1(1.5707963267948966) q[13], q[7];
id q[7];
rx(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[4];
id q[20];
rz(1.5707963267948966) q[14];
swap q[7], q[3];
p(0) q[14];
rx(1.5707963267948966) q[12];
u2(1.5707963267948966, 1.5707963267948966) q[15];
u2(1.5707963267948966, 1.5707963267948966) q[11];
tdg q[13];
rz(1.5707963267948966) q[19];
rz(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[2];
rz(1.5707963267948966) q[9];
rxx(0) q[4], q[9];
tdg q[20];
p(0) q[0];
t q[4];
p(0) q[24];
s q[20];
cu3(1.5707963267948966, 0, 0) q[18], q[23];
rx(1.5707963267948966) q[7];
cx q[2], q[14];
rx(1.5707963267948966) q[23];
cu3(1.5707963267948966, 0, 0) q[8], q[19];
u3(0, 0, 1.5707963267948966) q[16];
t q[15];
u3(0, 0, 1.5707963267948966) q[1];
t q[21];
u1(1.5707963267948966) q[2];
tdg q[0];
h q[4];
x q[17];
u1(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[0];
sdg q[20];
s q[21];
p(0) q[23];
u2(1.5707963267948966, 1.5707963267948966) q[19];
rx(1.5707963267948966) q[13];