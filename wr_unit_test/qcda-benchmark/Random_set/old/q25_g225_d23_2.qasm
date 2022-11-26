OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
creg c[25];
u2(1.5707963267948966, 1.5707963267948966) q[19];
sdg q[6];
rz(1.5707963267948966) q[2];
p(0) q[13];
id q[13];
swap q[18], q[6];
u3(0, 0, 1.5707963267948966) q[10];
u1(1.5707963267948966) q[2];
rx(1.5707963267948966) q[12];
ry(1.5707963267948966) q[2];
p(0) q[23];
p(0) q[6];
u3(0, 0, 1.5707963267948966) q[19];
ry(1.5707963267948966) q[11];
cu3(1.5707963267948966, 0, 0) q[15], q[9];
u3(0, 0, 1.5707963267948966) q[23];
tdg q[0];
ry(1.5707963267948966) q[21];
tdg q[10];
sdg q[15];
u3(0, 0, 1.5707963267948966) q[5];
cy q[19], q[10];
p(0) q[15];
u1(1.5707963267948966) q[15];
u2(1.5707963267948966, 1.5707963267948966) q[15];
t q[13];
u1(1.5707963267948966) q[7];
rxx(0) q[14], q[4];
x q[1];
t q[23];
u3(0, 0, 1.5707963267948966) q[18];
sdg q[19];
cy q[8], q[23];
h q[20];
swap q[4], q[3];
ry(1.5707963267948966) q[3];
tdg q[0];
tdg q[9];
h q[23];
cu1(1.5707963267948966) q[7], q[22];
sdg q[11];
cz q[17], q[13];
p(0) q[19];
u3(0, 0, 1.5707963267948966) q[22];
cu1(1.5707963267948966) q[3], q[12];
t q[17];
s q[18];
t q[9];
rz(1.5707963267948966) q[24];
t q[19];
u1(1.5707963267948966) q[20];
ry(1.5707963267948966) q[5];
swap q[19], q[18];
x q[0];
tdg q[10];
ch q[4], q[19];
u2(1.5707963267948966, 1.5707963267948966) q[4];
u1(1.5707963267948966) q[20];
sdg q[6];
swap q[0], q[14];
u1(1.5707963267948966) q[5];
s q[11];
h q[11];
cz q[9], q[22];
h q[14];
cu3(1.5707963267948966, 0, 0) q[13], q[19];
rx(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[20];
sdg q[16];
ch q[6], q[19];
rx(1.5707963267948966) q[19];
h q[11];
s q[5];
p(0) q[5];
h q[7];
cz q[9], q[12];
tdg q[14];
p(0) q[14];
u1(1.5707963267948966) q[13];
rz(1.5707963267948966) q[10];
u1(1.5707963267948966) q[2];
x q[23];
tdg q[8];
s q[1];
rz(1.5707963267948966) q[17];
s q[15];
tdg q[21];
id q[20];
swap q[1], q[8];
sdg q[17];
rzz(1.5707963267948966) q[10], q[17];
id q[7];
u2(1.5707963267948966, 1.5707963267948966) q[17];
h q[23];
crz(1.5707963267948966) q[15], q[17];
s q[18];
p(0) q[9];
rz(1.5707963267948966) q[21];
s q[22];
h q[14];
u3(0, 0, 1.5707963267948966) q[9];
u1(1.5707963267948966) q[5];
ry(1.5707963267948966) q[20];
id q[10];
u2(1.5707963267948966, 1.5707963267948966) q[0];
id q[21];
s q[12];
id q[17];
id q[17];
s q[11];
u2(1.5707963267948966, 1.5707963267948966) q[6];
cz q[19], q[18];
id q[13];
u2(1.5707963267948966, 1.5707963267948966) q[24];
u1(1.5707963267948966) q[8];
s q[2];
p(0) q[14];
cz q[17], q[6];
x q[1];
s q[5];
u2(1.5707963267948966, 1.5707963267948966) q[6];
tdg q[19];
p(0) q[16];
swap q[14], q[5];
rz(1.5707963267948966) q[22];
rzz(1.5707963267948966) q[5], q[9];
u2(1.5707963267948966, 1.5707963267948966) q[21];
sdg q[14];
h q[0];
rxx(0) q[16], q[1];
cu3(1.5707963267948966, 0, 0) q[1], q[0];
cx q[7], q[1];
rz(1.5707963267948966) q[9];
p(0) q[21];
u2(1.5707963267948966, 1.5707963267948966) q[15];
rz(1.5707963267948966) q[6];
h q[21];
t q[23];
u3(0, 0, 1.5707963267948966) q[8];
s q[12];
rxx(0) q[11], q[23];
sdg q[17];
t q[6];
rxx(0) q[17], q[1];
t q[0];
ryy(1.5707963267948966) q[23], q[0];
cu1(1.5707963267948966) q[16], q[22];
ry(1.5707963267948966) q[11];
t q[14];
u3(0, 0, 1.5707963267948966) q[14];
h q[20];
rx(1.5707963267948966) q[18];
sdg q[9];
s q[1];
rx(1.5707963267948966) q[0];
s q[11];
u3(0, 0, 1.5707963267948966) q[10];
s q[10];
h q[14];
tdg q[15];
id q[23];
h q[12];
p(0) q[10];
cz q[19], q[3];
rz(1.5707963267948966) q[7];
cu3(1.5707963267948966, 0, 0) q[3], q[1];
ry(1.5707963267948966) q[21];
id q[12];
rz(1.5707963267948966) q[21];
ch q[2], q[19];
h q[14];
h q[15];
rzz(1.5707963267948966) q[0], q[16];
x q[14];
u3(0, 0, 1.5707963267948966) q[1];
sdg q[22];
ry(1.5707963267948966) q[6];
sdg q[4];
cu3(1.5707963267948966, 0, 0) q[18], q[13];
u2(1.5707963267948966, 1.5707963267948966) q[20];
sdg q[24];
u3(0, 0, 1.5707963267948966) q[0];
rx(1.5707963267948966) q[0];
rxx(0) q[18], q[5];
h q[0];
s q[23];
tdg q[19];
u3(0, 0, 1.5707963267948966) q[12];
u1(1.5707963267948966) q[16];
cx q[23], q[4];
crz(1.5707963267948966) q[12], q[10];
h q[11];
id q[6];
u2(1.5707963267948966, 1.5707963267948966) q[17];
u2(1.5707963267948966, 1.5707963267948966) q[23];
x q[11];
id q[4];
ry(1.5707963267948966) q[6];
s q[1];
swap q[8], q[1];
cz q[11], q[15];
p(0) q[22];
rz(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[1];
p(0) q[12];
id q[12];
sdg q[17];
u2(1.5707963267948966, 1.5707963267948966) q[8];
x q[5];
id q[12];
id q[18];
ry(1.5707963267948966) q[6];
cz q[21], q[19];
rx(1.5707963267948966) q[1];
t q[2];
s q[11];
cx q[2], q[11];
ry(1.5707963267948966) q[8];
rx(1.5707963267948966) q[22];
u1(1.5707963267948966) q[10];
u3(0, 0, 1.5707963267948966) q[8];
p(0) q[21];
t q[9];
u2(1.5707963267948966, 1.5707963267948966) q[14];
t q[22];