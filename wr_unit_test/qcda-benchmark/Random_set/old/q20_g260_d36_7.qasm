OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
rx(1.5707963267948966) q[5];
sdg q[8];
s q[1];
t q[0];
crz(1.5707963267948966) q[15], q[8];
tdg q[0];
p(0) q[9];
h q[4];
rxx(0) q[0], q[4];
h q[17];
id q[7];
crz(1.5707963267948966) q[0], q[6];
x q[18];
p(0) q[1];
u3(0, 0, 1.5707963267948966) q[6];
rz(1.5707963267948966) q[11];
ryy(1.5707963267948966) q[15], q[3];
p(0) q[4];
cx q[18], q[0];
u2(1.5707963267948966, 1.5707963267948966) q[2];
p(0) q[14];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[2];
u1(1.5707963267948966) q[11];
sdg q[6];
h q[15];
t q[6];
crz(1.5707963267948966) q[5], q[15];
h q[7];
t q[6];
rxx(0) q[8], q[10];
ry(1.5707963267948966) q[10];
s q[0];
swap q[12], q[19];
ry(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[14];
id q[16];
cx q[17], q[18];
tdg q[4];
sdg q[18];
tdg q[18];
u2(1.5707963267948966, 1.5707963267948966) q[9];
crz(1.5707963267948966) q[0], q[16];
x q[11];
ry(1.5707963267948966) q[19];
p(0) q[15];
crz(1.5707963267948966) q[16], q[5];
rz(1.5707963267948966) q[16];
ry(1.5707963267948966) q[14];
ry(1.5707963267948966) q[14];
t q[19];
cz q[4], q[13];
h q[0];
sdg q[3];
h q[13];
u3(0, 0, 1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[5];
tdg q[8];
rx(1.5707963267948966) q[3];
rx(1.5707963267948966) q[3];
sdg q[10];
ry(1.5707963267948966) q[10];
sdg q[0];
id q[14];
p(0) q[5];
ry(1.5707963267948966) q[9];
t q[5];
ryy(1.5707963267948966) q[13], q[6];
h q[6];
u2(1.5707963267948966, 1.5707963267948966) q[15];
tdg q[13];
ryy(1.5707963267948966) q[11], q[12];
p(0) q[10];
u1(1.5707963267948966) q[11];
sdg q[9];
cz q[9], q[12];
cx q[2], q[1];
s q[6];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[11];
s q[1];
u2(1.5707963267948966, 1.5707963267948966) q[6];
x q[6];
ry(1.5707963267948966) q[5];
cu3(1.5707963267948966, 0, 0) q[7], q[12];
sdg q[17];
u3(0, 0, 1.5707963267948966) q[8];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[1];
x q[10];
crz(1.5707963267948966) q[10], q[17];
id q[7];
cz q[6], q[8];
rx(1.5707963267948966) q[16];
sdg q[5];
cz q[9], q[5];
u1(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[12];
tdg q[1];
tdg q[8];
x q[12];
t q[12];
id q[0];
tdg q[12];
t q[4];
tdg q[11];
h q[15];
t q[14];
rz(1.5707963267948966) q[1];
tdg q[15];
ry(1.5707963267948966) q[16];
cx q[2], q[7];
tdg q[11];
ry(1.5707963267948966) q[17];
s q[5];
h q[17];
cy q[3], q[8];
p(0) q[5];
sdg q[12];
cu3(1.5707963267948966, 0, 0) q[11], q[18];
sdg q[16];
s q[2];
sdg q[5];
s q[13];
h q[18];
rx(1.5707963267948966) q[10];
crz(1.5707963267948966) q[11], q[5];
u2(1.5707963267948966, 1.5707963267948966) q[1];
id q[14];
t q[3];
rx(1.5707963267948966) q[15];
swap q[11], q[17];
t q[5];
u1(1.5707963267948966) q[5];
rx(1.5707963267948966) q[14];
sdg q[2];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[13];
rx(1.5707963267948966) q[8];
ch q[2], q[7];
s q[19];
rxx(0) q[17], q[15];
u2(1.5707963267948966, 1.5707963267948966) q[13];
u1(1.5707963267948966) q[8];
s q[12];
x q[15];
rxx(0) q[19], q[4];
u3(0, 0, 1.5707963267948966) q[12];
cy q[3], q[11];
p(0) q[4];
u1(1.5707963267948966) q[11];
s q[3];
sdg q[9];
id q[7];
sdg q[18];
rx(1.5707963267948966) q[3];
rx(1.5707963267948966) q[18];
t q[4];
rz(1.5707963267948966) q[4];
tdg q[5];
x q[18];
ry(1.5707963267948966) q[11];
cu3(1.5707963267948966, 0, 0) q[3], q[16];
s q[15];
s q[6];
x q[4];
h q[4];
sdg q[12];
tdg q[11];
tdg q[5];
p(0) q[14];
cx q[5], q[1];
ry(1.5707963267948966) q[19];
u1(1.5707963267948966) q[9];
sdg q[10];
tdg q[11];
swap q[0], q[6];
rx(1.5707963267948966) q[11];
tdg q[7];
u2(1.5707963267948966, 1.5707963267948966) q[2];
x q[10];
p(0) q[10];
x q[11];
ryy(1.5707963267948966) q[6], q[11];
p(0) q[8];
cy q[16], q[6];
x q[9];
sdg q[8];
x q[1];
sdg q[1];
p(0) q[14];
u1(1.5707963267948966) q[13];
p(0) q[17];
u3(0, 0, 1.5707963267948966) q[19];
tdg q[19];
id q[19];
u1(1.5707963267948966) q[7];
rz(1.5707963267948966) q[10];
tdg q[0];
p(0) q[17];
ch q[1], q[19];
ry(1.5707963267948966) q[15];
u1(1.5707963267948966) q[6];
sdg q[9];
t q[13];
id q[3];
u1(1.5707963267948966) q[11];
s q[17];
s q[17];
id q[0];
t q[13];
ch q[6], q[2];
sdg q[17];
sdg q[7];
u1(1.5707963267948966) q[19];
id q[11];
ry(1.5707963267948966) q[18];
tdg q[12];
tdg q[8];
t q[0];
ry(1.5707963267948966) q[9];
id q[3];
x q[17];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[15];
s q[19];
u2(1.5707963267948966, 1.5707963267948966) q[12];
p(0) q[4];
swap q[11], q[12];
cu1(1.5707963267948966) q[5], q[16];
rz(1.5707963267948966) q[6];
s q[18];
id q[4];
cz q[4], q[12];
t q[6];
p(0) q[3];
id q[6];
swap q[6], q[7];
u3(0, 0, 1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[1];
sdg q[2];
ch q[18], q[5];
tdg q[8];
rx(1.5707963267948966) q[6];
tdg q[13];
ry(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[11];
t q[0];
u3(0, 0, 1.5707963267948966) q[6];
tdg q[19];
u2(1.5707963267948966, 1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[18];
s q[8];
cy q[3], q[10];
u2(1.5707963267948966, 1.5707963267948966) q[16];
u3(0, 0, 1.5707963267948966) q[8];
h q[5];
p(0) q[14];
ry(1.5707963267948966) q[19];
rz(1.5707963267948966) q[0];