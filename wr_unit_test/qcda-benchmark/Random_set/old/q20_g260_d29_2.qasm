OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
rx(1.5707963267948966) q[11];
cx q[10], q[0];
ryy(1.5707963267948966) q[10], q[7];
u2(1.5707963267948966, 1.5707963267948966) q[18];
h q[10];
rzz(1.5707963267948966) q[15], q[6];
s q[2];
rzz(1.5707963267948966) q[6], q[19];
ry(1.5707963267948966) q[19];
s q[6];
s q[4];
s q[0];
sdg q[13];
h q[6];
rx(1.5707963267948966) q[7];
crz(1.5707963267948966) q[9], q[19];
t q[14];
cy q[13], q[8];
swap q[16], q[1];
sdg q[0];
s q[18];
cx q[13], q[1];
ry(1.5707963267948966) q[18];
u3(0, 0, 1.5707963267948966) q[0];
t q[3];
rxx(0) q[11], q[7];
tdg q[13];
rz(1.5707963267948966) q[11];
sdg q[12];
x q[16];
id q[10];
h q[15];
tdg q[2];
t q[13];
ry(1.5707963267948966) q[9];
h q[4];
u2(1.5707963267948966, 1.5707963267948966) q[8];
tdg q[18];
rx(1.5707963267948966) q[12];
rzz(1.5707963267948966) q[7], q[0];
cu3(1.5707963267948966, 0, 0) q[2], q[4];
rx(1.5707963267948966) q[9];
u1(1.5707963267948966) q[3];
cy q[2], q[3];
ry(1.5707963267948966) q[12];
u3(0, 0, 1.5707963267948966) q[12];
s q[15];
u3(0, 0, 1.5707963267948966) q[17];
u3(0, 0, 1.5707963267948966) q[3];
h q[8];
ryy(1.5707963267948966) q[5], q[1];
id q[13];
rx(1.5707963267948966) q[17];
h q[10];
u1(1.5707963267948966) q[4];
cz q[2], q[5];
h q[16];
x q[4];
ch q[19], q[12];
cu3(1.5707963267948966, 0, 0) q[4], q[6];
h q[7];
u3(0, 0, 1.5707963267948966) q[14];
cz q[0], q[3];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[13];
u1(1.5707963267948966) q[9];
sdg q[12];
u3(0, 0, 1.5707963267948966) q[2];
cx q[2], q[4];
h q[3];
tdg q[2];
rzz(1.5707963267948966) q[13], q[5];
t q[12];
sdg q[8];
tdg q[14];
t q[15];
rxx(0) q[14], q[8];
ry(1.5707963267948966) q[5];
crz(1.5707963267948966) q[11], q[12];
u1(1.5707963267948966) q[11];
ch q[1], q[16];
h q[11];
s q[2];
x q[1];
u2(1.5707963267948966, 1.5707963267948966) q[5];
t q[15];
crz(1.5707963267948966) q[6], q[15];
rxx(0) q[6], q[1];
u1(1.5707963267948966) q[9];
s q[7];
p(0) q[15];
tdg q[12];
s q[16];
u1(1.5707963267948966) q[2];
p(0) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[16];
h q[9];
u2(1.5707963267948966, 1.5707963267948966) q[1];
id q[4];
u2(1.5707963267948966, 1.5707963267948966) q[15];
rz(1.5707963267948966) q[18];
s q[13];
tdg q[17];
x q[17];
ry(1.5707963267948966) q[6];
id q[11];
cz q[2], q[1];
t q[11];
h q[2];
id q[3];
x q[9];
s q[4];
s q[1];
id q[4];
t q[4];
ry(1.5707963267948966) q[16];
tdg q[11];
rx(1.5707963267948966) q[10];
h q[0];
tdg q[18];
u1(1.5707963267948966) q[14];
ry(1.5707963267948966) q[14];
s q[13];
cx q[17], q[5];
s q[2];
swap q[15], q[19];
ryy(1.5707963267948966) q[13], q[12];
cu3(1.5707963267948966, 0, 0) q[16], q[3];
h q[8];
s q[5];
tdg q[3];
ry(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[17];
s q[1];
u1(1.5707963267948966) q[10];
u1(1.5707963267948966) q[18];
rz(1.5707963267948966) q[13];
ry(1.5707963267948966) q[2];
p(0) q[3];
sdg q[1];
rx(1.5707963267948966) q[3];
h q[4];
cu1(1.5707963267948966) q[12], q[4];
rx(1.5707963267948966) q[1];
id q[2];
rz(1.5707963267948966) q[4];
sdg q[9];
u1(1.5707963267948966) q[4];
swap q[16], q[5];
crz(1.5707963267948966) q[9], q[4];
u2(1.5707963267948966, 1.5707963267948966) q[5];
t q[11];
id q[6];
ryy(1.5707963267948966) q[9], q[5];
s q[11];
u3(0, 0, 1.5707963267948966) q[16];
p(0) q[6];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[9];
ch q[5], q[19];
rx(1.5707963267948966) q[11];
p(0) q[13];
u3(0, 0, 1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[8];
id q[17];
tdg q[7];
sdg q[7];
u3(0, 0, 1.5707963267948966) q[19];
h q[12];
swap q[16], q[5];
s q[4];
rz(1.5707963267948966) q[16];
u1(1.5707963267948966) q[14];
h q[12];
id q[8];
tdg q[4];
p(0) q[18];
u3(0, 0, 1.5707963267948966) q[3];
s q[1];
p(0) q[1];
t q[12];
s q[18];
tdg q[4];
ryy(1.5707963267948966) q[14], q[11];
u1(1.5707963267948966) q[5];
sdg q[9];
t q[19];
crz(1.5707963267948966) q[5], q[13];
rz(1.5707963267948966) q[9];
t q[3];
rx(1.5707963267948966) q[0];
t q[4];
sdg q[18];
s q[19];
x q[1];
cu1(1.5707963267948966) q[4], q[11];
u1(1.5707963267948966) q[14];
s q[1];
x q[13];
tdg q[8];
u3(0, 0, 1.5707963267948966) q[13];
p(0) q[17];
rz(1.5707963267948966) q[10];
u3(0, 0, 1.5707963267948966) q[14];
id q[8];
u1(1.5707963267948966) q[19];
u1(1.5707963267948966) q[12];
x q[8];
h q[3];
u1(1.5707963267948966) q[11];
x q[5];
t q[13];
rx(1.5707963267948966) q[6];
crz(1.5707963267948966) q[1], q[8];
id q[5];
h q[7];
x q[19];
x q[9];
x q[6];
cz q[0], q[13];
ry(1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[17];
sdg q[9];
rz(1.5707963267948966) q[1];
tdg q[14];
x q[9];
id q[8];
ryy(1.5707963267948966) q[6], q[9];
cy q[13], q[1];
ry(1.5707963267948966) q[5];
p(0) q[8];
p(0) q[13];
t q[12];
rx(1.5707963267948966) q[6];
ry(1.5707963267948966) q[8];
p(0) q[10];
id q[12];
tdg q[14];
id q[0];
crz(1.5707963267948966) q[2], q[19];
id q[17];
id q[7];
rx(1.5707963267948966) q[1];
tdg q[6];
s q[19];
p(0) q[8];
crz(1.5707963267948966) q[18], q[1];
t q[3];
ryy(1.5707963267948966) q[7], q[9];
cx q[2], q[11];
u3(0, 0, 1.5707963267948966) q[19];
u1(1.5707963267948966) q[4];
x q[2];
rx(1.5707963267948966) q[13];
ry(1.5707963267948966) q[1];
h q[10];
cx q[17], q[16];
u3(0, 0, 1.5707963267948966) q[16];