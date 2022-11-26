OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
cu1(1.5707963267948966) q[17], q[9];
cz q[0], q[8];
id q[12];
s q[9];
u3(0, 0, 1.5707963267948966) q[2];
u1(1.5707963267948966) q[11];
ry(1.5707963267948966) q[4];
t q[5];
s q[19];
u1(1.5707963267948966) q[0];
rzz(1.5707963267948966) q[15], q[2];
rzz(1.5707963267948966) q[16], q[6];
rzz(1.5707963267948966) q[10], q[1];
tdg q[9];
s q[7];
id q[17];
t q[1];
cx q[14], q[5];
p(0) q[7];
p(0) q[7];
cu3(1.5707963267948966, 0, 0) q[15], q[18];
p(0) q[11];
cz q[10], q[9];
cy q[19], q[8];
t q[12];
cu3(1.5707963267948966, 0, 0) q[14], q[5];
u3(0, 0, 1.5707963267948966) q[3];
rzz(1.5707963267948966) q[13], q[8];
swap q[6], q[0];
cy q[3], q[11];
rx(1.5707963267948966) q[10];
h q[9];
h q[15];
cz q[15], q[4];
id q[16];
p(0) q[19];
cu1(1.5707963267948966) q[2], q[1];
s q[13];
rz(1.5707963267948966) q[18];
ry(1.5707963267948966) q[12];
tdg q[7];
h q[4];
id q[11];
cu3(1.5707963267948966, 0, 0) q[1], q[5];
tdg q[8];
cu1(1.5707963267948966) q[11], q[9];
s q[12];
u1(1.5707963267948966) q[16];
s q[18];
h q[6];
rx(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[15];
s q[9];
cx q[9], q[7];
t q[16];
ry(1.5707963267948966) q[8];
t q[18];
ry(1.5707963267948966) q[5];
p(0) q[5];
u1(1.5707963267948966) q[5];
p(0) q[5];
t q[16];
rxx(0) q[18], q[3];
id q[3];
ry(1.5707963267948966) q[14];
sdg q[4];
rxx(0) q[8], q[18];
ry(1.5707963267948966) q[12];
cy q[3], q[12];
tdg q[14];
u2(1.5707963267948966, 1.5707963267948966) q[7];
s q[19];
u1(1.5707963267948966) q[3];
rz(1.5707963267948966) q[11];
swap q[8], q[9];
p(0) q[15];
cu3(1.5707963267948966, 0, 0) q[2], q[3];
sdg q[15];
rx(1.5707963267948966) q[5];
rz(1.5707963267948966) q[4];
swap q[4], q[13];
swap q[11], q[8];
swap q[18], q[12];
h q[0];
tdg q[14];
rzz(1.5707963267948966) q[12], q[7];
t q[15];
p(0) q[6];
t q[12];
rxx(0) q[18], q[8];
id q[14];
p(0) q[0];
t q[7];
u1(1.5707963267948966) q[7];
t q[10];
tdg q[7];
rzz(1.5707963267948966) q[4], q[18];
u2(1.5707963267948966, 1.5707963267948966) q[18];
tdg q[14];
u1(1.5707963267948966) q[4];
rzz(1.5707963267948966) q[17], q[9];
p(0) q[14];
rz(1.5707963267948966) q[6];
u1(1.5707963267948966) q[15];
u1(1.5707963267948966) q[9];
rx(1.5707963267948966) q[15];
id q[9];
rx(1.5707963267948966) q[11];
h q[4];
rx(1.5707963267948966) q[14];
tdg q[18];
p(0) q[0];
tdg q[15];
t q[8];
u3(0, 0, 1.5707963267948966) q[9];
p(0) q[2];
sdg q[15];
rx(1.5707963267948966) q[0];
h q[6];
cu1(1.5707963267948966) q[10], q[2];
cy q[4], q[16];
u2(1.5707963267948966, 1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[9];
rx(1.5707963267948966) q[6];
tdg q[1];
rx(1.5707963267948966) q[17];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[17];
swap q[8], q[19];
s q[10];
p(0) q[17];
s q[10];
rxx(0) q[16], q[3];
u3(0, 0, 1.5707963267948966) q[12];
u1(1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[18];
id q[6];
id q[13];
u1(1.5707963267948966) q[17];
tdg q[10];
tdg q[1];
id q[5];
sdg q[7];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[2];
s q[7];
id q[7];
p(0) q[1];
id q[7];
rxx(0) q[8], q[7];
u2(1.5707963267948966, 1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[12];
rx(1.5707963267948966) q[19];
ry(1.5707963267948966) q[12];
h q[8];
tdg q[14];
tdg q[17];
t q[1];
p(0) q[12];
ry(1.5707963267948966) q[1];
swap q[1], q[5];
rx(1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[19];
tdg q[5];
tdg q[18];
ry(1.5707963267948966) q[15];
u2(1.5707963267948966, 1.5707963267948966) q[2];
cx q[3], q[1];
rx(1.5707963267948966) q[16];
rxx(0) q[8], q[10];
ry(1.5707963267948966) q[15];
u3(0, 0, 1.5707963267948966) q[17];
h q[13];
t q[1];
ry(1.5707963267948966) q[6];
id q[3];
id q[1];
ry(1.5707963267948966) q[7];
h q[19];
t q[19];