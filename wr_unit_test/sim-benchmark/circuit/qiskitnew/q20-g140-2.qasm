OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
rxx(0) q[15], q[12];
sdg q[4];
id q[18];
ry(1.5707963267948966) q[9];
id q[7];
u2(1.5707963267948966, 1.5707963267948966) q[13];
u3(0, 0, 1.5707963267948966) q[8];
rz(1.5707963267948966) q[10];
ry(1.5707963267948966) q[18];
rzz(1.5707963267948966) q[5], q[15];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u1(1.5707963267948966) q[1];
t q[4];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[5];
sdg q[3];
rx(1.5707963267948966) q[17];
rx(1.5707963267948966) q[1];
rxx(0) q[14], q[5];
cy q[19], q[18];
t q[11];
u2(1.5707963267948966, 1.5707963267948966) q[16];
sdg q[0];
sdg q[3];
cu1(1.5707963267948966) q[16], q[7];
h q[17];
u3(0, 0, 1.5707963267948966) q[11];
rz(1.5707963267948966) q[10];
cu1(1.5707963267948966) q[13], q[10];
u1(1.5707963267948966) q[8];
rxx(0) q[16], q[10];
swap q[12], q[2];
cx q[1], q[12];
s q[11];
id q[16];
sdg q[9];
rxx(0) q[11], q[16];
h q[1];
p(0) q[18];
t q[13];
sdg q[5];
t q[9];
swap q[4], q[14];
tdg q[14];
cx q[9], q[18];
ry(1.5707963267948966) q[17];
u2(1.5707963267948966, 1.5707963267948966) q[11];
id q[7];
id q[8];
u2(1.5707963267948966, 1.5707963267948966) q[12];
s q[10];
cy q[19], q[10];
u1(1.5707963267948966) q[4];
s q[6];
u3(0, 0, 1.5707963267948966) q[9];
h q[2];
t q[10];
u2(1.5707963267948966, 1.5707963267948966) q[11];
sdg q[19];
rx(1.5707963267948966) q[19];
sdg q[4];
s q[13];
u2(1.5707963267948966, 1.5707963267948966) q[8];
rz(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[15];
u1(1.5707963267948966) q[1];
swap q[18], q[16];
u1(1.5707963267948966) q[0];
s q[0];
p(0) q[11];
rx(1.5707963267948966) q[3];
p(0) q[9];
cz q[15], q[8];
t q[4];
id q[4];
rx(1.5707963267948966) q[16];
h q[17];
ry(1.5707963267948966) q[0];
u1(1.5707963267948966) q[18];
u1(1.5707963267948966) q[18];
rx(1.5707963267948966) q[18];
u1(1.5707963267948966) q[4];
rx(1.5707963267948966) q[10];
tdg q[19];
u1(1.5707963267948966) q[7];
h q[9];
t q[10];
u2(1.5707963267948966, 1.5707963267948966) q[17];
cu3(1.5707963267948966, 0, 0) q[0], q[17];
rxx(0) q[14], q[11];
rx(1.5707963267948966) q[6];
cu3(1.5707963267948966, 0, 0) q[5], q[6];
u3(0, 0, 1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[17];
u2(1.5707963267948966, 1.5707963267948966) q[7];
h q[14];
cu1(1.5707963267948966) q[11], q[15];
p(0) q[16];
rz(1.5707963267948966) q[18];
swap q[12], q[13];
u1(1.5707963267948966) q[15];
p(0) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[13];
u3(0, 0, 1.5707963267948966) q[9];
tdg q[0];
id q[5];
sdg q[1];
rz(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[18];
tdg q[11];
id q[19];
sdg q[19];
t q[0];
ry(1.5707963267948966) q[17];
tdg q[11];
s q[15];
u2(1.5707963267948966, 1.5707963267948966) q[2];
id q[7];
ry(1.5707963267948966) q[4];
s q[17];
id q[6];
cx q[16], q[4];
p(0) q[15];
rz(1.5707963267948966) q[7];
cx q[7], q[18];
swap q[15], q[7];
u1(1.5707963267948966) q[12];
p(0) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[19];
rz(1.5707963267948966) q[18];
rxx(0) q[16], q[11];
tdg q[3];
rz(1.5707963267948966) q[19];
tdg q[9];
rz(1.5707963267948966) q[16];
swap q[3], q[2];
rx(1.5707963267948966) q[18];
u1(1.5707963267948966) q[10];
u3(0, 0, 1.5707963267948966) q[19];