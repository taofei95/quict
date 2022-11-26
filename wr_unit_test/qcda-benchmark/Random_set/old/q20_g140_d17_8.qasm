OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
tdg q[14];
h q[17];
ch q[10], q[13];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[7];
ry(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[11];
s q[18];
rx(1.5707963267948966) q[17];
tdg q[6];
h q[13];
cu1(1.5707963267948966) q[9], q[6];
tdg q[7];
tdg q[11];
rx(1.5707963267948966) q[9];
sdg q[18];
t q[10];
cu3(1.5707963267948966, 0, 0) q[19], q[9];
u3(0, 0, 1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[5];
t q[1];
tdg q[6];
t q[3];
u1(1.5707963267948966) q[7];
rz(1.5707963267948966) q[13];
t q[17];
rz(1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[7];
cz q[11], q[4];
x q[4];
p(0) q[5];
x q[19];
cx q[5], q[6];
rxx(0) q[17], q[14];
id q[19];
t q[9];
u3(0, 0, 1.5707963267948966) q[19];
s q[19];
x q[7];
u1(1.5707963267948966) q[6];
id q[2];
tdg q[3];
ryy(1.5707963267948966) q[8], q[15];
u3(0, 0, 1.5707963267948966) q[13];
rx(1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[16];
rz(1.5707963267948966) q[18];
u2(1.5707963267948966, 1.5707963267948966) q[16];
rx(1.5707963267948966) q[15];
rz(1.5707963267948966) q[12];
id q[5];
rz(1.5707963267948966) q[11];
ry(1.5707963267948966) q[19];
sdg q[1];
id q[3];
h q[4];
t q[8];
s q[13];
id q[13];
u2(1.5707963267948966, 1.5707963267948966) q[15];
t q[14];
ch q[15], q[12];
ch q[8], q[19];
u1(1.5707963267948966) q[14];
sdg q[10];
tdg q[5];
rz(1.5707963267948966) q[16];
p(0) q[3];
rx(1.5707963267948966) q[1];
cx q[16], q[6];
id q[6];
u2(1.5707963267948966, 1.5707963267948966) q[16];
rx(1.5707963267948966) q[10];
sdg q[6];
u1(1.5707963267948966) q[13];
id q[3];
rz(1.5707963267948966) q[5];
s q[18];
tdg q[0];
id q[18];
u2(1.5707963267948966, 1.5707963267948966) q[3];
rzz(1.5707963267948966) q[15], q[11];
rxx(0) q[3], q[5];
t q[1];
t q[15];
u2(1.5707963267948966, 1.5707963267948966) q[18];
id q[13];
crz(1.5707963267948966) q[10], q[6];
tdg q[6];
h q[11];
u3(0, 0, 1.5707963267948966) q[12];
p(0) q[12];
cu3(1.5707963267948966, 0, 0) q[7], q[19];
t q[13];
swap q[16], q[9];
h q[15];
rz(1.5707963267948966) q[17];
t q[9];
t q[5];
u2(1.5707963267948966, 1.5707963267948966) q[19];
h q[11];
u1(1.5707963267948966) q[7];
rx(1.5707963267948966) q[3];
cu1(1.5707963267948966) q[18], q[2];
cy q[3], q[2];
cy q[8], q[13];
p(0) q[11];
cu3(1.5707963267948966, 0, 0) q[10], q[16];
id q[13];
cz q[12], q[3];
u2(1.5707963267948966, 1.5707963267948966) q[19];
t q[17];
t q[18];
tdg q[2];
ry(1.5707963267948966) q[1];
u1(1.5707963267948966) q[1];
tdg q[12];
u2(1.5707963267948966, 1.5707963267948966) q[7];
ch q[18], q[4];
u2(1.5707963267948966, 1.5707963267948966) q[13];
t q[14];
rx(1.5707963267948966) q[8];
t q[14];
p(0) q[4];
id q[1];
ryy(1.5707963267948966) q[7], q[4];
cz q[1], q[7];
sdg q[8];
p(0) q[17];
rx(1.5707963267948966) q[18];
swap q[7], q[14];
ryy(1.5707963267948966) q[15], q[9];
u1(1.5707963267948966) q[7];
tdg q[1];
rx(1.5707963267948966) q[13];
u1(1.5707963267948966) q[14];
rz(1.5707963267948966) q[13];
s q[13];
s q[15];