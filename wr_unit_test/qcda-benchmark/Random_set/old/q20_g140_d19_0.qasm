OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
h q[16];
id q[12];
sdg q[2];
tdg q[7];
u1(1.5707963267948966) q[19];
p(0) q[18];
tdg q[13];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[18];
s q[3];
x q[19];
h q[18];
id q[13];
u3(0, 0, 1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[15];
rx(1.5707963267948966) q[4];
rxx(0) q[9], q[6];
sdg q[6];
swap q[18], q[10];
swap q[9], q[10];
cu3(1.5707963267948966, 0, 0) q[2], q[4];
u2(1.5707963267948966, 1.5707963267948966) q[14];
cz q[2], q[15];
p(0) q[13];
crz(1.5707963267948966) q[10], q[13];
cz q[16], q[0];
u3(0, 0, 1.5707963267948966) q[11];
t q[11];
swap q[6], q[2];
ry(1.5707963267948966) q[14];
rz(1.5707963267948966) q[10];
t q[19];
ry(1.5707963267948966) q[0];
sdg q[15];
u1(1.5707963267948966) q[19];
rz(1.5707963267948966) q[13];
ryy(1.5707963267948966) q[3], q[2];
tdg q[7];
tdg q[19];
cu1(1.5707963267948966) q[12], q[18];
rz(1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[7];
x q[14];
h q[15];
x q[0];
tdg q[13];
h q[5];
u3(0, 0, 1.5707963267948966) q[5];
ch q[11], q[1];
t q[10];
cy q[2], q[1];
x q[19];
tdg q[1];
ry(1.5707963267948966) q[6];
s q[2];
x q[16];
id q[15];
rz(1.5707963267948966) q[1];
sdg q[11];
s q[8];
u1(1.5707963267948966) q[13];
x q[11];
t q[17];
s q[1];
ch q[8], q[17];
tdg q[17];
u2(1.5707963267948966, 1.5707963267948966) q[14];
ry(1.5707963267948966) q[17];
id q[14];
t q[4];
cu1(1.5707963267948966) q[15], q[2];
swap q[17], q[13];
id q[16];
cx q[13], q[9];
u1(1.5707963267948966) q[4];
id q[0];
ch q[14], q[9];
rxx(0) q[18], q[7];
p(0) q[10];
u1(1.5707963267948966) q[13];
t q[12];
x q[2];
h q[16];
id q[12];
rz(1.5707963267948966) q[10];
s q[14];
s q[3];
s q[16];
rz(1.5707963267948966) q[14];
rx(1.5707963267948966) q[9];
p(0) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[2];
cz q[9], q[1];
cu1(1.5707963267948966) q[3], q[14];
crz(1.5707963267948966) q[16], q[12];
u2(1.5707963267948966, 1.5707963267948966) q[14];
rz(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[12];
sdg q[17];
u3(0, 0, 1.5707963267948966) q[8];
u1(1.5707963267948966) q[8];
u1(1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[14];
ryy(1.5707963267948966) q[4], q[9];
t q[17];
cu1(1.5707963267948966) q[8], q[9];
x q[1];
t q[9];
swap q[2], q[5];
u2(1.5707963267948966, 1.5707963267948966) q[6];
crz(1.5707963267948966) q[12], q[18];
cx q[2], q[6];
h q[2];
rz(1.5707963267948966) q[11];
rzz(1.5707963267948966) q[12], q[3];
sdg q[0];
rzz(1.5707963267948966) q[10], q[0];
p(0) q[12];
u3(0, 0, 1.5707963267948966) q[17];
cu3(1.5707963267948966, 0, 0) q[7], q[17];
id q[18];
tdg q[17];
x q[11];
ch q[5], q[1];
id q[9];
x q[13];
crz(1.5707963267948966) q[12], q[2];
u1(1.5707963267948966) q[18];
t q[11];
sdg q[15];
u1(1.5707963267948966) q[13];
rz(1.5707963267948966) q[18];
x q[8];
cx q[18], q[16];
rz(1.5707963267948966) q[19];
cu3(1.5707963267948966, 0, 0) q[18], q[12];
cz q[8], q[4];
t q[9];
ry(1.5707963267948966) q[10];