OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
creg c[25];
s q[19];
h q[20];
u3(0, 0, 1.5707963267948966) q[12];
p(0) q[23];
cx q[16], q[15];
t q[24];
rzz(1.5707963267948966) q[18], q[11];
id q[14];
t q[2];
cx q[7], q[14];
sdg q[14];
cu1(1.5707963267948966) q[4], q[20];
tdg q[11];
u1(1.5707963267948966) q[12];
cu3(1.5707963267948966, 0, 0) q[3], q[17];
t q[4];
s q[1];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[22];
u3(0, 0, 1.5707963267948966) q[0];
sdg q[14];
p(0) q[12];
swap q[2], q[11];
tdg q[16];
tdg q[2];
crz(1.5707963267948966) q[23], q[17];
s q[4];
ryy(1.5707963267948966) q[22], q[5];
u3(0, 0, 1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[4];
ry(1.5707963267948966) q[16];
rz(1.5707963267948966) q[14];
u2(1.5707963267948966, 1.5707963267948966) q[21];
u1(1.5707963267948966) q[9];
swap q[8], q[16];
t q[23];
rz(1.5707963267948966) q[5];
sdg q[0];
ry(1.5707963267948966) q[14];
u3(0, 0, 1.5707963267948966) q[3];
id q[5];
t q[17];
id q[11];
u1(1.5707963267948966) q[5];
cz q[20], q[22];
h q[13];
id q[23];
tdg q[17];
tdg q[9];
s q[5];
ry(1.5707963267948966) q[20];
rz(1.5707963267948966) q[1];
tdg q[7];
cy q[11], q[12];
tdg q[17];
ry(1.5707963267948966) q[11];
ryy(1.5707963267948966) q[18], q[7];
crz(1.5707963267948966) q[17], q[18];
s q[7];
rzz(1.5707963267948966) q[16], q[7];
u3(0, 0, 1.5707963267948966) q[21];
u2(1.5707963267948966, 1.5707963267948966) q[5];
tdg q[4];
sdg q[5];
s q[16];
cz q[3], q[17];
rz(1.5707963267948966) q[17];
u2(1.5707963267948966, 1.5707963267948966) q[23];
u2(1.5707963267948966, 1.5707963267948966) q[13];
tdg q[12];
u1(1.5707963267948966) q[10];
cy q[7], q[4];
sdg q[18];
u2(1.5707963267948966, 1.5707963267948966) q[5];
tdg q[15];
u3(0, 0, 1.5707963267948966) q[6];
t q[18];
id q[2];
sdg q[6];
sdg q[17];
h q[20];
x q[20];
u2(1.5707963267948966, 1.5707963267948966) q[10];
x q[20];
rxx(0) q[19], q[9];
crz(1.5707963267948966) q[10], q[16];
tdg q[23];
rz(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[18];
ry(1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[24];
h q[15];
rz(1.5707963267948966) q[0];
s q[13];
cz q[15], q[13];
id q[9];
u3(0, 0, 1.5707963267948966) q[14];
h q[0];
crz(1.5707963267948966) q[2], q[12];
s q[15];
crz(1.5707963267948966) q[15], q[0];
rxx(0) q[20], q[11];
cu3(1.5707963267948966, 0, 0) q[3], q[8];
id q[1];
h q[4];
u1(1.5707963267948966) q[16];
p(0) q[18];
ch q[22], q[1];
t q[23];
x q[0];
rx(1.5707963267948966) q[22];
cu3(1.5707963267948966, 0, 0) q[12], q[18];
x q[20];
x q[19];
rx(1.5707963267948966) q[0];
sdg q[10];
cu1(1.5707963267948966) q[10], q[20];
rz(1.5707963267948966) q[17];
rz(1.5707963267948966) q[13];
rx(1.5707963267948966) q[2];
t q[10];
crz(1.5707963267948966) q[2], q[0];
ry(1.5707963267948966) q[20];
u2(1.5707963267948966, 1.5707963267948966) q[24];
cu3(1.5707963267948966, 0, 0) q[5], q[22];