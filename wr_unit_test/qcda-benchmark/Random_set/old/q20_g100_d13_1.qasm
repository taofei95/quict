OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
sdg q[13];
ch q[0], q[18];
t q[13];
p(0) q[9];
rz(1.5707963267948966) q[11];
cx q[17], q[13];
sdg q[6];
u2(1.5707963267948966, 1.5707963267948966) q[18];
t q[15];
u3(0, 0, 1.5707963267948966) q[13];
t q[19];
h q[10];
ryy(1.5707963267948966) q[3], q[17];
u1(1.5707963267948966) q[16];
ch q[18], q[10];
rzz(1.5707963267948966) q[11], q[8];
u1(1.5707963267948966) q[0];
rxx(0) q[15], q[3];
ry(1.5707963267948966) q[12];
rx(1.5707963267948966) q[17];
swap q[1], q[8];
cz q[16], q[14];
u1(1.5707963267948966) q[6];
rx(1.5707963267948966) q[15];
rx(1.5707963267948966) q[16];
cu3(1.5707963267948966, 0, 0) q[6], q[11];
s q[11];
crz(1.5707963267948966) q[2], q[14];
rx(1.5707963267948966) q[8];
x q[4];
u1(1.5707963267948966) q[19];
swap q[11], q[12];
rzz(1.5707963267948966) q[8], q[10];
crz(1.5707963267948966) q[7], q[5];
sdg q[2];
rx(1.5707963267948966) q[6];
rxx(0) q[4], q[16];
h q[18];
rz(1.5707963267948966) q[17];
sdg q[1];
rxx(0) q[13], q[16];
rx(1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[9];
h q[6];
h q[6];
x q[9];
u3(0, 0, 1.5707963267948966) q[8];
tdg q[3];
s q[14];
cy q[8], q[10];
tdg q[18];
id q[17];
id q[9];
sdg q[9];
ry(1.5707963267948966) q[17];
cy q[19], q[16];
u1(1.5707963267948966) q[14];
u3(0, 0, 1.5707963267948966) q[16];
u1(1.5707963267948966) q[8];
rz(1.5707963267948966) q[8];
t q[15];
rx(1.5707963267948966) q[19];
u1(1.5707963267948966) q[16];
s q[18];
id q[0];
s q[19];
ry(1.5707963267948966) q[11];
ry(1.5707963267948966) q[4];
u1(1.5707963267948966) q[0];
rx(1.5707963267948966) q[12];
p(0) q[16];
x q[0];
t q[15];
rz(1.5707963267948966) q[12];
id q[4];
u2(1.5707963267948966, 1.5707963267948966) q[3];
cy q[6], q[3];
swap q[3], q[1];
x q[3];
id q[19];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[19];
x q[9];
tdg q[12];
tdg q[9];
rxx(0) q[16], q[6];
u3(0, 0, 1.5707963267948966) q[16];
rz(1.5707963267948966) q[18];
t q[8];
x q[0];
sdg q[15];
u1(1.5707963267948966) q[14];
h q[8];
tdg q[1];
ry(1.5707963267948966) q[17];
u3(0, 0, 1.5707963267948966) q[18];
ryy(1.5707963267948966) q[16], q[8];
rx(1.5707963267948966) q[8];