OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
u1(1.5707963267948966) q[4];
rx(1.5707963267948966) q[9];
u1(1.5707963267948966) q[10];
rz(1.5707963267948966) q[1];
sdg q[16];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[10];
p(0) q[19];
u1(1.5707963267948966) q[14];
ry(1.5707963267948966) q[17];
u3(0, 0, 1.5707963267948966) q[8];
u1(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[14];
rzz(1.5707963267948966) q[18], q[8];
tdg q[6];
u3(0, 0, 1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[18];
u1(1.5707963267948966) q[16];
t q[6];
sdg q[10];
u3(0, 0, 1.5707963267948966) q[16];
rz(1.5707963267948966) q[3];
t q[17];
id q[3];
h q[8];
rz(1.5707963267948966) q[5];
rx(1.5707963267948966) q[6];
rx(1.5707963267948966) q[7];
p(0) q[17];
rxx(0) q[12], q[14];
rx(1.5707963267948966) q[3];
cu3(1.5707963267948966, 0, 0) q[10], q[18];
rx(1.5707963267948966) q[11];
cu1(1.5707963267948966) q[14], q[0];
tdg q[10];
cu3(1.5707963267948966, 0, 0) q[3], q[9];
u1(1.5707963267948966) q[6];
h q[13];
p(0) q[15];
u1(1.5707963267948966) q[16];
rx(1.5707963267948966) q[18];
h q[13];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[19];
rx(1.5707963267948966) q[17];
tdg q[9];
t q[10];
rz(1.5707963267948966) q[2];
t q[0];
rz(1.5707963267948966) q[0];
cu3(1.5707963267948966, 0, 0) q[3], q[12];
u3(0, 0, 1.5707963267948966) q[9];
t q[15];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[3];
h q[6];
tdg q[3];
rzz(1.5707963267948966) q[8], q[3];
p(0) q[16];
cx q[3], q[0];
rz(1.5707963267948966) q[15];
u3(0, 0, 1.5707963267948966) q[15];
sdg q[7];
u2(1.5707963267948966, 1.5707963267948966) q[2];
cy q[12], q[1];
t q[15];
id q[10];
s q[13];
rzz(1.5707963267948966) q[11], q[14];
t q[18];
u3(0, 0, 1.5707963267948966) q[16];
u3(0, 0, 1.5707963267948966) q[5];
cy q[15], q[17];
sdg q[14];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[3];
tdg q[2];
rx(1.5707963267948966) q[4];
p(0) q[3];
tdg q[18];
cy q[18], q[9];
rz(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[9];
t q[1];
rxx(0) q[13], q[10];
h q[8];
ry(1.5707963267948966) q[15];
rz(1.5707963267948966) q[4];
rzz(1.5707963267948966) q[3], q[14];
rx(1.5707963267948966) q[19];
sdg q[13];
rx(1.5707963267948966) q[8];
rz(1.5707963267948966) q[7];
ry(1.5707963267948966) q[1];
swap q[5], q[11];
s q[4];
cx q[17], q[19];
tdg q[17];
u2(1.5707963267948966, 1.5707963267948966) q[12];
ry(1.5707963267948966) q[13];
ry(1.5707963267948966) q[3];
cz q[16], q[5];
sdg q[2];
h q[8];
cy q[19], q[10];
u2(1.5707963267948966, 1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[13];
u3(0, 0, 1.5707963267948966) q[8];
rx(1.5707963267948966) q[7];
tdg q[0];
ry(1.5707963267948966) q[6];
sdg q[11];
t q[11];
p(0) q[18];
ry(1.5707963267948966) q[12];
p(0) q[13];
h q[0];
ry(1.5707963267948966) q[2];
swap q[9], q[7];
u1(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[1];
rz(1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[1];
tdg q[19];
ry(1.5707963267948966) q[16];
u3(0, 0, 1.5707963267948966) q[10];
s q[12];
tdg q[18];
u2(1.5707963267948966, 1.5707963267948966) q[10];
sdg q[6];
cy q[18], q[5];
p(0) q[13];
tdg q[6];
rx(1.5707963267948966) q[17];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[14];
t q[1];
s q[10];
u1(1.5707963267948966) q[0];
cy q[12], q[2];
t q[8];
h q[12];
rx(1.5707963267948966) q[17];
id q[12];
u2(1.5707963267948966, 1.5707963267948966) q[15];
cy q[3], q[10];
ry(1.5707963267948966) q[7];
p(0) q[7];
ry(1.5707963267948966) q[13];
rx(1.5707963267948966) q[0];
sdg q[19];
ry(1.5707963267948966) q[5];
s q[13];
rz(1.5707963267948966) q[19];
id q[8];
cu1(1.5707963267948966) q[9], q[19];
p(0) q[17];
t q[2];
rx(1.5707963267948966) q[0];
sdg q[4];
rz(1.5707963267948966) q[9];
cy q[17], q[10];
h q[0];
h q[12];
swap q[7], q[15];
tdg q[3];
u3(0, 0, 1.5707963267948966) q[17];
rz(1.5707963267948966) q[7];
p(0) q[2];
p(0) q[9];
u3(0, 0, 1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[6];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[13];
tdg q[19];
sdg q[0];
id q[12];
s q[14];
id q[3];