OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
id q[3];
p(0) q[14];
p(0) q[14];
s q[7];
u2(1.5707963267948966, 1.5707963267948966) q[4];
rz(1.5707963267948966) q[1];
sdg q[2];
s q[2];
h q[15];
cy q[8], q[9];
cy q[14], q[2];
rx(1.5707963267948966) q[2];
u1(1.5707963267948966) q[18];
h q[19];
u2(1.5707963267948966, 1.5707963267948966) q[17];
id q[11];
ry(1.5707963267948966) q[14];
rxx(0) q[18], q[13];
u1(1.5707963267948966) q[4];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[14];
u1(1.5707963267948966) q[14];
u2(1.5707963267948966, 1.5707963267948966) q[1];
id q[17];
swap q[14], q[8];
t q[1];
cu3(1.5707963267948966, 0, 0) q[13], q[8];
id q[11];
tdg q[3];
ry(1.5707963267948966) q[16];
id q[4];
id q[19];
u3(0, 0, 1.5707963267948966) q[10];
p(0) q[16];
cu3(1.5707963267948966, 0, 0) q[5], q[8];
s q[16];
u2(1.5707963267948966, 1.5707963267948966) q[19];
sdg q[18];
rz(1.5707963267948966) q[10];
u1(1.5707963267948966) q[1];
rz(1.5707963267948966) q[4];
rzz(1.5707963267948966) q[4], q[3];
rz(1.5707963267948966) q[12];
t q[11];
rz(1.5707963267948966) q[10];
id q[5];
u2(1.5707963267948966, 1.5707963267948966) q[10];
p(0) q[3];
cy q[11], q[4];
tdg q[7];
s q[9];
p(0) q[5];
cu1(1.5707963267948966) q[1], q[11];
sdg q[9];
u1(1.5707963267948966) q[18];
h q[17];
id q[2];
u3(0, 0, 1.5707963267948966) q[12];
t q[5];
sdg q[13];
h q[16];
s q[1];
u1(1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[5];
s q[13];
rz(1.5707963267948966) q[5];
p(0) q[8];
p(0) q[16];
cy q[13], q[18];
p(0) q[17];
cu3(1.5707963267948966, 0, 0) q[0], q[17];
u3(0, 0, 1.5707963267948966) q[10];
u1(1.5707963267948966) q[18];
u1(1.5707963267948966) q[15];
sdg q[11];
u1(1.5707963267948966) q[13];
rz(1.5707963267948966) q[14];
h q[19];
cz q[9], q[7];
p(0) q[7];
ry(1.5707963267948966) q[1];
u1(1.5707963267948966) q[1];
t q[18];
sdg q[2];
t q[17];
rx(1.5707963267948966) q[5];
u1(1.5707963267948966) q[18];
rx(1.5707963267948966) q[13];
rz(1.5707963267948966) q[18];
tdg q[9];
t q[0];
sdg q[1];
cz q[4], q[9];
ry(1.5707963267948966) q[13];
rz(1.5707963267948966) q[5];
tdg q[1];
t q[18];
cu3(1.5707963267948966, 0, 0) q[11], q[16];
p(0) q[13];
cz q[15], q[10];
sdg q[1];
cz q[17], q[10];
s q[6];
p(0) q[4];
rx(1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[5];
h q[4];
u2(1.5707963267948966, 1.5707963267948966) q[13];
p(0) q[3];
p(0) q[16];
u3(0, 0, 1.5707963267948966) q[13];
swap q[17], q[5];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[18];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[6];
u1(1.5707963267948966) q[16];
u2(1.5707963267948966, 1.5707963267948966) q[16];
u1(1.5707963267948966) q[2];
sdg q[4];
rxx(0) q[11], q[9];
t q[0];
id q[10];
tdg q[7];
rzz(1.5707963267948966) q[16], q[2];
u1(1.5707963267948966) q[10];
u1(1.5707963267948966) q[12];
sdg q[2];
id q[15];
rx(1.5707963267948966) q[19];
rx(1.5707963267948966) q[11];
p(0) q[3];
h q[4];
u1(1.5707963267948966) q[15];
rz(1.5707963267948966) q[19];
cy q[10], q[3];
rx(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[13];
cu3(1.5707963267948966, 0, 0) q[11], q[3];
tdg q[5];
cz q[4], q[6];
rx(1.5707963267948966) q[15];
rzz(1.5707963267948966) q[14], q[19];
swap q[1], q[15];
sdg q[1];
t q[8];
h q[5];
cu3(1.5707963267948966, 0, 0) q[11], q[7];
u1(1.5707963267948966) q[11];
s q[10];
sdg q[0];
swap q[7], q[3];
rz(1.5707963267948966) q[15];
u3(0, 0, 1.5707963267948966) q[3];
cu1(1.5707963267948966) q[8], q[6];
p(0) q[14];
cu1(1.5707963267948966) q[7], q[14];
id q[12];
tdg q[16];
h q[0];
sdg q[0];
sdg q[15];
s q[7];
id q[4];
rx(1.5707963267948966) q[6];
cx q[6], q[19];
cu1(1.5707963267948966) q[19], q[8];
tdg q[1];
sdg q[19];
tdg q[0];
rxx(0) q[13], q[3];
p(0) q[7];
id q[19];
t q[0];
sdg q[13];
sdg q[7];
rz(1.5707963267948966) q[17];
id q[0];
s q[10];
cu1(1.5707963267948966) q[6], q[17];