OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
sdg q[11];
cu3(1.5707963267948966, 0, 0) q[3], q[0];
cu1(1.5707963267948966) q[17], q[3];
id q[6];
cy q[1], q[3];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[12];
u1(1.5707963267948966) q[5];
swap q[15], q[2];
tdg q[13];
id q[1];
u1(1.5707963267948966) q[5];
cu1(1.5707963267948966) q[6], q[16];
rzz(1.5707963267948966) q[18], q[19];
ry(1.5707963267948966) q[11];
s q[11];
cy q[11], q[15];
sdg q[1];
sdg q[8];
id q[15];
id q[16];
rx(1.5707963267948966) q[2];
sdg q[7];
tdg q[0];
u1(1.5707963267948966) q[13];
u3(0, 0, 1.5707963267948966) q[3];
tdg q[19];
sdg q[3];
cu3(1.5707963267948966, 0, 0) q[15], q[16];
rx(1.5707963267948966) q[17];
cz q[11], q[18];
swap q[9], q[7];
u3(0, 0, 1.5707963267948966) q[2];
sdg q[4];
rxx(0) q[17], q[19];
cz q[14], q[1];
u1(1.5707963267948966) q[5];
s q[4];
u2(1.5707963267948966, 1.5707963267948966) q[13];
rx(1.5707963267948966) q[19];
u1(1.5707963267948966) q[16];
t q[4];
u1(1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rz(1.5707963267948966) q[19];
tdg q[8];
p(0) q[7];
u3(0, 0, 1.5707963267948966) q[17];
sdg q[14];
ry(1.5707963267948966) q[0];
u1(1.5707963267948966) q[14];
p(0) q[17];
cu1(1.5707963267948966) q[8], q[14];
id q[12];
cz q[17], q[11];
h q[0];
h q[5];
tdg q[15];
rx(1.5707963267948966) q[11];
tdg q[2];
rz(1.5707963267948966) q[6];
s q[5];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[12];
ry(1.5707963267948966) q[4];
cy q[0], q[15];
p(0) q[4];
t q[17];
rzz(1.5707963267948966) q[7], q[16];
id q[7];
ry(1.5707963267948966) q[11];
ry(1.5707963267948966) q[8];
cz q[10], q[13];
sdg q[0];
t q[19];
h q[12];
u1(1.5707963267948966) q[15];
t q[19];
s q[15];
sdg q[9];
u3(0, 0, 1.5707963267948966) q[17];
u1(1.5707963267948966) q[11];
id q[11];
sdg q[3];
sdg q[1];
cu1(1.5707963267948966) q[15], q[2];
u3(0, 0, 1.5707963267948966) q[7];
s q[8];
sdg q[7];
t q[7];
p(0) q[10];
swap q[0], q[11];
id q[3];
cy q[6], q[17];
s q[16];
h q[0];
tdg q[16];
sdg q[14];
rx(1.5707963267948966) q[7];
cu1(1.5707963267948966) q[11], q[9];
tdg q[4];
swap q[16], q[19];
cz q[7], q[16];
p(0) q[4];
p(0) q[19];
id q[13];
p(0) q[13];
rzz(1.5707963267948966) q[12], q[9];
u1(1.5707963267948966) q[5];
rx(1.5707963267948966) q[11];
u3(0, 0, 1.5707963267948966) q[19];
id q[9];
s q[11];
u1(1.5707963267948966) q[6];
cy q[4], q[1];
p(0) q[18];
u3(0, 0, 1.5707963267948966) q[10];
t q[10];
rz(1.5707963267948966) q[16];
rz(1.5707963267948966) q[17];
u2(1.5707963267948966, 1.5707963267948966) q[19];
ry(1.5707963267948966) q[19];
t q[12];
u3(0, 0, 1.5707963267948966) q[6];
h q[15];
cz q[7], q[5];
u3(0, 0, 1.5707963267948966) q[14];
cx q[18], q[0];
id q[15];
u2(1.5707963267948966, 1.5707963267948966) q[16];
sdg q[5];
t q[13];
h q[3];
id q[15];
s q[8];
u1(1.5707963267948966) q[7];
tdg q[3];
p(0) q[16];
h q[19];
s q[1];